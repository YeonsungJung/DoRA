import os
import argparse
from tqdm import tqdm
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from itertools import combinations
from collections import defaultdict
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

import wandb

from data import load_dataset, DATASET2CLSNUM, imagenet_templates
from core import clip, loralib, losses, utils
from core.hooker import Hooker
from eval import load_test_configs, load_val_configs
from utils import *



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default="CLIP")
    parser.add_argument("--arch", type=str, default="ViT-B/16")
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--prompt_id", type=int, default=9)
    
    
    # lora
    parser.add_argument("--r", type=int, default=2)
    parser.add_argument("--num_lora", type=int, default=2)
    parser.add_argument("--lora_alpha", type=float, default=1.)
    parser.add_argument("--lora_dropout", type=float, default=0.)
    parser.add_argument("--lora_modules", type=str, default="q,k,v,out")
    parser.add_argument("--lora_w_pretrain", action="store_true")
    parser.add_argument("--train_visual_proj", action="store_true")
    
    # orthogonality
    parser.add_argument("--last_num", type=int, default=24)
    parser.add_argument("--lambda_cls", type=float, default=1.)
    parser.add_argument("--lambda_desc_ortho", type=float, default=0.)
    parser.add_argument("--lambda_feat_ortho", type=float, default=1.)
    parser.add_argument("--lambda_param_ortho", type=float, default=0.)
    parser.add_argument("--only_wA", action="store_true")
    parser.add_argument("--compare_org", action="store_true")
    parser.add_argument("--kl", action="store_true")
    parser.add_argument("--dot", action="store_true")
    parser.add_argument("--feat_kk", action="store_true")
    parser.add_argument("--only_qk", action="store_true")
    parser.add_argument("--loss_gram", action="store_true")
    # 
    parser.add_argument("--text_cls", action="store_true")

    # sparsity regularization
    parser.add_argument("--lambda_reg", type=float, default=0.1)
    parser.add_argument("--entropy", action="store_true")
    parser.add_argument("--l1", action="store_true")
    
    #gating
    parser.add_argument("--gating", action="store_true")

    # train
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--optim", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=5e-5)
    parser.add_argument("--lr_schedule", action="store_true")
    
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default=f"./experiments/model/CLIP@LoRA_feat1.0_cosine@r2_num2_qkvout")
    
    parser.add_argument("--resume_id", type=str, default="")
    args = parser.parse_args()
    
    utils.set_seed(args.seed)
    args.n_cls = DATASET2CLSNUM[args.dataset]
    
    lora_idxs = list(range(args.num_lora))
    lora_pairs = list(combinations(lora_idxs, 2))
    lora_modules = [m for m in args.lora_modules.split(',') if m in ['q', 'k', 'v', 'out', 'mlp']]

    ## Load model
    if args.model == "CLIP":
        classnames = load_dataset(args.data_dir, args.dataset, "train", None, args.prompt_id).classnames
        model = clip.CLIP_FT(args.arch, "cuda", classnames=classnames, n_cls=args.n_cls, text_cls=args.text_cls)

    else:
        raise NotImplementedError(f'{args.model} is not implemented yet.')
    print('{} w/o LoRA: {:.1f}M'.format(args.model, sum(param.numel() for param in model.parameters())/1000000.0))
    
    cls_loss_fn = nn.CrossEntropyLoss()
    ortho_feat_loss_fn = losses.OrthoFeatLoss(lora_pairs, args=args)
    ortho_param_loss_fn = losses.OrthoParamLoss(lora_pairs, args.compare_org)
    if args.only_wA and args.compare_org:
        raise NotImplementedError('We cannot compare wA with the original weight.')
    
    ## Load data
    train_dataset = load_dataset(args.data_dir, args.dataset, "train", model.preprocess, args.prompt_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)

    valid_loader = load_val_configs(args, model.preprocess)
    infer, evaluate, test_datasets, test_loaders, projection_fns = load_test_configs(args, model.preprocess)

    delattr(model.model, 'transformer')

    if args.only_qk:
        num_lora = {"q": args.num_lora, "k": args.num_lora, "v": 1, "out": 1}
    else:
        num_lora = {key: args.num_lora for key in lora_modules}

    loralib.apply_lora(model, num_lora, args.r, args.lora_alpha, args.lora_dropout, lora_modules, ortho_feat_loss_fn, args.lora_w_pretrain, gating=args.gating)
    print('{} w/  LoRA: {:.1f}M'.format(args.model, sum(param.numel() for param in model.parameters())/1000000.0))
    hooker = Hooker(model, num_lora=args.num_lora)
    
    ## Train
    loralib.set_used_lora(model, lora_idxs)
    model = nn.DataParallel(model)
    
    latest_model_path = os.path.join(args.save_dir, "latest_model.pt")
    print(f"Resuming training from {latest_model_path}")
    loralib.load_lora(model.module, latest_model_path)

    latest_optimizer_path = os.path.join(args.save_dir, "latest_optimizer.pt")
    checkpoint = torch.load(latest_optimizer_path)
    epoch = checkpoint["epoch"] 

    os.makedirs(f'{args.save_dir}/heatmaps/{args.dataset}/epoch_{epoch}/', exist_ok=True)

    print(model.module.model.visual.input_resolution)
    transform_image = Compose([
        Resize(model.module.model.visual.input_resolution, interpolation=Image.BICUBIC),
        CenterCrop(model.module.model.visual.input_resolution),
        lambda image: image.convert("RGB"),
    ])
    
    def minmax(t):
        t_min = t.min()
        t_max = t.max()
        t -= t_min
        t /= t_max
        return t, t_min, t_max
    
    def draw_heatmap(imgs, classnames, attns, up_factor, reshape_size):
        for i, img in enumerate(imgs):
            attn = {k:[a[i].reshape(reshape_size,reshape_size) for a in v] for k,v in attns.items()}

            classname = classnames[i]
            for layer, tmp_attn in attn.items():
                tmp_attn = [tmp_attn[0], np.abs(tmp_attn[1]-tmp_attn[0]), np.abs(tmp_attn[2]-tmp_attn[0]), np.abs(tmp_attn[1]-tmp_attn[2]), np.abs(tmp_attn[3]-tmp_attn[0])]
                min_max_values = [minmax(tmp) for tmp in tmp_attn]
                tmp_attn = [item[0] for item in min_max_values]
                min_vals = [item[1] for item in min_max_values]
                max_vals = [item[2] for item in min_max_values]

                # tmp_attn = [tmp_attn[0], tmp_attn[1]-tmp_attn[0], tmp_attn[2]-tmp_attn[0], tmp_attn[1]+tmp_attn[2]-tmp_attn[0]-tmp_attn[0]]

                tmp_attn = [np.repeat(np.repeat(tmp.numpy(), up_factor, axis=0), up_factor, axis=1) for tmp in tmp_attn]
                tmp_attn = [cv2.applyColorMap(np.uint8(255 * tmp), cv2.COLORMAP_JET) for tmp in tmp_attn]
                tmp_attn = [cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB) for tmp in tmp_attn]
                tmp_attn = [cv2.addWeighted(img, 0.6, tmp, 0.4, 0) for tmp in tmp_attn]
                
                fig = plt.figure(figsize=[20, 5])
                fig.suptitle(f"Classname: {classname}", fontsize=16, fontweight='bold')
                ax = fig.add_subplot(1, 5, 1)
                ax.axis("off")
                ax.imshow(tmp_attn[0])
                ax.set_title("Pretrained")
                ax.text(0.5, 1.08, f"Min: {min_vals[0]:.8f}, Max: {max_vals[0]:.8f}",
                        transform=ax.transAxes, fontsize=10, ha="center", va="bottom", color="black")

                ax = fig.add_subplot(1, 5, 2)
                ax.axis("off")
                ax.imshow(tmp_attn[4])
                ax.set_title("Pretrained_LoRA1,2-pretrained")
                ax.text(0.5, 1.08, f"Min: {min_vals[4]:.8f}, Max: {max_vals[4]:.8f}",
                        transform=ax.transAxes, fontsize=10, ha="center", va="bottom", color="black")

                ax = fig.add_subplot(1, 5, 3)
                ax.axis("off")
                ax.imshow(tmp_attn[1])
                ax.set_title("|LoRA1-Pretrained|")
                ax.text(0.5, 1.08, f"Min: {min_vals[1]:.8f}, Max: {max_vals[1]:.8f}",
                        transform=ax.transAxes, fontsize=10, ha="center", va="bottom", color="black")
                
                ax = fig.add_subplot(1, 5, 4)
                ax.axis("off")
                ax.imshow(tmp_attn[2])
                ax.set_title("|LoRA2-Pretrained|")
                ax.text(0.5, 1.08, f"Min: {min_vals[2]:.8f}, Max: {max_vals[2]:.8f}",
                        transform=ax.transAxes, fontsize=10, ha="center", va="bottom", color="black")
                
                ax = fig.add_subplot(1, 5, 5)
                ax.axis("off")
                ax.imshow(tmp_attn[3])
                ax.set_title("|LoRA1-LoRA2|")
                ax.text(0.5, 1.08, f"Min: {min_vals[3]:.8f}, Max: {max_vals[3]:.8f}",
                        transform=ax.transAxes, fontsize=10, ha="center", va="bottom", color="black")
                # fig.subplots_adjust(hspace=0, wspace=0)

                fig.savefig(f"./{args.save_dir}/heatmaps/{args.dataset}/epoch_{epoch}/{i}_L{layer}.png")

                plt.close()
    
    if args.dataset == 'waterbirds':
        up_factor = 14
        reshape_size = 16
    elif args.dataset == 'imagenet':
        up_factor = 16 
        reshape_size = 14

    for data in tqdm(train_loader,  ncols=100):
        if args.dataset == 'waterbirds':
            images, attrs, _, img_paths = data
            labels = attrs[:,0]
            spurious = attrs[:,1]
        elif args.dataset == 'imagenet':
            images, labels, img_paths = data['images'], data['labels'], data['image_paths']
            spurious = labels
        
        outputs, _, _ = model(images.to("cuda"))
        attns = hooker.data.attn_weights
        
        
        
        if args.dataset == 'waterbirds':
            raw_imgs = [transform_image(Image.open(img_path).convert('RGB')) for img_path in img_paths[0]]
        elif args.dataset == 'imagenet':
            raw_imgs = [transform_image(Image.open(img_path).convert('RGB')) for img_path in img_paths]
        raw_imgs = [np.array(img).astype(np.uint8) for img in raw_imgs]
        
        


        draw_heatmap(raw_imgs, [classnames[label] for label in labels], attns, up_factor, reshape_size)
        
        # import pdb; pdb.set_trace()
        break
