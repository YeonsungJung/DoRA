import os
import glob
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import combinations

from data import load_dataset, DATASET2CLSNUM, imagenet_templates
from core import clip, loralib, losses, utils
from utils import *



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arch", type=str, default="CLIP")
    parser.add_argument("--dataset", type=str, default="waterbirds")
    parser.add_argument("--n_cls", type=int, default=2)
    parser.add_argument("--prompt_id", type=int, default=0)
    
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--train", action="store_true")
    
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./experiments/models/CLIP@SepLoRA@q_v@r4")
    parser.add_argument("--sim_dir", type=str, default="./experiments/desc_sim")
    
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--org", action="store_true")
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()
    
    
    if args.infer:
        try:
            train_config = read_json(args.save_dir + '/config.json')
            for k, v in train_config.items(): 
                if k not in ["data_dir", "save_dir", "batch_size", "epochs"]: setattr(args, k, v)
        except:
            print("Train config is not provided.")
        
        utils.set_seed(args.seed)
        
        lora_idxs = list(range(args.num_lora))
        lora_pairs = list(combinations(lora_idxs, 2))
        lora_modules = [m for m in args.lora_modules.split(',') if m in ['q', 'k', 'v', 'out', 'mlp']]
        
        ## Load data and model
        if args.model == "CLIP":
            model = clip.CLIP_FT(args.arch, "cuda", n_cls=args.n_cls).eval()
        else:
            raise NotImplementedError(f'{args.model} is not implemented yet.')
        model = nn.DataParallel(model)
        
        split = "train" if args.train else "test"
        dataset = load_dataset(args.data_dir, args.dataset, split, model.module.preprocess, args.prompt_id)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
        
        all_descs = [[template.format(desc) for template in imagenet_templates] for desc in dataset.all_descs]
        all_descs = [model.module.model.encode_text(clip.tokenize(desc).to("cuda")) for desc in all_descs]
        all_descs = [desc / desc.norm(dim=-1, keepdim=True) for desc in all_descs]
        all_descs = [desc.mean(dim=0) for desc in all_descs]
        all_descs = [desc / desc.norm() for desc in all_descs]
        desc_feats = torch.stack(all_descs, dim=0).detach()
        
        target_layers = [i for i in range(model.module.n_layers-args.last_num, model.module.n_layers)]
        ortho_feat_loss_fn = losses.OrthoFeatLoss(lora_pairs, args=args, save_dist=True)
        model.module.set_desc_loss_fn(target_layers, desc_feats, ortho_feat_loss_fn)
        
        def extract_desc(model, sim_dir):
            os.makedirs(sim_dir, exist_ok=True)
            
            for idx, data in enumerate(tqdm(dataloader, desc="extracting desc sim")):
                if args.dataset == 'waterbirds':
                    images, attrs, _ = data
                    labels = attrs[:,0]
                    spurious = attrs[:,1]
                elif args.dataset == 'imagenet':
                    images, labels = data['images'], data['labels']
                    spurious = labels
                
                ortho_feat_loss_fn.dists = []
                outputs, _, _ = model(images.to("cuda"))
                _, preds = torch.max(outputs, 1)
                
                result = {
                    "features": ortho_feat_loss_fn.dists,
                    "preds": preds,
                    "labels": labels,
                    "spurious": spurious,
                }
                import pdb;pdb.set_trace()
                torch.save(result, f"{sim_dir}/{idx}.pt")
                if idx==20: break
        
        if args.org:
            extract_desc(model, f"{args.sim_dir}/CLIP")
            
        loralib.apply_lora(model.module, args.num_lora, args.r, args.lora_alpha, args.lora_dropout, lora_modules)
        loralib.load_lora(model.module, args.save_dir + f'/epoch{args.epochs}.pt')
        loralib.set_used_lora(model.module, lora_idxs)
        model.eval()
        
        sim_dir = f"{args.sim_dir}/{args.save_dir.split('/')[-1]}_epoch{args.epochs}"
        sim_dir += "_train" if args.train else "_test"
        extract_desc(model, sim_dir)


    if args.vis:
        sim_dir = f"{args.sim_dir}/{args.save_dir.split('/')[-1]}_epoch{args.epochs}"
        sim_dir += "_train" if args.train else "_test"
        save_dir = sim_dir.replace('desc_sim/', 'desc_sim_vis/')
        os.makedirs(save_dir, exist_ok=True)
        
        split = "train" if args.train else "test"
        dataset = load_dataset(args.data_dir, args.dataset, split, model.module.preprocess, args.prompt_id)
        classes = dataset.classnames
        class_descs = dataset.class_descs
        all_descs = dataset.all_descs
        
        def find_set(desc):
            check = [i+1 for i, k in enumerate(classes) if desc in class_descs[k]]
            if len(check)==1: return check[0]
            else: return 0
        
        set_list = [find_set(desc) for desc in all_descs]
        idxs = np.argsort(set_list)
        color_dict = {0:'r', 1:'b', 2:'g'}
        color_list = [color_dict[set_list[i]] for i in idxs]
        all_descs = [all_descs[i] for i in idxs]
        
        result_paths = glob.glob(f"{sim_dir}/*.pt")
        for path in tqdm(result_paths):
            idx = path.split('/')[-1][:-3]
            data = torch.load(path)
            features = data["features"]
            preds = data["preds"]
            labels = data["labels"]
            spurious = data["spurious"]
            
            correct = preds == labels
            
            n = len(features)
            feat_labels = list(range(24-n,24))
            
            for feat_idx in range(n):
                sims = [features[feat_idx][i][:, idxs] for i in range(args.num_lora)]  # [bsz, num_desc]
                label = feat_labels[feat_idx]
                
                for i in range(sims[0].shape[0]):
                    tmp_sims = [sim[i] for sim in sims]
                    y_min = min([sim.min().item() for sim in sims])
                    y_max = max([sim.max().item() for sim in sims])
                    
                    fig, axs = plt.subplots(2, 2)
                    for j, sim in enumerate(tmp_sims):
                        data = pd.Series(
                            sim.tolist(),
                            index=all_descs
                        )
                        data.plot( 
                            kind='bar', 
                            ax = axs[j//2, j%2],
                            color=color_list,
                        )
                        axs[j//2, j%2].set_title(f"lora {j}")
                        axs[j//2, j%2].set_ylim(y_min, y_max)
                        axs[j//2, j%2].tick_params(axis='x', labelsize=2.5)
                    
                    for ax in fig.get_axes():
                        ax.label_outer()
                    
                    title = f"{label} | {'correct' if correct[i] else 'wrong'} | {'landbird' if labels[i]==0 else 'waterbird'} | {'land' if spurious[i]==0 else 'water'}"
                    fig.suptitle(title)
                    plt.subplots(constrained_layout=True)
                    plt.tight_layout()
                    fig.savefig(f"{save_dir}/{title.replace(' | ', '_')}_{idx}_{i}.png", dpi=250)
                    plt.close()