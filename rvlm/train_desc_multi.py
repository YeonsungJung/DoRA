import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from itertools import combinations
from collections import defaultdict

import wandb

from data import load_dataset, imagenet_templates
from core import clip, loralib, losses, utils
from eval import evaluate, infer, evaluate_IN, infer_IN
from utils import *



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arch", type=str, default="CLIP")
    parser.add_argument("--dataset", type=str, default="waterbird", choices=['waterbird', 'celeba', 'imagenet'])
    parser.add_argument("--n_cls", type=int, default=2)
    parser.add_argument("--prompt_id", type=int, default=6)
    
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--num_lora", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=1.)
    parser.add_argument("--lora_dropout", type=float, default=0.)
    parser.add_argument("--lora_modules", type=str, default="q,k,v,out")
    parser.add_argument("--lora_w_pretrain", action="store_true")

    # orthogonality
    parser.add_argument("--last_num", type=int, default=24)
    parser.add_argument("--feat_ortho", action="store_true")
    parser.add_argument("--kl", action="store_true")
    parser.add_argument("--dot", action="store_true")
    parser.add_argument("--lambda_ortho", type=float, default=1.)

    # sparsity reg.
    parser.add_argument("--entropy", action="store_true")
    parser.add_argument("--l1", action="store_true")
    parser.add_argument("--lambda_reg", type=float, default=0.1)

    # 
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--optim", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=5e-5)
    parser.add_argument("--lambda_cls", type=float, default=1.)
    
    parser.add_argument("--data_dir", type=str, default="../../rvlm_ys/data")
    parser.add_argument("--save_dir", type=str, default="./experiments/models/CLIP@LoRA_desc_multi")
    
    parser.add_argument("--resume_id", type=str, default="")
    args = parser.parse_args()
    
    ##IN
    if args.dataset == 'imagenet':
        args.n_cls = 1000
        args.batch_size = 256
        args.lr = 1e-5
        args.wd = 0.1
        args.epochs = 10
        args.last_num = 12

    ## Set ENV
    utils.set_seed(args.seed)
    
    lora_idxs = list(range(args.num_lora))
    lora_pairs = list(combinations(lora_idxs, 2))
    lora_modules = [m for m in args.lora_modules.split(',') if m in ['q', 'k', 'v', 'out', 'mlp']]
    
    save_dir = args.save_dir
    save_dir += f"@{'_'.join(lora_modules)}"
    if args.lora_w_pretrain: save_dir += "@wp"
    save_dir += f"@r{args.r}/"
    os.makedirs(save_dir, exist_ok=True)
    write_json(f"{save_dir}config.json", vars(args))
    f = open(f"{save_dir}log.txt", 'a')
    
    if args.resume_id:
        wandb.init(project="rvlm", id=args.resume_id, resume=True)
    else:
        wandb.init(project="rvlm")
    wandb.config.update(args)
    wandb.run.name = save_dir.split('/')[-2]
    
    ## Load data and model
    if args.arch == "CLIP":
        if args.dataset == 'imagenet':
            model = clip.CLIP_FT("ViT-B/16", "cuda", n_cls=args.n_cls)
        else:    
            model = clip.CLIP_FT("ViT-L/14", "cuda", n_cls=args.n_cls)
    else:
        raise NotImplementedError(f'{args.arch} is not implemented yet.')
    print('{} w/o LoRA: {:.1f}M'.format(args.arch, sum(param.numel() for param in model.parameters())/1000000.0))
    

    if args.feat_ortho:
        loralib.apply_lora(model, args.num_lora, args.r, args.lora_alpha, args.lora_dropout, lora_modules, feat_loss_fn=losses.OrthoFeatLoss(lora_pairs, args=args))
        
    else:
        loralib.apply_lora(model, args.num_lora, args.r, args.lora_alpha, args.lora_dropout, lora_modules)
    print('{} w/  LoRA: {:.1f}M'.format(args.arch, sum(param.numel() for param in model.parameters())/1000000.0))
    
    train_dataset = load_dataset(args.data_dir, args.dataset, "train", model.preprocess, args.prompt_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)

    if args.dataset == 'imagenet':
        test_dataset_v2 = load_dataset(args.data_dir, 'imagenet-v2', "test", model.preprocess, args.prompt_id)
        test_loader_v2 = DataLoader(test_dataset_v2, batch_size=64, shuffle=False, drop_last=False, num_workers=args.num_workers)

        test_dataset_r = load_dataset(args.data_dir, 'imagenet-r', "test", model.preprocess, args.prompt_id)
        test_loader_r = DataLoader(test_dataset_r, batch_size=64, shuffle=False, drop_last=False, num_workers=args.num_workers)

        test_dataset_a = load_dataset(args.data_dir, 'imagenet-a', "test", model.preprocess, args.prompt_id)
        test_loader_a = DataLoader(test_dataset_a, batch_size=64, shuffle=False, drop_last=False, num_workers=args.num_workers)

        test_dataset_s = load_dataset(args.data_dir, 'imagenet-sketch', "test", model.preprocess, args.prompt_id)
        test_loader_s = DataLoader(test_dataset_s, batch_size=64, shuffle=False, drop_last=False, num_workers=args.num_workers)

    else:
        test_dataset = load_dataset(args.data_dir, args.dataset, "test", model.preprocess, args.prompt_id)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=args.num_workers)

    cls_loss_fn = nn.CrossEntropyLoss()
    ortho_loss_fn = losses.OrthoFeatLoss(lora_pairs, args=args)
    
    ## Train
    wandb.define_metric("step/iter")
    wandb.define_metric("step/*", step_metric="step/iter")
    loralib.set_used_lora(model, lora_idxs)
    _, trainable_params = loralib.get_lora_params(model, fc=True, idxs=lora_idxs, train_visual_proj=args.train_visual_proj)
    
    if args.optim=="adamw": optimizer = optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.wd)
    elif args.optim=="sgd": optimizer = optim.SGD(trainable_params, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    
    if args.dataset == "imagenet":
        all_descs = [[template.format(f"object with {desc.lower()}") for template in imagenet_templates] for desc in train_dataset.all_descs]
    elif args.dataset == "waterbird":
        all_descs = [[template.format(f"bird with {desc.lower()}") for template in imagenet_templates] for desc in train_dataset.all_descs]

    all_descs = [model.model.encode_text(clip.tokenize(desc).to("cuda")) for desc in all_descs]
    all_descs = [desc / desc.norm(dim=-1, keepdim=True) for desc in all_descs]
    all_descs = [desc.mean(dim=0) for desc in all_descs]
    all_descs = [desc / desc.norm() for desc in all_descs]
    desc_feats = torch.stack(all_descs, dim=0).cuda().detach()
    

    if args.dataset == 'imagenet':
        target_layers = [i for i in range(12-args.last_num, 12)]
    else:
        target_layers = [i for i in range(24-args.last_num, 24)]
    if not args.feat_ortho:
        model.set_desc_loss_fn(target_layers, desc_feats, ortho_loss_fn, args.l1)
    model = nn.DataParallel(model).cuda()

    iteration = 0
    for epoch in range(1, args.epochs+1):
        if os.path.exists(save_dir + f'epoch{epoch}.pt'):
            loralib.load_lora(model.module, save_dir + f'epoch{epoch}.pt')
            optimizer.load_state_dict(torch.load(save_dir + f'epoch{epoch}_op.pt'))
            iteration = epoch * len(train_loader)
            continue
        
        for data in tqdm(train_loader, desc=f'Epoch: {epoch:03d}', ncols=100):

            if args.dataset == 'imagenet':
                images = data['images'].to("cuda")
                labels = data['labels'].to("cuda")
                bias_labels = labels
            else:
                images, attrs, _ = data
                labels = attrs[:,0]
                bias_labels = attrs[:,1]
            
            outputs, ortho_loss, feat_loss = model(images.to("cuda"))

            if not args.feat_ortho:
                ortho_loss = ortho_loss.mean()
            else:
                ortho_loss = feat_loss.mean()
            cls_loss = cls_loss_fn(outputs, labels.to("cuda"))
            loss = args.lambda_cls * cls_loss + args.lambda_ortho * ortho_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)

            if args.dataset == 'imagenet':
                train_avg_acc = evaluate_IN(preds.detach().cpu().numpy(), labels.cpu().numpy())
                wandb.log({
                    "step/iter": iteration,
                    "step/loss": loss.item(),
                    "step/loss_cls": cls_loss.item(),
                    "step/loss_ortho": ortho_loss.item(),
                    "step/train_avg_acc": train_avg_acc,
                })
            else:
                train_worst_acc, train_avg_acc, _ = evaluate(preds.detach().cpu().numpy(), labels.cpu().numpy(), bias_labels.cpu().numpy())
            iteration += 1

        loralib.save_lora(model.module, save_dir + f'epoch{epoch}.pt', idxs=lora_idxs, train_visual_proj=args.train_visual_proj)
        torch.save(optimizer.state_dict(), save_dir + f'epoch{epoch}_op.pt')

        # Evaluation on test set
        model.eval()
        if args.dataset == 'imagenet':
            for idx, test_loader in enumerate([test_loader_v2, test_loader_r, test_loader_a, test_loader_s]):
                dataset_name = ["IN_v2", "IN_r", "IN_a", "IN_s"][idx]
                with torch.no_grad():
                    avg_acc = evaluate_IN(*infer_IN(model, test_loader, desc=f"Eval {dataset_name} CLIP+LoRA"))
                if idx == 0:
                    f.write(f"Epoch {epoch} - Avg. Acc.\n")
                    print(f"Epoch {epoch} - Avg. Acc.")
                
                f.write(f"{dataset_name} : {avg_acc:.2f}\n")
                print(f"{dataset_name} : {avg_acc:.2f}")
        else:
            with torch.no_grad():
                worst_acc, avg_acc, accs_by_group = evaluate(*infer(model, test_loader, desc="Eval CLIP+LoRA"))
            f.write(f"Epoch {epoch}) Test Set - Average Accuracy: {avg_acc:.2f}, Worst Group Accuracy: {worst_acc:.2f}\n")
            f.write(f"[{accs_by_group[0]:.2f}, {accs_by_group[1]:.2f}, {accs_by_group[2]:.2f}, {accs_by_group[3]:.2f}]\n\n")
            print(f"Epoch {epoch}) Test Set - Average Accuracy: {avg_acc:.2f}, Worst Group Accuracy: {worst_acc:.2f}")
            print(f"[{accs_by_group[0]:.2f}, {accs_by_group[1]:.2f}, {accs_by_group[2]:.2f}, {accs_by_group[3]:.2f}]")
        model.train()
    
    wandb.finish()
    f.write('\n')
    f.close()
