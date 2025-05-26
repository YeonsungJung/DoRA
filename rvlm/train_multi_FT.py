import os
import argparse
from tqdm import tqdm
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import datasets

from itertools import combinations
from collections import defaultdict

import wandb

from data import load_dataset, DATASET2CLSNUM, imagenet_templates
from core import clip, loralib, losses, utils
from eval import load_test_configs
from utils import *



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default="CLIP")
    parser.add_argument("--arch", type=str, default="ViT-L/14")
    parser.add_argument("--dataset", type=str, default="waterbirds")
    parser.add_argument("--prompt_id", type=int, default=0)
    
    # lora
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--num_lora", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=1.)
    parser.add_argument("--lora_dropout", type=float, default=0.)
    parser.add_argument("--lora_modules", type=str, default="q,k,v,out")
    parser.add_argument("--lora_w_pretrain", action="store_true")
    parser.add_argument("--train_visual_proj", action="store_false")
    
    # orthogonality
    parser.add_argument("--last_num", type=int, default=24)
    parser.add_argument("--lambda_cls", type=float, default=1.)
    parser.add_argument("--lambda_desc_ortho", type=float, default=0.)
    parser.add_argument("--lambda_feat_ortho", type=float, default=0.)
    parser.add_argument("--lambda_param_ortho", type=float, default=0.)
    parser.add_argument("--only_wA", action="store_true")
    parser.add_argument("--compare_org", action="store_true")
    parser.add_argument("--kl", action="store_true")
    parser.add_argument("--dot", action="store_true")

    # sparsity regularization
    parser.add_argument("--lambda_reg", type=float, default=0.1)
    parser.add_argument("--entropy", action="store_true")
    parser.add_argument("--l1", action="store_true")
    
    #gating
    parser.add_argument("--gating", action="store_true")

    # train
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--optim", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=5e-5)
    parser.add_argument("--lr_schedule", action="store_true")
    
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./experiments/models/CLIP@LoRA")
    
    parser.add_argument("--resume_id", type=str, default="")
    args = parser.parse_args()
    
    ## Set ENV
    utils.set_seed(args.seed)
    args.n_cls = DATASET2CLSNUM[args.dataset]
    #args.n_cls = 10
       
    save_dir = args.save_dir

    if args.train_visual_proj:
        save_dir += f"{'_'.join('w/proj')}"
    else:
        save_dir += f"{'_'.join('w/oproj')}"
    os.makedirs(save_dir, exist_ok=True)
    write_json(f"{save_dir}config.json", vars(args))
    f = open(f"{save_dir}log.txt", 'a')
    
    if args.resume_id:
        wandb.init(project="rvlm", id=args.resume_id, resume=True)
    else:
        wandb.init(project="rvlm")
    wandb.config.update(args)
    wandb.run.name = save_dir.split('/')[-2]
    
    ## Load model
    if args.model == "CLIP":
        model = clip.CLIP_FT_tmp(args.arch, "cuda", n_cls=args.n_cls)
    else:
        raise NotImplementedError(f'{args.model} is not implemented yet.')
    print('{} w/o LoRA: {:.1f}M'.format(args.model, sum(param.numel() for param in model.parameters())/1000000.0))
    
    cls_loss_fn = nn.CrossEntropyLoss()
    
    ## Load data
    #train_dataset = datasets.CIFAR10(root="./data/", train=True, download=True, transform=model.preprocess)
    #test_dataset = datasets.CIFAR10(root="./data/", train=False, download=True, transform=model.preprocess)

    #train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)
    #test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    
     # from eval import infer_cifar, evaluate_cls
    # infer, evaluate = infer_cifar, evaluate_cls
    # test_datasets, test_loaders, projection_fns = ["cifar10"], [test_loader], [None]

    train_dataset = load_dataset(args.data_dir, args.dataset, "train", model.preprocess, args.prompt_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)
    infer, evaluate, test_datasets, test_loaders, projection_fns = load_test_configs(args, model.preprocess)

    for name, param in model.named_parameters():
        if "model.proj" in name:
            param.requires_grad = False


    ## Train
    wandb.define_metric("step/iter")
    wandb.define_metric("step/*", step_metric="step/iter")
    #loralib.set_used_lora(model, lora_idxs)
    model = nn.DataParallel(model)
    model.train()
    
    if args.optim=="adamw": optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.wd)
    elif args.optim=="sgd": optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    if args.lr_schedule: scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs*len(train_loader))
    
    iteration = 0
    for epoch in range(1, args.epochs+1):        
        for data in tqdm(train_loader, desc=f'Epoch: {epoch:03d}', ncols=100):
            if args.dataset == 'waterbirds':
                images, attrs, _ = data
                labels = attrs[:,0]
                spurious = attrs[:,1]
            elif args.dataset == 'imagenet':
                images, labels = data['images'], data['labels']
                spurious = labels
            else:
                images, labels = data
                spurious = labels
            
            outputs, desc_loss, feat_loss = model(images.to("cuda"))
            
            loss = cls_loss_fn(outputs, labels.to("cuda"))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.lr_schedule: scheduler.step()
            
            _, preds = torch.max(outputs, 1)
            train_worst_acc, train_avg_acc, _ = evaluate(preds.detach().cpu().numpy(), labels.numpy(), spurious.numpy())

            iteration += 1
            wandb.log({
                "step/iter": iteration,
                "step/loss": loss.item(),
                "step/train_worst_acc": train_worst_acc,
                "step/train_avg_acc": train_avg_acc,
                "step/lr" : optimizer.param_groups[0]['lr'],
            })
    
        # Evaluation on test sets
        model.eval()
        for test_dataset, test_loader, projection_fn in zip(test_datasets, test_loaders, projection_fns):
            with torch.no_grad():
                worst_acc, avg_acc, accs_by_group = evaluate(*infer(model, test_loader, projection_fn, desc="Eval CLIP+LoRA" + f" ({test_dataset})"), str=True)
            f.write(f"Epoch {epoch}) Test Set ({test_dataset}) - Average accuracy: {avg_acc} | Worst Group accuracy: {worst_acc} | Acc by group: {accs_by_group}\n")
            print(f"Epoch {epoch}) Test Set ({test_dataset}) - Average accuracy: {avg_acc} | Worst Group accuracy: {worst_acc} | Acc by group: {accs_by_group}\n")
        model.train()
    
    wandb.finish()
    f.write('\n')
    f.close()