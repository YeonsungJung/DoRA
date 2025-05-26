import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

import wandb

from data import load_dataset
from core import clip, loralib, losses, utils
from utils import *



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arch", type=str, default="CLIP")
    parser.add_argument("--dataset", type=str, default="waterbirds")
    parser.add_argument("--n_cls", type=int, default=1000)
    parser.add_argument("--prompt_id", type=int, default=0)
    
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--num_lora", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=1.)
    parser.add_argument("--lora_dropout", type=float, default=0.)
    parser.add_argument("--lora_modules", type=str, default="q,v")
    parser.add_argument("--lora_w_pretrain", action="store_true")
    parser.add_argument("--kl", action="store_true")
    parser.add_argument("--dot", action="store_true")
    parser.add_argument("--entropy", action="store_true")
    
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--optim", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=0.1)
    
    parser.add_argument("--lambda_cls", type=float, default=1.)
    parser.add_argument("--lambda_feat_ortho", type=float, default=0.)
    parser.add_argument("--lambda_param_ortho", type=float, default=0.)
    parser.add_argument("--l1", action="store_true")
    parser.add_argument("--only_wA", action="store_true")
    parser.add_argument("--compare_org", action="store_true")
    
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./experiments/models/CLIP@IN/")
    
    parser.add_argument("--resume_id", type=str, default="")
    args = parser.parse_args()
    
    ## Set ENV
    utils.set_seed(args.seed)
    
    save_dir = args.save_dir
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
    model = clip.CLIP_FT("ViT-B/16", "cuda", n_cls=args.n_cls)
    for name, param in model.named_parameters():
        param.requires_grad = True
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.wd)
    
    cls_loss_fn = nn.CrossEntropyLoss()
    
    train_dataset = load_dataset("../../hdd", "imagenet", "train", model.preprocess, args.prompt_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)
    
    ## Train
    wandb.define_metric("step/iter")
    wandb.define_metric("step/*", step_metric="step/iter")
    
    model = nn.DataParallel(model).cuda()
    model.train()
    
    iteration = 0
    for epoch in range(1, args.epochs+1):
        if os.path.exists(save_dir + f'epoch{epoch}.pt'):
            loralib.load_lora(model, save_dir + f'epoch{epoch}.pt')
            optimizer.load_state_dict(torch.load(save_dir + f'epoch{epoch}_op.pt'))
            iteration = epoch * len(train_loader)
            continue
        
        for data in tqdm(train_loader, desc=f'Epoch: {epoch:03d}', ncols=100):
            images = data['images'].to("cuda")
            labels = data['labels'].to("cuda")
            
            outputs, _, _ = model(images)
            cls_loss = cls_loss_fn(outputs, labels)
            
            optimizer.zero_grad()
            cls_loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct = preds.detach().cpu().numpy() == labels.cpu().numpy()
            train_acc = np.sum(correct) / labels.shape[0]
            
            iteration += 1
            wandb.log({
                "step/iter": iteration,
                "step/loss": cls_loss.item(),
                "step/train_acc": train_acc*100,
            })

        torch.save(model.state_dict(), save_dir + f'epoch{epoch}.pt')
        torch.save(optimizer.state_dict(), save_dir + f'epoch{epoch}_op.pt')
    
    wandb.finish()
    f.write('\n')
    f.close()