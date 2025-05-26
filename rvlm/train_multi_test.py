import os
import argparse
from tqdm import tqdm
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from itertools import combinations
from collections import defaultdict

import wandb

from data import load_dataset, DATASET2CLSNUM, imagenet_templates
from core import clip, loralib, losses, utils
from eval import load_test_configs, load_val_configs
from utils import *



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default="CLIP")
    parser.add_argument("--arch", type=str, default="ViT-L/14")
    parser.add_argument("--dataset", type=str, default="waterbirds")
    parser.add_argument("--prompt_id", type=int, default=8)
    
    # lora
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--num_lora", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=1.)
    parser.add_argument("--lora_dropout", type=float, default=0.)
    parser.add_argument("--lora_modules", type=str, default="q,k,v,out")
    parser.add_argument("--lora_w_pretrain", action="store_true")
    parser.add_argument("--train_visual_proj", action="store_true")
    
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
    parser.add_argument("--feat_kk", action="store_true")
    
    # 
    parser.add_argument("--text_cls", action="store_true")
    parser.add_argument("--desc_cls", action="store_true")
    

    # sparsity regularization
    parser.add_argument("--lambda_reg", type=float, default=0.1)
    parser.add_argument("--entropy", action="store_true")
    parser.add_argument("--l1", action="store_true")
    
    #gating
    parser.add_argument("--gating", action="store_true")

    # train
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--optim", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=5e-5)
    parser.add_argument("--lr_schedule", action="store_true")
    
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default=f"./experiments/models")
    
    parser.add_argument("--resume_id", type=str, default="")
    args = parser.parse_args()
    
    ## Set ENV
    utils.set_seed(args.seed)
    args.n_cls = DATASET2CLSNUM[args.dataset]
    
    lora_idxs = list(range(args.num_lora))
    lora_pairs = list(combinations(lora_idxs, 2))
    lora_modules = [m for m in args.lora_modules.split(',') if m in ['q', 'k', 'v', 'out', 'mlp']]
    
    save_dir = f"{args.save_dir}/{args.dataset}/CLIP@LoRA"

    if args.text_cls:
        save_dir += f"_textCls"
    if args.desc_cls:
        save_dir += f"_textClswithDesc"


    if args.lambda_desc_ortho > 0.:
        save_dir += f"_desc{args.lambda_desc_ortho}_prompt{args.prompt_id}"

    if args.lambda_feat_ortho > 0.:
        if args.kl:
            save_dir += f"_feat{args.lambda_feat_ortho}_kl"
        elif args.dot:
            save_dir += f"_feat{args.lambda_feat_ortho}_dot"
        elif args.feat_kk:
            save_dir += f"_feat{args.lambda_feat_ortho}_KK"
        else:
            save_dir += f"_feat{args.lambda_feat_ortho}_cosine"
    if args.lambda_param_ortho > 0.:
        save_dir += f"_param{args.lambda_param_ortho}"

    if args.train_visual_proj:
        save_dir += '_proj'
    if args.lora_w_pretrain: save_dir += "@wp"
    if args.gating: save_dir += "@gating"
    save_dir += f"@r{args.r}_num{args.num_lora}_{''.join(lora_modules)}/"
    os.makedirs(save_dir, exist_ok=True)
    write_json(f"{save_dir}config.json", vars(args))
    f = open(f"{save_dir}log.txt", 'a')
    
    if args.resume_id:
        wandb.init(project="rvlm", id=args.resume_id, resume=True)
    else:
        wandb.init(project="rvlm")
    wandb.config.update(args)
    wandb.run.name = save_dir.split('/')[-2]
    
    latest_model_path = os.path.join(save_dir, "latest_model.pt")
    latest_optimizer_path = os.path.join(save_dir, "latest_optimizer.pt")


    ## Load model
    if args.model == "CLIP":
        model = clip.CLIP_FT(args.arch, "cuda", n_cls=args.n_cls, text_cls=args.text_cls or args.desc_cls)
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
    

    if args.lambda_desc_ortho > 0.: 
        all_descs = [[template.format(desc) for template in imagenet_templates] for desc in train_dataset.all_descs]
        all_descs = [model.model.encode_text(clip.tokenize(desc).to("cuda")) for desc in all_descs]
        all_descs = [desc / desc.norm(dim=-1, keepdim=True) for desc in all_descs]
        all_descs = [desc.mean(dim=0) for desc in all_descs]
        all_descs = [desc / desc.norm() for desc in all_descs]
        desc_feats = torch.stack(all_descs, dim=0).detach()
        
        target_layers = [i for i in range(model.n_layers-args.last_num, model.n_layers)]
        model.set_desc_loss_fn(target_layers, desc_feats, ortho_feat_loss_fn)
    
    class_embs = None
    if args.text_cls or args.desc_cls:
        # class_descs = train_dataset.class_descs
        # class_embs, desc_embs = [], []\


        descs_embs = []

        class_embs = []
        for k in train_dataset.classnames:
            

            print(k)
            print(train_dataset.class_descs[k])

            class_emb = model.model.encode_text(clip.tokenize(k).to("cuda"))
            desc_embs = [model.model.encode_text(clip.tokenize(desc).to("cuda")) for desc in train_dataset.class_descs[k]]
            
            class_emb = class_emb / class_emb.norm(dim=-1, keepdim=True)
            desc_emb = [desc / desc.norm(dim=-1, keepdim=True) for desc in desc_embs]

            class_embs.append(class_emb)
            descs_embs.append(desc_emb)



        from torch.nn import functional as F
        for cls_emb in class_embs:
            for desc_emb in descs_embs:
                scores = [F.cosine_similarity(cls_emb, desc) for desc in desc_emb]
                print(scores)

            

        #     if args.desc_cls:
        #         attr_sentence = ", ".join(train_dataset.class_descs[k][:-1]) + ", and " + train_dataset.class_descs[k][-1]
        #         classname_attr_sentence = "{} with {}.".format(k, attr_sentence)
        #         class_emb = [template.format(classname_attr_sentence) for template in imagenet_templates]
        #     else:
        #         class_emb = [template.format(k) for template in imagenet_templates]
        #     class_emb = model.model.encode_text(clip.tokenize(class_emb).to("cuda"))
        #     class_emb = class_emb / class_emb.norm(dim=-1, keepdim=True)
        #     class_emb = class_emb.mean(dim=0)
        #     class_emb = class_emb / class_emb.norm()
        #     class_embs.append(class_emb)     
        # class_embs = torch.stack(class_embs, dim=0)


    model.model = model.model.visual

    if args.lambda_feat_ortho > 0.:
        loralib.apply_lora(model, args.num_lora, args.r, args.lora_alpha, args.lora_dropout, lora_modules, ortho_feat_loss_fn, args.lora_w_pretrain, gating=args.gating)
    else:
        loralib.apply_lora(model, args.num_lora, args.r, args.lora_alpha, args.lora_dropout, lora_modules, gating=args.gating)
    print('{} w/  LoRA: {:.1f}M'.format(args.model, sum(param.numel() for param in model.parameters())/1000000.0))
    
    ## Train
    wandb.define_metric("step/iter")
    wandb.define_metric("step/*", step_metric="step/iter")
    loralib.set_used_lora(model, lora_idxs)
    _, trainable_params = loralib.get_lora_params(model, fc=True, idxs=lora_idxs, train_visual_proj=args.train_visual_proj, gating=args.gating)
    model = nn.DataParallel(model)
    
    if args.optim=="adamw": optimizer = optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.wd)
    elif args.optim=="sgd": optimizer = optim.SGD(trainable_params, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    if args.lr_schedule: scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs*len(train_loader))
    
    # Check for existing checkpoint
    if os.path.exists(latest_model_path) and os.path.exists(latest_optimizer_path):
        print(f"Resuming training from {latest_model_path}")
        loralib.load_lora(model.module, latest_model_path)
        checkpoint = torch.load(latest_optimizer_path)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        print("Starting training from scratch.")
        start_epoch = 1

    iteration = (start_epoch-1) * len(train_loader)
    best_val_worst_acc = -1  
    for epoch in range(start_epoch, args.epochs+1):
        model.train()
        for data in tqdm(train_loader, desc=f'Epoch: {epoch:03d}', ncols=100):
            if args.dataset == 'waterbirds':
                images, attrs, _ = data
                labels = attrs[:,0]
                spurious = attrs[:,1]
            elif args.dataset == 'imagenet':
                images, labels = data['images'], data['labels']
                spurious = labels
            
            outputs, desc_loss, feat_loss = model(images.to("cuda"))


            

            for label in range(args.n_cls):
                cls_emb = class_embs[label]
                output = outputs[labels==label]
                output = output/output.norm(dim=-1, keepdim=True)

                cls_similarities = F.cosine_similarity(output.unsqueeze(1), torch.cat(class_embs, dim=0).unsqueeze(0), dim=-1)
                print(cls_similarities)
                for desc_emb in descs_embs:
                    desc_emb = torch.cat(desc_emb, dim=0)                   
                    desc_similarities = F.cosine_similarity(output.unsqueeze(1), desc_emb.unsqueeze(0), dim=-1).mean(dim=1)
                    print(desc_similarities)
            quit()







            if args.text_cls:
                cls_loss = cls_loss_fn((outputs/outputs.norm(dim=-1, keepdim=True)) @ class_embs.T, labels.to("cuda"))
            else:
                cls_loss = cls_loss_fn(outputs, labels.to("cuda"))
            ortho_loss = torch.tensor([0.]).cuda()
            if args.lambda_desc_ortho > 0.: 
                desc_loss = desc_loss.mean()
                ortho_loss += args.lambda_desc_ortho * desc_loss
            if args.lambda_feat_ortho > 0.: 
                feat_loss = feat_loss.mean()
                ortho_loss += args.lambda_feat_ortho * feat_loss
            if args.lambda_param_ortho > 0.:
                tmp_params = defaultdict(list)
                org_params = []
                for name, param in model.module.model.transformer.resblocks.named_parameters():
                    if "lora0_A" in name:
                        idx = name.split('.')[0]
                        end_name = '.'.join(name.split('.')[1:-1])
                        name_for_eval = f"model.module.model.transformer.resblocks[{idx}].{end_name}"
                        
                        for i in lora_idxs:
                            wA = eval(name_for_eval.replace('lora0_A', f'lora{i}_A')).weight
                            wB = eval(name_for_eval.replace('lora0_A', f'lora{i}_B')).weight
                            if args.only_wA: tmp_params[i].append(wA)
                            else: tmp_params[i].append(torch.mm(wB, wA))
                        org_w = eval(name_for_eval.replace('lora0_A', 'org_linear')).weight
                        org_params.append(org_w)
                ortho_loss += args.lambda_param_ortho * ortho_param_loss_fn(tmp_params, org_params)
            
            loss = args.lambda_cls * cls_loss + ortho_loss
            
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
                "step/loss_cls": cls_loss.item(),
                "step/loss_ortho": ortho_loss.item(),
                "step/train_worst_acc": train_worst_acc,
                "step/train_avg_acc": train_avg_acc,
                "step/lr" : optimizer.param_groups[0]['lr'],

            })

        loralib.save_lora(model.module, latest_model_path, idxs=lora_idxs, train_visual_proj=args.train_visual_proj)
        torch.save({"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()}, latest_optimizer_path)

        # Evaluation on test sets
        model.eval()
        with torch.no_grad():
            worst_acc, avg_acc, accs_by_group = evaluate(*infer(model, valid_loader, None, desc=f"Eval Validation", class_embs=class_embs), str=False)
        print(f"Epoch {epoch}) Validation Set - Average accuracy: {avg_acc:.2f} | Worst Group accuracy: {worst_acc:.2f} | Acc by group: {[round(acc, 2) for acc in accs_by_group]}\n")

        # 최고 validation worst_acc를 기록하고 해당 epoch의 test 결과도 저장
        if worst_acc >= best_val_worst_acc:
            best_val_worst_acc = worst_acc
            f.write(f"### Epoch {epoch}\n")
            f.write(f"Validation Set - Average accuracy: {avg_acc:.2f} | Worst Group accuracy: {worst_acc:.2f} | Acc by group: {[round(acc, 2) for acc in accs_by_group]}\n")

            # Best validation을 기록한 epoch의 test 결과 저장
            for test_dataset, test_loader, projection_fn in zip(test_datasets, test_loaders, projection_fns):
                with torch.no_grad():
                    test_worst_acc, test_avg_acc, test_accs_by_group = evaluate(*infer(model, test_loader, projection_fn, desc=f"Eval Test ({test_dataset})", class_embs=class_embs), str=False)
                    print(f"Epoch {epoch}) Test Set ({test_dataset}) - Average accuracy: {test_avg_acc:.2f} | Worst Group accuracy: {test_worst_acc:.2f} | Acc by group: {[round(acc, 2) for acc in test_accs_by_group]}\n")
                f.write(f"Test Set ({test_dataset}) - Average accuracy: {test_avg_acc:.2f} | Worst Group accuracy: {test_worst_acc:.2f} | Acc by group: {[round(acc, 2) for acc in test_accs_by_group]}\n")
    
    f.write('\n')
    f.close()
    wandb.finish()