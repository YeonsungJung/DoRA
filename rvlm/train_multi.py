import os
import argparse
from tqdm import tqdm
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from itertools import combinations
from collections import defaultdict

import wandb

from data import load_dataset, DATASET2CLSNUM, imagenet_templates
from core import clip, loralib, losses, utils
from eval import load_test_configs, load_val_configs
from utils import *
from scheduler import cosine_lr


def collate_fn_padded_dict(batch):
    batch_size = args.batch_size

    images = [b["images"] for b in batch]
    labels = [b["labels"] for b in batch]
    texts = [b["texts"] for b in batch]

    max_size = images[0].shape
    current_batch_size = len(images)

    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    if current_batch_size < batch_size:
        pad_size = batch_size - current_batch_size

        pad_images = torch.zeros((pad_size, *max_size), dtype=images.dtype)
        images = torch.cat([images, pad_images], dim=0)

        pad_labels = torch.full((pad_size,), -1, dtype=labels.dtype)
        labels = torch.cat([labels, pad_labels], dim=0)

        texts.extend(texts[:pad_size])

    return {"images": images, "labels": labels, "texts":texts}


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default="CLIP")
    parser.add_argument("--arch", type=str, default="ViT-B/16")
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--prompt_id", type=int, default=10)
    
    # lora
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--num_lora", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=16.)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_modules", type=str, default="q,k,v,out,mlp")
    parser.add_argument("--lora_w_pretrain", action="store_true")
    parser.add_argument("--train_visual_proj", action="store_false")
    parser.add_argument("--only_qk", action="store_true")
    parser.add_argument("--only_qkv", action="store_true")
    
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
    parser.add_argument("--ortho_pretrained", action="store_true")
    parser.add_argument("--loss_gram", action="store_true")

    parser.add_argument("--lora_intermediate", action="store_true")

    # multi-modal
    parser.add_argument("--cl", action="store_true")
    parser.add_argument("--freeze_text", action="store_true")
    
    #
    parser.add_argument("--text_cls", action="store_true")
    parser.add_argument("--desc_cls", action="store_true")
    parser.add_argument("--desc_lambda", type=float, default=0.)
    
    # pca
    parser.add_argument("--n_pca", type=int, default=100)
    parser.add_argument("--pcaOrtho_lambda", type=float, default=0.)
    parser.add_argument("--pca_perCls", action="store_true")

    
    
    
    # sparsity regularization
    parser.add_argument("--lambda_reg", type=float, default=0.1)
    parser.add_argument("--entropy", action="store_true")
    parser.add_argument("--l1", action="store_true")
    
    #gating
    parser.add_argument("--gating", action="store_true")

    # train
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--optim", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--lr_schedule", action="store_true")
    parser.add_argument("--warmup", type=int, default=500)
    
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default=f"./experiments/models")
    
    parser.add_argument("--resume_id", type=str, default="")
    args = parser.parse_args()
    
    ## Set ENV
    utils.set_seed(args.seed)
    args.n_cls = DATASET2CLSNUM[args.dataset]
    args.lr_schedule = True

    # if args.cl:
    #     args.train_visual_proj = True
    
    if args.ortho_pretrained:
        lora_idxs = list(range(args.num_lora+1))
    else:
        lora_idxs = list(range(args.num_lora))
    
    lora_pairs = list(combinations(lora_idxs, 2))
    lora_modules = [m for m in args.lora_modules.split(',') if m in ['q', 'k', 'v', 'out', 'mlp']]
    
    save_dir = f"{args.save_dir}/{args.dataset}/CLIP@LoRA_{args.arch.replace('/', '')}"
    # save_dir = f"{args.save_dir}/{args.dataset}/test"

    if args.pcaOrtho_lambda > 0.:
        # save_dir += f"_CL_PCA{args.n_pca}_lambda{args.pcaOrtho_lambda}"
        # save_dir += f"_CLwTextLoRA_PCA{args.n_pca}_lambda{args.pcaOrtho_lambda}"
        # save_dir += f"_CLwTextLoRA_freezeproj_PCA{args.n_pca}_lambda{args.pcaOrtho_lambda}"
        save_dir += f"_CLwTextLoRA_freezeproj_PCA{args.n_pca}_lambda{args.pcaOrtho_lambda}_after5"
    else:
        save_dir += "_CLwTextLoRA_freezeproj"

    if args.freeze_text:
        save_dir += f"_freezeText"

    if args.text_cls:
        save_dir += f"_textCls"
    if args.desc_cls:
        save_dir += f"_textClswithDesc"
        if args.desc_lambda != 0:
            save_dir += f"_{args.desc_lambda}"


    if args.lambda_desc_ortho > 0.:
        save_dir += f"_desc{args.lambda_desc_ortho}_prompt{args.prompt_id}"

    if args.lambda_feat_ortho > 0.:
        if args.loss_gram:
            save_dir += f"_feat{args.lambda_feat_ortho}_gram"
        elif args.kl:
            save_dir += f"_feat{args.lambda_feat_ortho}_kl"
        elif args.dot:
            save_dir += f"_feat{args.lambda_feat_ortho}_dot"
        elif args.feat_kk:
            save_dir += f"_feat{args.lambda_feat_ortho}_KK"
        else:
            save_dir += f"_feat{args.lambda_feat_ortho}_cosine"
        if args.ortho_pretrained:
            save_dir += "_preW"
        if args.lora_intermediate:
            save_dir += "_interFeat"


    if args.lambda_param_ortho > 0.:
        save_dir += f"_param{args.lambda_param_ortho}"

    save_dir += f"_lr{args.lr}_wd{args.wd}"
    if args.lr_schedule:
        save_dir += f"lrschedule"


    if args.train_visual_proj:
        save_dir += '_proj'

    save_dir += f'_batch{args.batch_size}'

    if args.lora_w_pretrain: save_dir += "@wp"
    if args.gating: save_dir += "@gating"

    save_dir += f"@r{args.r}_alpha{args.lora_alpha}_num{args.num_lora}"

    if args.lora_dropout > 0:
        save_dir += f"_dropout{args.lora_dropout}"

    if args.only_qk:
        save_dir += "_qk/"
    elif args.only_qkv:
        save_dir +='_qkv/'
    else:
        save_dir += f"_{''.join(lora_modules)}/"

    
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
        train_dataset = load_dataset(args.data_dir, args.dataset, "train", None, args.prompt_id)
        model = clip.CLIP_FT_desc(args, "cuda", train_dataset.class_descs, freeze_encoder=True).to("cuda")
    else:
        raise NotImplementedError(f'{args.model} is not implemented yet.')
    print('{} w/o LoRA: {:.1f}M'.format(args.model, sum(param.numel() for param in model.parameters())/1000000.0))
    
    # PCA ortho
    ortho_feat_loss_fn = losses.OrthoFeatLoss(args.ortho_pretrained)
    model.set_pcaOrtho(ortho_feat_loss_fn)

    ortho_param_loss_fn = losses.OrthoParamLoss(lora_pairs, args.compare_org)
    if args.only_wA and args.compare_org:
        raise NotImplementedError('We cannot compare wA with the original weight.')
    
    ## Load data
    train_dataset = load_dataset(args.data_dir, args.dataset, "train", model.preprocess, args.prompt_id, args.cl)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)
    num_batches = len(train_loader)

    valid_loader = load_val_configs(args, model.val_preprocess)
    infer, evaluate, test_datasets, test_loaders, projection_fns = load_test_configs(args, model.val_preprocess)

    all_class_texts = [template.format(k) for k in train_dataset.classnames for template in imagenet_templates]
    num_classes = len(train_dataset.classnames)
    num_templates = len(imagenet_templates)

    if args.only_qk:
        num_lora = {"q": args.num_lora, "k": args.num_lora, "v": 1, "out": 1}
    elif args.only_qkv:
        num_lora = {"q": args.num_lora, "k": args.num_lora, "v": args.num_lora, "out": 1}
    else:
        num_lora = {key: args.num_lora for key in lora_modules}

    loralib.apply_multilinear(
        model,
        num_loras=args.num_lora,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        visual_only=False,
        proj=args.train_visual_proj,
    )
    names, trainable_params = loralib.get_multilinear_params(model)
    print('{} w/  DoRA: {:.1f}M'.format(args.model, sum(param.numel() for param in model.parameters())/1000000.0))
    
    ## Train
    wandb.define_metric("step/iter")
    wandb.define_metric("step/*", step_metric="step/iter")
    
    # obtain parameters from multilinear layers
    model = nn.DataParallel(model)
    
    if args.optim=="adamw": optimizer = optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.wd)
    elif args.optim=="sgd": optimizer = optim.SGD(trainable_params, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    if args.lr_schedule: scheduler = cosine_lr(optimizer, args.lr, args.warmup, args.epochs*num_batches)
    # if args.lr_schedule: scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs*len(train_loader))
    # fp16_scaler = torch.cuda.amp.GradScaler()
    
    # Check for existing checkpoint
    if os.path.exists(latest_model_path) and os.path.exists(latest_optimizer_path):
        print(f"Resuming training from {latest_model_path}")
        loralib.load_multilinear(model.module, latest_model_path)
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
        for data_i, data in enumerate(tqdm(train_loader, desc=f'Epoch: {epoch:03d}', ncols=100)):
            if args.dataset == 'waterbirds':
                images, attrs, _ = data
                labels = attrs[:,0]
                spurious = attrs[:,1]
            elif args.dataset == 'imagenet':
                images, labels = data['images'], data['labels']
                spurious = labels
                images, labels = images.to("cuda"), labels.to("cuda")

            # if args.lr_schedule: scheduler.step()
            if args.lr_schedule: scheduler(data_i + epoch * num_batches)
            optimizer.zero_grad()
            

            class_prompts = data['texts']
            # image_features, text_features, logit_scale, feat_loss, pca_ortho_loss = model(images, clip.tokenize(class_prompts).to("cuda"))
            image_features, text_features, logit_scale, feat_loss, pca_ortho_loss = model(images, clip.tokenize(class_prompts).to("cuda"), False)
            # pca_ortho_loss = pca_ortho_loss.mean()
            # image_features, text_features, logit_scale, feat_loss, pca_ortho_loss = model(images, clip.tokenize(class_prompts).to("cuda"), return_dict=False)

            ## CL loss
            logit_scale = logit_scale if len(list(range(torch.cuda.device_count()))) == 1 else logit_scale[0]

            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            labels = torch.arange(logits_per_image.shape[0], dtype=torch.long).to("cuda")
            cl_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2

            # loss = cl_loss + args.pcaOrtho_lambda * pca_ortho_loss
            loss = cl_loss
            # if epoch > 5:
            #     loss = cl_loss + args.pcaOrtho_lambda * pca_ortho_loss
            # else:
            #     loss = cl_loss
            
            ortho_loss = torch.tensor(0.).to("cuda")
            if args.lambda_feat_ortho > 0.: 
                feat_loss = feat_loss.mean()
                ortho_loss += args.lambda_feat_ortho * feat_loss
            if args.lambda_param_ortho > 0.:
                tmp_params = defaultdict(list)
                org_params = []
                for name, param in model.module.model.visual.transformer.resblocks.named_parameters():
                    if "lora0_A" in name:
                        idx = name.split('.')[0]
                        end_name = '.'.join(name.split('.')[1:-1])
                        name_for_eval = f"model.module.model.visual.transformer.resblocks[{idx}].{end_name}"
                        
                        for i in lora_idxs:
                            wA = eval(name_for_eval.replace('lora0_A', f'lora{i}_A')).weight
                            wB = eval(name_for_eval.replace('lora0_A', f'lora{i}_B')).weight
                            if args.only_wA: tmp_params[i].append(wA)
                            else: tmp_params[i].append(torch.mm(wB, wA))
                        org_w = eval(name_for_eval.replace('lora0_A', 'org_linear')).weight
                        org_params.append(org_w)
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

            loss += ortho_loss
            
            loss.backward()
            optimizer.step()

            iteration += 1
            wandb.log({
                "step/iter": iteration,
                "step/loss": loss.item(),
                "step/loss_cl": cl_loss.item(),
                "step/loss_ortho": ortho_loss.item(),
                # "step/loss_ortho": pca_ortho_loss.item(),
                "step/lr" : optimizer.param_groups[0]['lr'],
            })
            
            del cl_loss
            del pca_ortho_loss
            del loss
            del ortho_loss


        loralib.save_multilinear(model.module, latest_model_path)
        torch.save({"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()}, latest_optimizer_path)

        # Evaluation on test sets
        model.eval()


        with torch.no_grad():
            class_embs = []
    
            for i in range(0, len(all_class_texts), args.batch_size):
                batch_texts = all_class_texts[i : i + args.batch_size]
                batch_embs = model(None, clip.tokenize(batch_texts).to("cuda")) 
                class_embs.append(batch_embs)  
            all_class_embs = torch.cat(class_embs, dim=0)
            class_embs = all_class_embs.view(num_classes, num_templates, -1).mean(dim=1)
            class_embs = class_embs / class_embs.norm(dim=-1, keepdim=True)

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
