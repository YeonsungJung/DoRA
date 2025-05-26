import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import combinations

from data import load_dataset, DATASET2CLSNUM, imagenet_templates
from core import clip, loralib, losses, utils
from utils import *


# https://github.com/chingyaoc/debias_vl.git
def infer_wg(model, data_loader, projection_fn=None, desc='', class_embs=None):
    all_preds = []
    all_labels = []
    all_spurious = []
    for data in tqdm(data_loader, desc=desc):
        images, attrs, _ = data
        
        outputs, _, _ = model(images.to("cuda"))
        if class_embs is not None:
            outputs = (outputs/outputs.norm(dim=-1, keepdim=True)) @ class_embs.T

        outputs = outputs.detach().cpu()
        if projection_fn is not None: 
            outputs = projection_fn(outputs, "cpu")
        _, preds = torch.max(outputs, 1)
        
        all_preds.append(preds)
        all_labels.append(attrs[:,0])
        all_spurious.append(attrs[:,1])
        
    all_preds, all_labels, all_spurious = torch.concat(all_preds, dim=0).numpy(), torch.concat(all_labels, dim=0).numpy(), torch.concat(all_spurious, dim=0).numpy()
    return all_preds, all_labels, all_spurious

def evaluate_wg(all_preds, all_labels, all_spurious, str=False):
    correct_by_group = [[0, 0], [0, 0]]
    total_by_group   = [[0, 0], [0, 0]]
    accs_by_group    = [[0, 0], [0, 0]]
    correct = all_preds == all_labels
    
    for t in [0, 1]:
        for s in [0 ,1]:
            ix = np.where(np.logical_and(all_labels == t, all_spurious == s))[0]
            correct_by_group[t][s] += np.sum(correct[ix])
            total_by_group[t][s] += len(ix)
            accs_by_group[t][s] = np.sum(correct[ix]) / len(ix)
        
    # Average accuracy
    avg_acc = (
        correct_by_group[0][0] +
        correct_by_group[0][1] +
        correct_by_group[1][0] +
        correct_by_group[1][1]
    )
    avg_acc = avg_acc * 100 / np.sum(np.array(total_by_group))
    
    accs_by_group = np.array(accs_by_group).flatten() * 100
    accs_by_group = accs_by_group.tolist()
    worst_acc = np.min(accs_by_group)
    
    del all_preds
    if str: return f"{worst_acc:2f}", f"{avg_acc:2f}", f"[{accs_by_group[0]:.2f}, {accs_by_group[1]:.2f}, {accs_by_group[2]:.2f}, {accs_by_group[3]:.2f}]"
    else: return worst_acc, avg_acc, accs_by_group

def infer_cls(model, data_loader, projection_fn=None, desc='', class_embs=None):
    all_preds = []
    all_labels = []
    for data in tqdm(data_loader, desc=desc):
        outputs, _, _ = model(data['images'].to("cuda"))
        if class_embs is not None:
            outputs = outputs @ class_embs.T
    
        outputs = outputs.detach().cpu()
        if projection_fn is not None: 
            outputs = projection_fn(outputs, "cpu")
        _, preds = torch.max(outputs, 1)
        
        all_preds.append(preds)
        all_labels.append(data['labels'])
        
    all_preds, all_labels = torch.concat(all_preds, dim=0).numpy(), torch.concat(all_labels, dim=0).numpy()
    return all_preds, all_labels

def infer_cifar(model, data_loader, projection_fn=None, desc=''):
    all_preds = []
    all_labels = []
    for data in tqdm(data_loader, desc=desc):
        images, labels = data
        outputs, _, _ = model(images.to("cuda"))
        outputs = outputs.detach().cpu()
        if projection_fn is not None: 
            outputs = projection_fn(outputs, "cpu")
        _, preds = torch.max(outputs, 1)
        
        all_preds.append(preds)
        all_labels.append(labels)
        
    all_preds, all_labels = torch.concat(all_preds, dim=0).numpy(), torch.concat(all_labels, dim=0).numpy()
    return all_preds, all_labels

def evaluate_cls(all_preds, all_labels, all_spurious=None, str=False):
    correct = np.sum(all_preds == all_labels)
    total = len(all_labels)
    avg_acc = (correct / total) * 100 
    
    del all_preds, all_labels
    if str: return "None", f"{avg_acc:2f}", "None"
    else: return 0, avg_acc, [0]

def load_test_configs(args, preprocess):
    if args.dataset=="imagenet":
        infer = infer_cls
        evaluate = evaluate_cls
        test_datasets = ["imagenet-v2", "imagenet-r", "imagenet-a", "imagenet-sketch"]
        #test_datasets = ["imagenet-r", "imagenet-a"]
    else:
        infer = infer_wg
        evaluate = evaluate_wg
        test_datasets = [args.dataset]
        
    test_loaders = []
    projection_fns = []
    for dataset in test_datasets:
        test_dataset = load_dataset(args.data_dir, dataset, "test", preprocess, args.prompt_id)
        projection_fns.append(test_dataset.project_logits)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
        test_loaders.append(test_loader)    
    
    return infer, evaluate, test_datasets, test_loaders, projection_fns

def load_val_configs(args, preprocess):
    valid_dataset = load_dataset(args.data_dir, args.dataset, "valid", preprocess, args.prompt_id)
    valid_loader = DataLoader(valid_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers) 
    return valid_loader



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default="CLIP")
    parser.add_argument("--arch", type=str, default="ViT-L/14")
    parser.add_argument("--dataset", type=str, default="waterbirds")
    parser.add_argument("--prompt_id", type=int, default=6)
    
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--epochs_per_step", type=str, default="4,4")
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=16)
    
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./experiments/models/imagenet/CLIP@LoRA_ViT-B16_desc_CLwTextLoRA_freezeproj_param1.0_lr0.0002_wd0.01lrschedule_proj_batch256@r256_alpha256.0_num1_dropout0.1_qkvoutmlp")
    parser.add_argument("--log_path", type=str, default="./eval_log.txt")
    
    parser.add_argument("--eval_org", action="store_true")
    parser.add_argument("--eval_step", action="store_true")
    args = parser.parse_args()
    
    try:
        train_config = read_json(args.save_dir + '/config.json')
        for k, v in train_config.items(): 
            if k not in ["data_dir", "save_dir", "test_batch_size", "epochs", "epochs_per_step"]: setattr(args, k, v)
        args.gating=False
    except:
        print("Train config is not provided.")
    
    utils.set_seed(args.seed)
    f = open(args.log_path, 'a')
    
    ## Load data and model
    if args.model == "CLIP":
        train_dataset = load_dataset(args.data_dir, args.dataset, "train", None, args.prompt_id)
        model = clip.CLIP_FT_desc(args, "cuda", train_dataset.class_descs, freeze_encoder=True).to("cuda")
    else:
        raise NotImplementedError(f'{args.model} is not implemented yet.')
    
    infer, evaluate, test_datasets, test_loaders, projection_fns = load_test_configs(args, model.preprocess)
    
    
    latest_model_path = os.path.join(args.save_dir, "latest_model.pt")
    latest_optimizer_path = os.path.join(args.save_dir, "latest_optimizer.pt")
    
    if args.ortho_pretrained:
        lora_idxs = list(range(args.num_lora+1))
    else:
        lora_idxs = list(range(args.num_lora))
    
    lora_pairs = list(combinations(lora_idxs, 2))
    lora_modules = [m for m in args.lora_modules.split(',') if m in ['q', 'k', 'v', 'out', 'mlp']]
    
    if args.only_qk:
        num_lora = {"q": args.num_lora, "k": args.num_lora, "v": 1, "out": 1}
    elif args.only_qkv:
        num_lora = {"q": args.num_lora, "k": args.num_lora, "v": args.num_lora, "out": 1}
    else:
        num_lora = {key: args.num_lora for key in lora_modules}
    
    # args.lora_alpha = 128
    if args.lambda_feat_ortho > 0. :
        ortho_feat_loss_fn = losses.OrthoFeatLoss(args.ortho_pretrained)
        # loralib.apply_lora(model, num_lora, args.r, args.lora_alpha, args.lora_dropout, lora_modules, ortho_feat_loss_fn, gating=args.gating, lora_intermediate=args.lora_intermediate)
        loralib.apply_lora(model, num_lora, args.r, args.lora_alpha, args.lora_dropout, lora_modules, ortho_feat_loss_fn, gating=args.gating, lora_intermediate=args.lora_intermediate, visual_only=False)
    else:
        # loralib.apply_lora(model, num_lora, args.r, args.lora_alpha, args.lora_dropout, lora_modules, gating=args.gating, lora_intermediate=args.lora_intermediate)
        loralib.apply_lora(model, num_lora, args.r, args.lora_alpha, args.lora_dropout, lora_modules, gating=args.gating, lora_intermediate=args.lora_intermediate, visual_only=False)
    # loralib.set_used_lora(model, lora_idxs)
    loralib.set_used_lora(model, lora_idxs, visual_only=False)
    model = nn.DataParallel(model)
    
    print(f"Loading model from {latest_model_path}")
    loralib.load_lora(model.module, latest_model_path)
    model.eval()
    
    with torch.no_grad():
        all_class_texts = [template.format(k) for k in train_dataset.classnames for template in imagenet_templates]
        num_classes = len(train_dataset.classnames)
        num_templates = len(imagenet_templates)
        class_embs = []

        for i in range(0, len(all_class_texts), args.batch_size):
            batch_texts = all_class_texts[i : i + args.batch_size]
            batch_embs = model(None, clip.tokenize(batch_texts).to("cuda")) 
            class_embs.append(batch_embs)  
        all_class_embs = torch.cat(class_embs, dim=0)
        class_embs = all_class_embs.view(num_classes, num_templates, -1).mean(dim=1)
        class_embs = class_embs / class_embs.norm(dim=-1, keepdim=True)

    for test_dataset, test_loader, projection_fn in zip(test_datasets, test_loaders, projection_fns):
        with torch.no_grad():
            test_worst_acc, test_avg_acc, test_accs_by_group = evaluate(*infer(model, test_loader, projection_fn, desc=f"Eval Test ({test_dataset})", class_embs=class_embs), str=False)
            print(f"Test Set ({test_dataset}) - Average accuracy: {test_avg_acc:.2f} | Worst Group accuracy: {test_worst_acc:.2f} | Acc by group: {[round(acc, 2) for acc in test_accs_by_group]}\n")
        f.write(f"Test Set ({test_dataset}) - Average accuracy: {test_avg_acc:.2f} | Worst Group accuracy: {test_worst_acc:.2f} | Acc by group: {[round(acc, 2) for acc in test_accs_by_group]}\n")
    