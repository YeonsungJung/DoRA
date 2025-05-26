import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import load_dataset, DATASET2CLSNUM
from core import clip, loralib, utils
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
        outputs = model(data['images'].to("cuda"))
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
    parser.add_argument("--save_dir", type=str, default="./experiments/models/CLIP@SepLoRA@r4")
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
        model = clip.CLIP_FT(args.arch, "cuda", n_cls=args.n_cls).eval()
    else:
        raise NotImplementedError(f'{args.model} is not implemented yet.')
    model = nn.DataParallel(model)
    
    infer, evaluate, test_datasets, test_loaders, projection_fns = load_test_configs(args, model.module.preprocess)
    
    ## Evaluation on original CLIP
    if args.eval_org:
        for test_dataset, test_loader, projection_fn in zip(test_datasets, test_loaders, projection_fns):
            worst_acc, avg_acc, accs_by_group = evaluate(*infer(model, test_loader, projection_fn, desc="Eval CLIP" + f" ({test_dataset})"), str=True)
            f.write(f"== CLIP ==  ({test_dataset})\n")
            f.write(f"Average accuracy: {avg_acc} | Worst Group accuracy: {worst_acc} | Acc by group: {accs_by_group}\n")
    
    ## Evaluation on fine-tuned model
    else:
        f.write(f"Evaluation on \"{args.save_dir}\"\n")
        
        lora_idxs = list(range(args.num_lora))
        lora_modules = [m for m in args.lora_modules.split(',') if m in ['q', 'k', 'v', 'out', 'mlp']]
        loralib.apply_lora(model.module, args.num_lora, args.r, args.lora_alpha, args.lora_dropout, lora_modules, gating=args.gating)
        
        if not args.eval_step:
            loralib.load_lora(model.module, args.save_dir + f'/epoch{args.epochs}.pt')
            loralib.set_used_lora(model.module, lora_idxs)
            model.eval()
            
            for test_dataset, test_loader, projection_fn in zip(test_datasets, test_loaders, projection_fns):
                worst_acc, avg_acc, accs_by_group = evaluate(*infer(model, test_loader, projection_fn, desc="Eval CLIP+LoRA" + f" ({test_dataset})"), str=True)
                f.write(f"== CLIP+LoRA (epoch {args.epochs}) ==  ({test_dataset})\n")
                f.write(f"Average accuracy: {avg_acc} | Worst Group accuracy: {worst_acc} | Acc by group: {accs_by_group}\n")
        
        else:
            train_epochs = [int(m) for m in args.epochs_per_step.split(',')]
            if len(train_epochs) != args.num_lora:
                raise NotImplementedError('Wrong number of training steps.')
            
            for i in range(args.num_lora):
                loralib.load_lora(model.module, args.save_dir + f'/step{i+1}_epoch{train_epochs[i]}.pt')
                loralib.set_used_lora(model.module, [i])
                model.eval()
                
                for test_dataset, test_loader, projection_fn in zip(test_datasets, test_loaders, projection_fns):
                    worst_acc, avg_acc, accs_by_group = evaluate(*infer(model, test_loader, projection_fn, desc=f"Eval CLIP+LoRA{i+1}" + f" ({test_dataset})"), str=True)
                    f.write(f"== Step{i+1}) CLIP+LoRA{i+1} (epoch {train_epochs[i]}) ==  ({test_dataset})\n")
                    f.write(f"Average accuracy: {avg_acc} | Worst Group accuracy: {worst_acc} | Acc by group: {accs_by_group}\n")
            
        f.write('\n')
        f.close()
