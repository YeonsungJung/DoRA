#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn

from typing import Dict

from .layers import LoRALayer, LoRAInjectedLinear, LoRAInjectedMultiheadAttention


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


# soohyun
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py
def find_modules(model, ancestor_class=["ResidualAttentionBlock"], search_class=[], exclude_children_of=[LoRAInjectedLinear, LoRAInjectedMultiheadAttention, nn.MultiheadAttention]):
    if ancestor_class is not None:
        ancestors = (
            module
            for module in model.modules()
            if module.__class__.__name__ in ancestor_class
        )
    else:
        # this, incase you want to naively iterate over all modules.
        ancestors = [module for module in model.modules()]

    # For each target find every linear_class module that isn't a child of a LoraInjectedLinear
    for ancestor in ancestors:
        for fullname, module in ancestor.named_modules():
            if any([isinstance(module, _class) for _class in search_class]):
                # Find the direct parent if this is a descendant, not a child, of target
                *path, name = fullname.split(".")
                parent = ancestor
                while path:
                    parent = parent.get_submodule(path.pop(0))
                # Skip this linear if it's a child of a LoraInjectedLinear
                if exclude_children_of and any(
                    [isinstance(parent, _class) for _class in exclude_children_of]
                ):
                    continue
                # Otherwise, yield it
                yield parent, name, module


def apply_lora(model, num_lora=1, r=4, lora_alpha=1, lora_dropout=0., lora_modules=[], feat_loss_fn=None, visual_only=True, proj=False, gating=False, lora_intermediate=False, lambda_scale=1.0):
    target_blocks = [model.model.visual.transformer.resblocks] if visual_only else [model.model.visual.transformer.resblocks, model.model.transformer.resblocks]
    search_classes = []
    if "mlp" in lora_modules: 
        search_classes.append(nn.Linear)
        lora_modules.remove("mlp")
    if len(lora_modules)>0: search_classes.append(nn.MultiheadAttention)
    device, dtype = model.device, model.model.dtype
    
    for target_block in target_blocks:
        for _module, name, _child_module in find_modules(target_block, ["ResidualAttentionBlock"], search_classes):
            if _child_module.__class__ == nn.Linear:
                _tmp = LoRAInjectedLinear(_child_module, num_lora["mlp"], r, lora_alpha, lora_dropout, feat_loss_fn, lora_intermediate=lora_intermediate, lambda_scale=lambda_scale).to(device).to(dtype)
                _module._modules[name] = _tmp
            if _child_module.__class__ == nn.MultiheadAttention:
                _tmp = LoRAInjectedMultiheadAttention(_child_module, lora_modules, num_lora, r, lora_alpha, lora_dropout, feat_loss_fn, gating=gating, lora_intermediate=lora_intermediate, lambda_scale=lambda_scale).to(device).to(dtype)
                _module._modules[name] = _tmp

    if proj: 
        proj_params = model.model.visual.proj
        to_layer = nn.Linear(proj_params.shape[0], proj_params.shape[1], bias=False)
        to_layer.weight.data.copy_(proj_params.t())
        del model.model.visual.proj
        model.model.visual.proj = LoRAInjectedLinear(to_layer, num_lora["mlp"], r, lora_alpha, lora_dropout, feat_loss_fn, lora_intermediate=lora_intermediate, lambda_scale=lambda_scale).to(device).to(dtype)
        if not visual_only:
            proj_params = model.model.text_projection
            to_layer = nn.Linear(proj_params.shape[0], proj_params.shape[1], bias=False)
            to_layer.weight.data.copy_(proj_params.t())
            del model.model.text_projection
            model.model.text_projection = LoRAInjectedLinear(to_layer, num_lora["mlp"], r, lora_alpha, lora_dropout, feat_loss_fn, lora_intermediate=lora_intermediate, lambda_scale=lambda_scale).to(device).to(dtype)


def get_lora_params(model, fc=True, idxs=[], train_visual_proj=False, train_text_proj=False, train_text_encoder=False, gating=False):
    names, params = [], []
    for name, param in model.named_parameters():
        requires_grad = False        
        if train_visual_proj and "model.visual.proj" in name: requires_grad = True
        elif train_text_proj and "model.text_projection" in name: requires_grad = True
        elif "desc_proj" in name: requires_grad = True
        elif gating and "gating" in name: requires_grad = True
        elif fc and name.startswith("fc."): requires_grad = True
        elif train_text_encoder and ("model.visual" not in name): requires_grad = True  # Text encoder (Transformer + token embeddings)

        for i in idxs:
            if f'lora{i}' in name:
                requires_grad = True
                break
                
        if requires_grad:
            names.append(name)
            params.append(param)
        param.requires_grad = requires_grad
    return names, params

def save_lora(model, path, fc=True, idxs=[], train_visual_proj=False, train_text_proj=False, train_text_encoder=False, gating=False):
    checkpoint = model.state_dict()
    keys = []
    for key in checkpoint.keys():
        if train_visual_proj and "model.visual.proj" in key: keys.append(key)
        elif train_text_proj and "model.text_projection" in key: keys.append(key)
        elif "desc_proj" in key: keys.append(key)
        elif gating and "gating" in key: keys.append(key)
        elif fc and key.startswith("fc."): keys.append(key)
        elif train_text_encoder and ("model.visual" not in key): keys.append(key)  # Text encoder (Transformer + token embeddings)
        
        for i in idxs:
            if f'lora{i}' in key: 
                keys.append(key)
                break
            
    checkpoint = {k:v for k,v in checkpoint.items() if k in keys}
    torch.save(checkpoint, path)
    
def load_lora(model, path, device='cuda:0'):
    model.load_state_dict(torch.load(path, map_location={device: 'cuda:0'}), strict=False)
    
def set_used_lora(model, idxs, visual_only=True, proj=False):
    target_blocks = [model.model.visual.transformer.resblocks] if visual_only else [model.model.visual.transformer.resblocks, model.model.transformer.resblocks]
    target_block_names = ["model.model.visual.transformer.resblocks"] if visual_only else ["model.model.visual.transformer.resblocks", "model.model.transformer.resblocks"]
    for target_block, target_block_name in zip(target_blocks, target_block_names):
        for name, submodule in target_block.named_modules():
            idx = name.split('.')[0]
            param = '.'.join(name.split('.')[1:])
            if isinstance(submodule, LoRAInjectedLinear): 
                eval(f"{target_block_name}[{idx}].{param}").used_lora = idxs
                eval(f"{target_block_name}[{idx}].{param}").lora_only = False
    
    if proj: 
        model.model.visual.proj.used_lora = idxs
        model.model.visual.proj.lora_only = False
        if not visual_only:
            model.model.text_projection.used_lora = idxs
            model.model.text_projection.lora_only = False

def set_used_lora_target(model, all_idxs, target_idxs, target_layer, visual_only=True):
    target_blocks = [model.model.visual.transformer.resblocks] if visual_only else [model.model.visual.transformer.resblocks, model.model.transformer.resblocks]
    target_block_names = ["model.model.visual.transformer.resblocks"] if visual_only else ["model.model.visual.transformer.resblocks", "model.model.transformer.resblocks"]
    
    for target_block, target_block_name in zip(target_blocks, target_block_names):
        for name, submodule in target_block.named_modules():
            idx = name.split('.')[0]
            param = '.'.join(name.split('.')[1:])
            if isinstance(submodule, LoRAInjectedLinear):
                if idx != target_layer:
                    eval(f"{target_block_name}[{idx}].{param}").used_lora = all_idxs
                    eval(f"{target_block_name}[{idx}].{param}").lora_only = False
                else:
                    eval(f"{target_block_name}[{idx}].{param}").used_lora = target_idxs
                    eval(f"{target_block_name}[{idx}].{param}").lora_only = True

