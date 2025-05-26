import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

from .loralib import LoRAInjectedMultiheadAttention


@dataclass
class HookData:
    attn_weights: dict=field(default_factory=dict)


class Hooker():
    def __init__(self, pipeline, num_lora=2):
        index = 1
        for layer in pipeline.model.visual.modules():
            if isinstance(layer, LoRAInjectedMultiheadAttention):
                print(layer)
                AttentionHooker(layer, self, index)
                index += 1

        self.pipeline = pipeline
        self.num_lora = num_lora
        self.data = HookData()


class AttentionHooker:
    def __init__(self, module: LoRAInjectedMultiheadAttention, hooker: Hooker, index: int=0):
        self.module = module
        self._hooker = hooker
        self._index = index

        self.module.forward = self.__call__
    
    def __call__(self, query, key, value, key_padding_mask= None, need_weights= True, attn_mask= None, average_attn_weights=True, is_causal=False):
        why_not_fast_path = ''
        if ((attn_mask is not None and torch.is_floating_point(attn_mask))
           or (key_padding_mask is not None) and torch.is_floating_point(key_padding_mask)):
            why_not_fast_path = "floating-point masks are not supported for fast path."

        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )
        
        if self.module.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))
                
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        
        if isinstance(embed_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(self.module.num_heads, rounding_mode="trunc")
        else:
            head_dim = embed_dim // self.module.num_heads
                
        feat_loss = torch.zeros(2, device=query.device, dtype=query.dtype)  # loss value, cnt
            
        
        
        q = self.module.q_proj(query)
        k = self.module.k_proj(key)
        v = self.module.v_proj(value)
        
        if isinstance(q, list): 
            feat_loss[0] += q[1]
            feat_loss[1] += 1
            q = q[0]
        if isinstance(k, list): 
            feat_loss[0] += k[1]
            feat_loss[1] += 1
            k = k[0]
        if isinstance(v, list): 
            feat_loss[0] += v[1]
            feat_loss[1] += 1
            v = v[0]
        
        #####################################
        self.module.q_proj.used_lora = []
        self.module.k_proj.used_lora = []
        q_variants = [self.module.q_proj(query)[0].detach().cpu()]
        k_variants = [self.module.k_proj(key)[0].detach().cpu()]
        for i in range(self.module.num_lora):
            self.module.q_proj.used_lora = [i]
            self.module.k_proj.used_lora = [i]
        
            q_variants.append(self.module.q_proj(query)[0].detach().cpu())
            k_variants.append(self.module.k_proj(key)[0].detach().cpu())

        self.module.q_proj.used_lora = list(range(self.module.num_lora))
        self.module.k_proj.used_lora = list(range(self.module.num_lora))

        q_variants.append(self.module.q_proj(query)[0].detach().cpu())
        k_variants.append(self.module.k_proj(key)[0].detach().cpu())
        #####################################

        if attn_mask is not None:
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                    )
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * self.module.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                    )
            else:
                raise RuntimeError(
                    f"attn_mask's dimension {attn_mask.dim()} is not supported"
                )
        
        # attn_output_all, attn_output_weights_all = {}, {}
        # for key in q_all:
            # q, k, v = q_all[key], k_all[key], v_all[key]
            
        if self.module.bias_k is not None and self.module.bias_v is not None:
            k = torch.cat([k, self.module.bias_k.repeat(1, bsz, 1)])
            k_variants = [torch.cat([tmp, self.module.bias_k.detach().cpu().repeat(1, bsz, 1)]) for tmp in k_variants]
            v = torch.cat([v, self.module.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        else:
            assert self.module.bias_k is None
            assert self.module.bias_v is None
            
        q = q.view(tgt_len, bsz * self.module.num_heads, head_dim).transpose(0, 1)
        k = k.view(k.shape[0], bsz * self.module.num_heads, head_dim).transpose(0, 1)
        v = v.view(v.shape[0], bsz * self.module.num_heads, head_dim).transpose(0, 1)
        
        #####################################
        q_variants = [tmp.view(tgt_len, bsz * self.module.num_heads, head_dim).transpose(0, 1) for tmp in q_variants]
        k_variants = [tmp.view(tgt_len, bsz * self.module.num_heads, head_dim).transpose(0, 1) for tmp in k_variants]
        #####################################
        
        if self.module.add_zero_attn:
            zero_attn_shape = (bsz * self.module.num_heads, 1, head_dim)
            k = torch.cat(
                [k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1
            )
            k_variants = [torch.cat([tmp, torch.zeros(zero_attn_shape, dtype=tmp.dtype, device=tmp.device)], dim=1) for tmp in k_variants]
            v = torch.cat(
                [v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1
            )
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
                
        src_len = k.size(1)
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (
                bsz,
                src_len,
            ), f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = (
                key_padding_mask.view(bsz, 1, 1, src_len)
                .expand(-1, self.module.num_heads, -1, -1)
                .reshape(bsz * self.module.num_heads, 1, src_len)
            )
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask
        
        #####################################
        _B, _Nt, E = q.shape
        attn_weights = []
        for q_tmp, k_tmp in zip(q_variants, k_variants):
            q_tmp_scaled = q_tmp * math.sqrt(1.0 / float(E))
            
            if attn_mask is not None:
                attn_weight = torch.baddbmm(
                    attn_mask, q_tmp_scaled, k_tmp.transpose(-2, -1)
                )
            else:
                attn_weight = torch.bmm(q_tmp_scaled, k_tmp.transpose(-2, -1))
            attn_weight = F.softmax(attn_weight, dim=-1)
            attn_weight = attn_weight.view(bsz, self.module.num_heads, tgt_len, src_len)
            attn_weight = attn_weight.mean(dim=1)
            attn_weights.append(attn_weight[:, 0, 1:])
        self._hooker.data.attn_weights[self._index] = attn_weights
        #####################################
                
        if need_weights:
            _B, _Nt, E = q.shape
            q_scaled = q * math.sqrt(1.0 / float(E))
            
            if attn_mask is not None:
                attn_output_weights = torch.baddbmm(
                    attn_mask, q_scaled, k.transpose(-2, -1)
                )
            else:
                attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
            attn_output_weights = F.softmax(attn_output_weights, dim=-1)

            attn_output = torch.bmm(attn_output_weights, v)

            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
            )
            attn_output = self.module.out_proj(attn_output)
            if isinstance(attn_output, list): 
                # if key=='org':
                feat_loss[0] += attn_output[1]
                feat_loss[1] += 1
                attn_output = attn_output[0]
            # if isinstance(attn_output, dict): attn_output = attn_output[key]
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

            attn_output_weights = attn_output_weights.view(bsz, self.module.num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)

            if not is_batched:
                attn_output = attn_output.squeeze(1)
                attn_output_weights = attn_output_weights.squeeze(0)
        else:
            if attn_mask is not None:
                if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                    attn_mask = attn_mask.unsqueeze(0)
                else:
                    attn_mask = attn_mask.view(bsz, self.module.num_heads, -1, src_len)

            q = q.view(bsz, self.module.num_heads, tgt_len, head_dim)
            k = k.view(bsz, self.module.num_heads, src_len, head_dim)
            v = v.view(bsz, self.module.num_heads, src_len, head_dim)

            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask, 0, is_causal
            )
            attn_output = (
                attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
            )

            attn_output = self.module.out_proj(attn_output)
            if isinstance(attn_output, list): 
                # if key=='org':
                feat_loss[0] += attn_output[1]
                feat_loss[1] += 1
                attn_output = attn_output[0]
            # if isinstance(attn_output, dict): attn_output = attn_output[key]
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
            if not is_batched:
                # squeeze the output if input was unbatched
                attn_output = attn_output.squeeze(1)
            attn_output_weights = None
        
        if self.module.batch_first and is_batched:
            attn_output = attn_output.transpose(1, 0)

        return attn_output, attn_output_weights, feat_loss