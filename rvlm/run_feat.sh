#!/bin/bash

# Define LoRA configurations
lora_nums=(2)
ranks=(256)
lambda_values=(1.0)

# Loop for lambda_param_ortho
for num_lora in "${lora_nums[@]}"; do
    for r in "${ranks[@]}"; do
        for lambda_feat_ortho in "${lambda_values[@]}"; do
            python train_multi.py --num_lora $num_lora --r $r --lambda_feat_ortho $lambda_feat_ortho --lora_modules q,k,v,out,mlp --batch_size 128 #--desc_cls --desc_lambda 0.5 #--ortho_pretrained --loss_gram #--lora_alpha 4 #--feat_kk #--text_cls #--desc_lambda 0.5
        done
    done
done


# # Loop for lambda_feat_ortho
# for num_lora in "${lora_nums[@]}"; do
#     for r in "${ranks[@]}"; do
#         for lambda_feat_ortho in "${lambda_values[@]}"; do
#             python train_multi.py --num_lora $num_lora --r $r --lambda_feat_ortho $lambda_feat_ortho
#         done
#     done
# done
