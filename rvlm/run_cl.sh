#!/bin/bash

# Define LoRA configurations
lora_nums=(2)
ranks=(32)
lambda_values=(1.0)

# Loop for lambda_param_ortho
for num_lora in "${lora_nums[@]}"; do
    for r in "${ranks[@]}"; do
        for pcaOrtho_lambda in "${lambda_values[@]}"; do
            python train_multi.py --num_lora $num_lora --r $r --pcaOrtho_lambda $pcaOrtho_lambda --lora_modules q,k,v,out,mlp --batch_size 256 --lora_dropout 0.1 --n_pca 100 --lora_alpha 0.0 #--lora_alpha 4 #--feat_kk #--text_cls #--desc_lambda 0.5
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
