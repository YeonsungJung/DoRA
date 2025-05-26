#!/bin/bash

# Define LoRA configurations
lora_nums=(4)
ranks=(2 4 8 16)
lambda_values=(0.0)

# Loop for lambda_param_ortho
for num_lora in "${lora_nums[@]}"; do
    for r in "${ranks[@]}"; do
        for lambda_param_ortho in "${lambda_values[@]}"; do
            python train_multi.py --num_lora $num_lora --r $r
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
