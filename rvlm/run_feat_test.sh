#!/bin/bash

# Define LoRA configurations
lora_nums=(2)
ranks=(2)
lambda_values=(100.0)

# Loop for lambda_param_ortho
for num_lora in "${lora_nums[@]}"; do
    for r in "${ranks[@]}"; do
        for lambda_feat_ortho in "${lambda_values[@]}"; do
            python train_multi_test.py --num_lora $num_lora --r $r --lambda_feat_ortho $lambda_feat_ortho --desc_cls #--feat_kk
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
