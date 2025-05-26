#!/bin/bash

# 모델 설정
ARCH="ViT-B/16"
DATASET="imagenet"
PROMPT_ID=6
LORA_MODULES="q,k,v,out"

# 평가 설정
NUM_WORKERS=32
BATCH_SIZE=512
EPOCH=4
SAVE_DIR="./experiments/models/IN/CLIP@LoRA_desc_multi_p6_dot1.0@q_k_v_out@r4"

# 실행 명령어
python eval.py \
  --arch $ARCH \
  --dataset $DATASET \
  --prompt_id $PROMPT_ID \
  --lora_modules "$LORA_MODULES" \
  --num_workers $NUM_WORKERS \
  --test_batch_size $BATCH_SIZE \
  --save_dir "$SAVE_DIR" \
  --epochs $EPOCH \
  --eval_org
