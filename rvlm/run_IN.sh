# Fixed
ARCH="ViT-B/16"
DATASET="imagenet"
LAST_NUM=12
NUM_WORKERS=32
EPOCHS=30


# 
BATCH_SIZE=256
TEST_BATCH_SIZE=128
LR=2e-4
WD=5e-5
#
PROMPT_ID=6
LORA_MODULES="q,k,v,out"
RANK=4
#DOT="--dot"
LAMBDA_DESC_ORTHO=1.0
SAVE_DIR="./experiments/models/IN/CLIP@LoRA_qkvout_256_kl1.0_lr2e-4_wd5e-5"




# 실행 명령어
python train_multi.py \
  --arch $ARCH \
  --dataset $DATASET \
  --prompt_id $PROMPT_ID \
  --lora_modules "$LORA_MODULES" \
  --last_num $LAST_NUM \
  --epochs $EPOCHS \
  --num_workers $NUM_WORKERS \
  --batch_size $BATCH_SIZE \
  --test_batch_size $TEST_BATCH_SIZE \
  --r $RANK \
  --kl \
  --lambda_desc_ortho $LAMBDA_DESC_ORTHO \
  --save_dir "$SAVE_DIR" \
  --lr_schedule \
  --lr $LR \
  --wd $WD