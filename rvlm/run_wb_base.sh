# Fixed
NUM_WORKERS=32
EPOCHS=200
NUM_LORA=2

# 
BATCH_SIZE=32
TEST_BATCH_SIZE=64
LR=1e-4
WD=5e-5
#
LORA_MODULES="q,k,v,out"
RANK=4
SAVE_DIR="./experiments/models/WB/CLIP@LoRA_${LORA_MODULES//,/""}_base_numLora${NUM_LORA}_batch${BATCH_SIZE}"

# SAVE_DIR="./experiments/models/WB/CLIP@LoRA_${LORA_MODULES//,/""}_prompt${PROMPT_ID}_batch${BATCH_SIZE}_usingpre_dot${LAMBDA_DESC_ORTHO}_l1${LAMBDA_REG}"
# 실행 명령어
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_multi.py \
  --lora_modules "$LORA_MODULES" \
  --epochs $EPOCHS \
  --num_workers $NUM_WORKERS \
  --batch_size $BATCH_SIZE \
  --test_batch_size $TEST_BATCH_SIZE \
  --r $RANK \
  --save_dir "$SAVE_DIR" \
  --lr $LR \
  --wd $WD \
  --num_lora $NUM_LORA\
  --train_visual_proj
  # --l1 \
  # --lambda_reg 0.1
#  --usingPre
#  --lr_schedule \
