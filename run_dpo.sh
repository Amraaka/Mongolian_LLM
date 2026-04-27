#!/bin/bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
python src/3_train_dpo.py \
    --trainer Bokhbat \
    --peft qlora \
    --batch_size 1 \
    --eval_batch 1 \
    --steps 1500 \
    --lr 5e-6 \
    --grad_accum_step 16 \
    --warmup_step 50 \
    --save_version 0.1
