#!/bin/bash
python src/3_train_dpo.py \
    --trainer Bokhbat \
    --peft qlora \
    --batch_size 2 \
    --eval_batch 1 \
    --steps 1500 \
    --lr 5e-6 \
    --grad_accum_step 8 \
    --warmup_step 50 \
    --save_version 0.1
