#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.

python -m src.train.train_pretrain \
  --model-config configs/model/qwen2b-continued-pretrain.yaml \
  --train-config configs/train/train_smoke.yaml \
  --data-config configs/data/data_mongolian.yaml \
  --deepspeed-config configs/deepspeed/ds_zero2.json

python -m src.train.train_pretrain \
  --model-config configs/model/qwen2b-continued-pretrain.yaml \
  --train-config configs/train/train_smoke.yaml \
  --data-config configs/data/data_mongolian.yaml \
  --deepspeed-config configs/deepspeed/ds_zero2.json \
  --resume

python -m src.eval.eval_perplexity \
  --model-config configs/model/qwen2b-continued-pretrain.yaml \
  --train-config configs/train/train_smoke.yaml \
  --data-config configs/data/data_mongolian.yaml

python -m src.eval.sample_generations \
  --model-config configs/model/qwen2b-continued-pretrain.yaml \
  --train-config configs/train/train_smoke.yaml
