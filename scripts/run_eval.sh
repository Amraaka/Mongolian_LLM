#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.

python -m src.eval.eval_perplexity \
  --model-config configs/model/qwen2b-continued-pretrain.yaml \
  --train-config configs/train/train_base.yaml \
  --data-config configs/data/data_mongolian.yaml

python -m src.eval.sample_generations \
  --model-config configs/model/qwen2b-continued-pretrain.yaml \
  --train-config configs/train/train_base.yaml
