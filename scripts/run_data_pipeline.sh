#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.

python -m src.data.download --data-config configs/data/data_mongolian.yaml
python -m src.data.clean --data-config configs/data/data_mongolian.yaml
python -m src.data.dedupe --data-config configs/data/data_mongolian.yaml
python -m src.data.split --data-config configs/data/data_mongolian.yaml
python -m src.tokenization.train_tokenizer --model-config configs/model/qwen2b-continued-pretrain.yaml --data-config configs/data/data_mongolian.yaml
python -m src.tokenization.build_dataset --model-config configs/model/qwen2b-continued-pretrain.yaml --train-config configs/train/train_base.yaml --data-config configs/data/data_mongolian.yaml

echo "Data pipeline completed successfully."
