from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from src.utils.io import read_jsonl, read_yaml
from src.utils.logging import get_logger

LOGGER = get_logger("train_tokenizer")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenizer preparation placeholder.")
    parser.add_argument("--model-config", default="configs/model/qwen2b-continued-pretrain.yaml")
    parser.add_argument("--data-config", default="configs/data/data_mongolian.yaml")
    args = parser.parse_args()

    model_cfg = read_yaml(args.model_config)
    data_cfg = read_yaml(args.data_config)

    train_manifest = Path(data_cfg["train_manifest"])
    rows = read_jsonl(train_manifest)
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model"], trust_remote_code=bool(model_cfg.get("trust_remote_code", True))
    )

    LOGGER.info("Loaded tokenizer vocab size: %d", tokenizer.vocab_size)
    LOGGER.info("Training manifest records: %d", len(rows))
    LOGGER.info("Tokenizer training step is optional for continued pretraining and currently no-op.")


if __name__ == "__main__":
    main()
