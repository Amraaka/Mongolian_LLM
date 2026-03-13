from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from src.utils.io import ensure_dir, read_jsonl, read_yaml
from src.utils.logging import get_logger

LOGGER = get_logger("build_dataset")


def tokenize_rows(tokenizer: AutoTokenizer, rows: list[dict[str, str]], max_len: int) -> Dataset:
    ds = Dataset.from_list(rows)

    def _tok(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        return tokenized

    cols = [c for c in ds.column_names if c not in {"text"}]
    return ds.map(_tok, batched=True, remove_columns=cols)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tokenized dataset shards.")
    parser.add_argument("--model-config", default="configs/model/qwen2b-continued-pretrain.yaml")
    parser.add_argument("--train-config", default="configs/train/train_base.yaml")
    parser.add_argument("--data-config", default="configs/data/data_mongolian.yaml")
    args = parser.parse_args()

    model_cfg = read_yaml(args.model_config)
    train_cfg = read_yaml(args.train_config)
    data_cfg = read_yaml(args.data_config)

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model"], trust_remote_code=bool(model_cfg.get("trust_remote_code", True))
    )
    max_len = int(train_cfg["max_seq_length"])

    train_rows = read_jsonl(Path(data_cfg["train_manifest"]))
    val_rows = read_jsonl(Path(data_cfg["val_manifest"]))
    train_ds = tokenize_rows(tokenizer, train_rows, max_len)
    val_ds = tokenize_rows(tokenizer, val_rows, max_len)

    train_out = ensure_dir(data_cfg["tokenized_train_path"])
    val_out = ensure_dir(data_cfg["tokenized_val_path"])
    train_ds.save_to_disk(str(train_out))
    val_ds.save_to_disk(str(val_out))

    LOGGER.info("Saved tokenized train dataset to %s", train_out)
    LOGGER.info("Saved tokenized val dataset to %s", val_out)


if __name__ == "__main__":
    main()
