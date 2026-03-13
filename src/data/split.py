from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.utils.io import ensure_dir, read_yaml, write_jsonl
from src.utils.logging import get_logger

LOGGER = get_logger("split")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val manifests.")
    parser.add_argument("--data-config", default="configs/data/data_mongolian.yaml")
    args = parser.parse_args()
    cfg = read_yaml(args.data_config)

    interim_dir = Path(cfg["interim_dir"])
    manifest_dir = ensure_dir(cfg["manifest_dir"])
    val_ratio = float(cfg.get("val_ratio", 0.01))
    seed = int(cfg.get("random_seed", 42))

    rows: list[dict[str, str]] = []
    for path in sorted(interim_dir.glob("*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text:
                rows.append({"text": text, "source": path.name})

    if len(rows) < 2:
        raise ValueError("Need at least 2 records to split train/val.")

    train_rows, val_rows = train_test_split(rows, test_size=val_ratio, random_state=seed)

    train_path = Path(cfg["train_manifest"])
    val_path = Path(cfg["val_manifest"])
    ensure_dir(train_path.parent)
    ensure_dir(val_path.parent)
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    LOGGER.info("Manifest dir: %s", manifest_dir)
    LOGGER.info("Train rows: %d -> %s", len(train_rows), train_path)
    LOGGER.info("Val rows: %d -> %s", len(val_rows), val_path)


if __name__ == "__main__":
    main()
