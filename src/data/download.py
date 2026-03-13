from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.io import ensure_dir, read_yaml
from src.utils.logging import get_logger

LOGGER = get_logger("download")


def main() -> None:
    parser = argparse.ArgumentParser(description="Data source download placeholder.")
    parser.add_argument(
        "--data-config",
        default="configs/data/data_mongolian.yaml",
        help="Path to data YAML config.",
    )
    args = parser.parse_args()
    cfg = read_yaml(args.data_config)

    raw_dir = ensure_dir(cfg["raw_dir"])
    sample_path = Path(raw_dir) / "sample_mn.txt"
    if not sample_path.exists():
        sample_path.write_text(
            "Энэ бол Монгол хэлний жишээ өгөгдөл.\n",
            encoding="utf-8",
        )
        LOGGER.info("Created sample raw text at %s", sample_path)
    else:
        LOGGER.info("Raw sample already exists at %s", sample_path)

    LOGGER.info("Download step placeholder completed.")


if __name__ == "__main__":
    main()
