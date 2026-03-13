from __future__ import annotations

import argparse
import re
from pathlib import Path

from src.utils.io import ensure_dir, read_yaml
from src.utils.logging import get_logger

LOGGER = get_logger("clean")


WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\ufeff", "")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize raw text files.")
    parser.add_argument("--data-config", default="configs/data/data_mongolian.yaml")
    args = parser.parse_args()

    cfg = read_yaml(args.data_config)
    raw_dir = Path(cfg["raw_dir"])
    interim_dir = ensure_dir(cfg["interim_dir"])

    txt_files = sorted(raw_dir.glob("*.txt"))
    if not txt_files:
        LOGGER.warning("No raw .txt files found in %s", raw_dir)
        return

    for path in txt_files:
        out_path = interim_dir / path.name
        cleaned_lines = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = normalize_text(line)
            if line:
                cleaned_lines.append(line)
        out_path.write_text("\n".join(cleaned_lines) + "\n", encoding="utf-8")
        LOGGER.info("Wrote cleaned text: %s", out_path)


if __name__ == "__main__":
    main()
