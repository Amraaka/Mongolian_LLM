from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from src.utils.io import read_yaml
from src.utils.logging import get_logger

LOGGER = get_logger("dedupe")


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate interim text files.")
    parser.add_argument("--data-config", default="configs/data/data_mongolian.yaml")
    args = parser.parse_args()
    cfg = read_yaml(args.data_config)

    interim_dir = Path(cfg["interim_dir"])
    min_chars = int(cfg.get("min_chars", 40))
    files = sorted(interim_dir.glob("*.txt"))
    if not files:
        LOGGER.warning("No interim files found in %s", interim_dir)
        return

    seen: set[str] = set()
    total, kept = 0, 0
    for path in files:
        out_lines: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            total += 1
            line = line.strip()
            if len(line) < min_chars:
                continue
            h = content_hash(line)
            if h in seen:
                continue
            seen.add(h)
            out_lines.append(line)
            kept += 1
        path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
        LOGGER.info("Deduped %s -> %d lines", path.name, len(out_lines))

    LOGGER.info("Deduplication done: kept %d / %d lines", kept, total)


if __name__ == "__main__":
    main()
