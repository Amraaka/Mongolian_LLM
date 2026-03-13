from pathlib import Path


def latest_checkpoint(output_dir: str) -> str | None:
    path = Path(output_dir)
    if not path.exists():
        return None
    checkpoints = sorted(path.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        return None
    return str(checkpoints[-1])
