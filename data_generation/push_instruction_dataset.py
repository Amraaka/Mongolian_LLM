"""
Push the Step 4 web-search instruction dataset to the HuggingFace Hub.

Loads the raw Dataset from data/instruction_data/raw (built by
build_search_dataset.py) and pushes it as a single split to the configured
hub repo. After a successful push, prints the YAML snippet to drop into
configs/data_locations.yaml so the loader can pull from the Hub.

Usage:
  python data_generation/push_instruction_dataset.py
  python data_generation/push_instruction_dataset.py --repo Amraaka/other-name --private
"""
import argparse
import os
import sys
from pathlib import Path

from datasets import load_from_disk
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

CURRENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CURRENT_DIR / "data" / "instruction_data" / "raw"
DEFAULT_REPO = "Amraaka/mongolian-web-search-dataset"


def parse_args():
    p = argparse.ArgumentParser(description="Push the web-search instruction dataset to HF Hub")
    p.add_argument("--repo", default=DEFAULT_REPO,
                   help=f"HF repo id, e.g. user/name (default: {DEFAULT_REPO})")
    p.add_argument("--data-dir", default=str(DATA_DIR),
                   help=f"Path to the saved Dataset (default: {DATA_DIR})")
    p.add_argument("--private", action="store_true",
                   help="Create the repo as private")
    p.add_argument("--commit-message", default="Add Mongolian web-search function-calling dataset",
                   help="Commit message for the push")
    return p.parse_args()


def main():
    args = parse_args()

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not set in environment (.env). Export an HF write token first.")
    login(token=token)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        sys.exit(f"Dataset directory not found: {data_dir}\n"
                 f"Run data_generation/build_search_dataset.py first.")

    print(f"Loading dataset from {data_dir}")
    ds = load_from_disk(str(data_dir))
    print(f"Loaded {len(ds)} rows with columns: {ds.column_names}")

    print(f"\nPushing to https://huggingface.co/datasets/{args.repo} "
          f"({'private' if args.private else 'public'})")
    ds.push_to_hub(
        repo_id=args.repo,
        private=args.private,
        commit_message=args.commit_message,
    )

    print("\nDone.")
    print("Update configs/data_locations.yaml so the loader can fall back to the Hub:")
    print(f"  instruction_data:\n    ...\n    hub_id: {args.repo}")


if __name__ == "__main__":
    main()
