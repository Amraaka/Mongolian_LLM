  # save locally AND push to Amara/mongolian-dpo-orca (private)
#   python3 dataset_prep/3_filter_and_save.py --source orca --push                                                                 
                                                                
  # push publicly, or override repo name                                                                                         
#   python3 dataset_prep/3_filter_and_save.py --source orca --push --public
#   python3 dataset_prep/3_filter_and_save.py --source orca --push --hub_repo Amara/mn-dpo-v1 

import argparse
import os
import shutil
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import login

from utils import cache_path, project_root, row_is_valid


def parse_args():
    p = argparse.ArgumentParser(description="Filter translated DPO rows and save arrow dataset")
    p.add_argument("--source", default="orca")
    p.add_argument("--min_cyr", type=float, default=0.5,
                   help="Minimum Cyrillic letter ratio per field")
    p.add_argument("--min_len", type=int, default=3)
    p.add_argument("--max_len_ratio", type=float, default=5.0,
                   help="Max length ratio between chosen and rejected")
    p.add_argument("--out_subdir", default="data/dpo_data",
                   help="Where to save the final dataset (relative to project root)")
    p.add_argument("--val_size", type=float, default=0.1,
                   help="Fraction of rows for the validation split")
    p.add_argument("--test_size", type=float, default=0.1,
                   help="Fraction of rows for the test split")
    p.add_argument("--seed", type=int, default=42,
                   help="Shuffle seed for the train/val/test split")
    p.add_argument("--push", action="store_true",
                   help="Also push the dataset to HuggingFace Hub")
    p.add_argument("--hub_repo", default=None,
                   help="Hub repo id (default: Bokhbat/mongolian-dpo-<source>)")
    p.add_argument("--public", action="store_true",
                   help="Make the hub repo public (default: private)")
    return p.parse_args()


def main():
    args = parse_args()

    in_path = cache_path(f"translated_{args.source}.parquet")
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Run 2_translate_nllb.py first; missing {in_path}")

    df = pd.read_parquet(in_path)
    n_in = len(df)

    rows = df.to_dict(orient="records")
    kept = [r for r in rows
            if row_is_valid(r, args.min_cyr, args.min_len, args.max_len_ratio)]
    n_out = len(kept)

    print(f"Input rows:    {n_in}")
    print(f"Kept rows:     {n_out}")
    print(f"Dropped rows:  {n_in - n_out} ({(n_in - n_out) / max(n_in, 1):.1%})")

    if n_out == 0:
        raise RuntimeError("All rows filtered out — relax thresholds or check translation quality.")

    if not 0 < args.val_size + args.test_size < 1:
        raise ValueError("val_size + test_size must be in (0, 1)")

    ds = Dataset.from_list(kept)

    test_frac = args.test_size
    val_frac_of_remaining = args.val_size / (1 - args.test_size)

    split_a = ds.train_test_split(test_size=test_frac, seed=args.seed)
    split_b = split_a["train"].train_test_split(test_size=val_frac_of_remaining, seed=args.seed)
    splits = DatasetDict({
        "train": split_b["train"],
        "validation": split_b["test"],
        "test": split_a["test"],
    })

    out_dir = os.path.join(project_root(), args.out_subdir)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    splits.save_to_disk(out_dir)
    sizes = {k: len(v) for k, v in splits.items()}
    print(f"Saved splits {sizes} -> {out_dir}")

    if args.push:
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN not set in .env — cannot push to hub.")
        login(token=token)

        repo_id = args.hub_repo or f"Amara/mongolian-dpo-{args.source}"
        private = not args.public
        print(f"Pushing to hub: {repo_id} (private={private})")
        splits.push_to_hub(repo_id, private=private, token=token)
        print(f"Pushed -> https://huggingface.co/datasets/{repo_id}")

    print("Next: run `python3 src/3_train_dpo.py --peft qlora ...`")


if __name__ == "__main__":
    main()
