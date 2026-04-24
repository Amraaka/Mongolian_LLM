#   Smoke-test path:                                                                                       
#   python3 dataset_prep/1_download_source.py --source truthy --limit 50                                                           
#   python3 dataset_prep/2_translate_nllb.py  --source truthy --model facebook/nllb-200-distilled-600M                             
#   python3 dataset_prep/3_filter_and_save.py --source truthy                                                                      
                                                                                                                                 

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login

from utils import cache_path


SOURCES = {
    "orca": {
        "hub_id": "Intel/orca_dpo_pairs",
        "split": "train",
        "columns": {"prompt": "question", "chosen": "chosen", "rejected": "rejected"},
    },
    "ultrafeedback": {
        "hub_id": "HuggingFaceH4/ultrafeedback_binarized",
        "split": "train_prefs",
        "columns": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
        "chat_format": True,
    },
    "truthy": {
        "hub_id": "jondurbin/truthy-dpo-v0.1",
        "split": "train",
        "columns": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="Download source English DPO dataset")
    p.add_argument("--source", choices=list(SOURCES.keys()), default="orca")
    p.add_argument("--limit", type=int, default=None,
                   help="Keep only the first N rows (for quick tests)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)

    args = parse_args()
    cfg = SOURCES[args.source]

    ds = load_dataset(cfg["hub_id"], split=cfg["split"])

    mapping = cfg["columns"]
    ds = ds.rename_columns({v: k for k, v in mapping.items() if v != k})
    ds = ds.select_columns(list(mapping.keys()))

    if cfg.get("chat_format"):
        def last_assistant(msgs):
            if not isinstance(msgs, list):
                return ""
            for m in reversed(msgs):
                if isinstance(m, dict) and m.get("role") == "assistant":
                    return m.get("content", "") or ""
            return ""

        ds = ds.map(lambda r: {
            "chosen": last_assistant(r["chosen"]),
            "rejected": last_assistant(r["rejected"]),
        })

    ds = ds.filter(lambda r: all(isinstance(r[k], str) and r[k].strip()
                                 for k in ("prompt", "chosen", "rejected")))

    if args.limit is not None:
        ds = ds.shuffle(seed=args.seed).select(range(min(args.limit, len(ds))))

    out = cache_path(f"source_{args.source}.parquet")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ds.to_parquet(out)

    print(f"Saved {len(ds)} rows -> {out}")


if __name__ == "__main__":
    main()
