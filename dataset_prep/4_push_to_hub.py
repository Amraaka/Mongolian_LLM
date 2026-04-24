#   python3 dataset_prep/4_push_to_hub.py --source orca --hub_repo <user_or_org>/mongolian-dpo-orca
#   python3 dataset_prep/4_push_to_hub.py --source orca --hub_repo <user_or_org>/mongolian-dpo-orca --public

import argparse
import os
import sys
import tempfile
import textwrap

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets import Dataset, DatasetDict, load_from_disk
from dotenv import load_dotenv
from huggingface_hub import HfApi, login

from utils import project_root


README_TEMPLATE = """\
---
license: apache-2.0
task_categories:
- text-generation
language:
- mn
tags:
- dpo
- preference
- mongolian
- rlhf
size_categories:
- {size_bucket}
configs:
- config_name: default
  data_files:
{data_files_yaml}
dataset_info:
  features:
  - name: prompt
    dtype: string
  - name: chosen
    dtype: string
  - name: rejected
    dtype: string
  splits:
{splits_yaml}
---

# {repo_name}

Mongolian (Cyrillic) DPO preference pairs, machine-translated from
[`{source_hub_id}`]({source_url}) with `facebook/nllb-200-3.3B` and filtered
for Cyrillic ratio, minimum length, and chosen/rejected length balance.

## Schema

| column    | type   | description                       |
|-----------|--------|-----------------------------------|
| prompt    | string | user prompt (Mongolian Cyrillic)  |
| chosen    | string | preferred response                |
| rejected  | string | dispreferred response             |

## Stats

- Rows: {num_rows}
- Source: `{source_hub_id}`
- Translator: `facebook/nllb-200-3.3B` (eng_Latn -> khk_Cyrl)

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="train")
print(ds[0])
```
"""

SOURCE_URLS = {
    "orca": ("Intel/orca_dpo_pairs", "https://huggingface.co/datasets/Intel/orca_dpo_pairs"),
    "ultrafeedback": ("HuggingFaceH4/ultrafeedback_binarized",
                      "https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized"),
    "truthy": ("jondurbin/truthy-dpo-v0.1", "https://huggingface.co/datasets/jondurbin/truthy-dpo-v0.1"),
}


def size_bucket(n: int) -> str:
    if n < 1_000:
        return "n<1K"
    if n < 10_000:
        return "1K<n<10K"
    if n < 100_000:
        return "10K<n<100K"
    if n < 1_000_000:
        return "100K<n<1M"
    return "1M<n<10M"


def parse_args():
    p = argparse.ArgumentParser(description="Push filtered DPO dataset to HF Hub (orca-style repo)")
    p.add_argument("--source", default="orca", choices=list(SOURCE_URLS.keys()))
    p.add_argument("--local_dir", default="data/dpo_data",
                   help="Path (relative to project root) holding the saved_to_disk dataset")
    p.add_argument("--hub_repo", required=True,
                   help="Target repo id, e.g. <username>/mongolian-dpo-orca")
    p.add_argument("--public", action="store_true",
                   help="Make the repo public (default: private)")
    return p.parse_args()


def main():
    args = parse_args()

    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set in .env — cannot push to hub.")
    login(token=token)

    local_path = os.path.join(project_root(), args.local_dir)
    if not os.path.isdir(local_path):
        raise FileNotFoundError(
            f"{local_path} not found — run 3_filter_and_save.py first."
        )

    ds = load_from_disk(local_path)
    if isinstance(ds, Dataset):
        ds = DatasetDict({"train": ds})

    split_sizes = {name: len(d) for name, d in ds.items()}
    total_rows = sum(split_sizes.values())
    print(f"Loaded splits {split_sizes} from {local_path}")

    private = not args.public
    print(f"Pushing parquet shards to {args.hub_repo} (private={private})")
    ds.push_to_hub(args.hub_repo, private=private, token=token)

    data_files_yaml = "\n".join(
        f"  - split: {name}\n    path: data/{name}-*.parquet" for name in ds.keys()
    )
    splits_yaml = "\n".join(
        f"  - name: {name}\n    num_examples: {n}" for name, n in split_sizes.items()
    )

    source_hub_id, source_url = SOURCE_URLS[args.source]
    readme = README_TEMPLATE.format(
        repo_id=args.hub_repo,
        repo_name=args.hub_repo.split("/")[-1],
        source_hub_id=source_hub_id,
        source_url=source_url,
        num_rows=total_rows,
        size_bucket=size_bucket(total_rows),
        data_files_yaml=data_files_yaml,
        splits_yaml=splits_yaml,
    )

    api = HfApi(token=token)
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(textwrap.dedent(readme))
        readme_path = f.name
    try:
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=args.hub_repo,
            repo_type="dataset",
            commit_message="Add dataset card",
        )
    finally:
        os.unlink(readme_path)

    print(f"Pushed -> https://huggingface.co/datasets/{args.hub_repo}")


if __name__ == "__main__":
    main()
