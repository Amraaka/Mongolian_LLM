import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import torch
from tqdm import tqdm

from utils import NLLBTranslator, cache_path


FIELDS = ("prompt", "chosen", "rejected")


def parse_args():
    p = argparse.ArgumentParser(description="Translate source DPO dataset en -> khk_Cyrl")
    p.add_argument("--source", default="orca",
                   help="Matches the --source name used in 1_download_source.py")
    p.add_argument("--model", default="facebook/nllb-200-3.3B",
                   choices=["facebook/nllb-200-distilled-600M",
                            "facebook/nllb-200-distilled-1.3B",
                            "facebook/nllb-200-3.3B"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_beams", type=int, default=2)
    p.add_argument("--max_input_tokens", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--chunk_rows", type=int, default=200,
                   help="Translate and flush this many rows per checkpoint")
    return p.parse_args()


def translate_in_batches(translator: NLLBTranslator,
                         texts: list, batch_size: int) -> list:
    out = []
    for i in range(0, len(texts), batch_size):
        out.extend(translator.translate_batch(texts[i:i + batch_size]))
    return out


def main():
    args = parse_args()

    src_path = cache_path(f"source_{args.source}.parquet")
    out_path = cache_path(f"translated_{args.source}.parquet")

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Run 1_download_source.py first; missing {src_path}")

    df = pd.read_parquet(src_path)
    n_total = len(df)

    if os.path.exists(out_path):
        done_df = pd.read_parquet(out_path)
        start = len(done_df)
        print(f"Resuming from row {start}/{n_total}")
    else:
        done_df = pd.DataFrame(columns=list(FIELDS))
        start = 0

    if start >= n_total:
        print("Nothing to translate.")
        return

    translator = NLLBTranslator(
        model_name=args.model,
        dtype=torch.float16,
        num_beams=args.num_beams,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
    )

    pbar = tqdm(total=n_total, initial=start, desc="rows")

    for chunk_start in range(start, n_total, args.chunk_rows):
        chunk_end = min(chunk_start + args.chunk_rows, n_total)
        chunk = df.iloc[chunk_start:chunk_end]

        translated = {}
        for field in FIELDS:
            translated[field] = translate_in_batches(
                translator, chunk[field].tolist(), args.batch_size
            )

        new_rows = pd.DataFrame(translated)
        done_df = pd.concat([done_df, new_rows], ignore_index=True)
        done_df.to_parquet(out_path)
        pbar.update(chunk_end - chunk_start)

    pbar.close()
    print(f"Translated {len(done_df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
