# DPO Dataset Preparation

Translate an English DPO dataset into Mongolian (Khalkha, Cyrillic) using
NLLB-200 and save it where `src/3_train_dpo.py` expects it.

Final output schema (read by `utils/utils.py::CustomDataLoader.format_dpo`):

| column   | content                       |
|----------|-------------------------------|
| prompt   | user instruction (Mongolian)  |
| chosen   | preferred response            |
| rejected | worse response                |

Saved to `data/dpo_data/` as a HuggingFace `datasets` arrow folder.
The DPO trainer's loader will split it 80/10/10 on first run.

## Pipeline

```
1_download_source.py  -> dataset_prep/cache/source_<name>.parquet
2_translate_nllb.py   -> dataset_prep/cache/translated_<name>.parquet  (resumable)
3_filter_and_save.py  -> data/dpo_data/                                (final arrow)
```

## Quick start (recommended for RTX 5080, 16 GB)

```bash
# 1) Smoke test with a tiny slice (5 min total)
python3 dataset_prep/1_download_source.py --source truthy --limit 50
python3 dataset_prep/2_translate_nllb.py  --source truthy \
        --model facebook/nllb-200-distilled-600M --batch_size 16
python3 dataset_prep/3_filter_and_save.py --source truthy

# 2) Real run with Intel/orca_dpo_pairs (~12k rows)
python3 dataset_prep/1_download_source.py --source orca
python3 dataset_prep/2_translate_nllb.py  --source orca \
        --model facebook/nllb-200-3.3B --batch_size 16 --num_beams 2
python3 dataset_prep/3_filter_and_save.py --source orca
```

## Sources

| --source        | hub_id                                                      | rows  |
|-----------------|-------------------------------------------------------------|-------|
| truthy          | jondurbin/truthy-dpo-v0.1                                   | ~1k   |
| orca            | Intel/orca_dpo_pairs                                        | ~12k  |
| ultrafeedback   | argilla/ultrafeedback-binarized-preferences-cleaned         | ~60k  |

Use `--limit N` on step 1 to subsample before translating.

## Models (translation, fp16)

| model                                  | VRAM  | speed | quality |
|----------------------------------------|-------|-------|---------|
| facebook/nllb-200-distilled-600M       | ~2 GB | fast  | OK      |
| facebook/nllb-200-distilled-1.3B       | ~3 GB | mid   | better  |
| facebook/nllb-200-3.3B (default)       | ~7 GB | slow  | best    |

## Resuming step 2

`2_translate_nllb.py` writes `translated_<source>.parquet` every `--chunk_rows`
rows. If interrupted, re-run the same command — it picks up from the row count
already saved.

## Filtering

Step 3 drops rows where any field is empty / too short, where Cyrillic letter
ratio falls below `--min_cyr` (translation likely failed), where
`chosen == rejected`, or where chosen/rejected length ratio exceeds
`--max_len_ratio`. Tune these if too many rows are dropped.

## Auth

Reads `HF_TOKEN` from `.env` at the project root, same as the training scripts.
