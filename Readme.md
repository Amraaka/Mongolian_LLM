# Mongolian LLM (Qwen2B Continued Pretraining)

This repository contains a practical starter pipeline for continued pretraining of Qwen2B on Mongolian text.

## Quick Start

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Copy environment template:
   - `cp .env.example .env`
4. Prepare input text files under `data/raw/` (one `.txt` file per source).
5. Run the data pipeline:
   - `bash scripts/run_data_pipeline.sh`
6. Launch a smoke training run:
   - `bash scripts/run_train_single.sh`
7. Run eval:
   - `bash scripts/run_eval.sh`
8. Run smoke test (includes resume + eval):
   - `bash scripts/run_smoke.sh`

## Environment Notes (CUDA / PyTorch)

- Install a PyTorch wheel matching your CUDA runtime.
- Example matrix:
  - CUDA 12.1 -> use PyTorch wheels tagged for cu121.
  - CUDA 11.8 -> use PyTorch wheels tagged for cu118.
- Verify GPU visibility:
  - `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
- If you only have CPU, keep smoke configs tiny and expect very slow runs.

## Project Structure

- `configs/`: YAML and DeepSpeed config files
- `src/data/`: data ingestion, cleaning, deduplication, and splitting
- `src/tokenization/`: tokenizer and tokenized dataset builders
- `src/train/`: continued pretraining script and training helpers
- `src/eval/`: perplexity and generation checks
- `scripts/`: executable shell wrappers
- `data/`: raw, interim, processed, and manifests
- `checkpoints/`: model checkpoints (gitignored)
- `logs/`: run logs (gitignored)

## Notes

- For multi-GPU runs, use `scripts/run_train_distributed.sh`.
- Adjust all hyperparameters through files in `configs/`.
- `configs/train/train_smoke.yaml` is for quick validation only.