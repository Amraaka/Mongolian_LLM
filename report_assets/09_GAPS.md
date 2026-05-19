# 9. Issues & Gaps Report

What the report requires but **could not be found in the repository**, plus how to fill each gap.

## 9.1 Missing data / artifacts

| # | Gap | Evidence |
|---|-----|----------|
| 1 | **Stage-1 pretraining corpus absent** | `data/text_data/` does not exist; only referenced in `configs/data_locations.yaml`. No samples, token counts, or size computable. |
| 2 | **Stage-1 pretrained model absent** | No `models/..._mongolian_text_0.1/` dir; `run_pipeline.sh` Stage-1 line is commented out. |
| 3 | **Stage-1 training metrics absent** | No log, no `trainer_state.json` → no pretraining loss curve, no **perplexity before/after** (Fig 1, Fig 6 cannot be made). |
| 4 | **Stage-2 QA model absent locally** | No `models/..._mongolian_qa_0.3/` dir (only on HF hub). |
| 5 | **Stage-2 training metrics absent** | No QA log / `trainer_state.json` → no QA loss curve (Fig 2). |
| 6 | **No EM / F1 results anywhere** | Eval code exists (`src/2_finetune_qa.py`, `test/test_qa_model.py`, `test/test_qa_model_v0.3.ipynb`) but the notebook's score cell has **no saved output**; no results file. Fig 7 + §8.1 = TBD. |
| 7 | **No saved inference outputs** | No `outputs/` / `predictions/` / `generations/` dirs. Only QA-tuned (ver_0.3) sample generations exist (notebook cell 14). Cross-model comparison (§6 of spec) impossible from repo. |
| 8 | **Base model `config.json` absent** | Only QLoRA adapters saved → hidden size / #layers / #heads / exact param count not readable (need upstream `Qwen/Qwen3.5-2B-Base`). |
| 9 | **`training_args.bin` not deserializable** | Requires the `unsloth` package (not importable here) → optimizer β/ε/weight_decay reported as TRL/HF defaults, not confirmed values. |
| 10 | **Config vs on-disk split mismatch** | `configs/data_split_partition.yaml` says DPO 8995/500/500 and QA 98000/1025/10000; actual DPO on-disk = 7496/625/1874 (matches training log), instruction test = 50 not 100. On-disk Arrow splits are authoritative. |
| 11 | **Empty scripts** | `src/0.preprocess.py` and `src/5_vl_fine_tuning.py` are 0 bytes (no preprocessing/VL code). |
| 12 | **Stage-1 epoch bug** | `SFTConfig(num_train_epoch=...)` is misspelled (correct: `num_train_epochs`) → silently ignored; only `max_steps` would govern Stage 1. Worth noting in the report's limitations. |
| 13 | **Accidental Stage-4 re-run** | Search log has a second invocation that resumed from checkpoint-57 doing no training (`train_runtime=0.0085s`, `train_loss=0.0`). Ignore it; use the real run (steps 10→57). |

## 9.2 Figures skipped (see `figures/MISSING.md`)

- `fig_01_pretrain_loss.png` — no Stage-1 data.
- `fig_02_qa_loss.png` — no Stage-2 data.
- `fig_06_perplexity_comparison.png` — no perplexity values.
- `fig_07_em_f1_comparison.png` — no EM/F1 values.
- `fig_05_all_losses_combined.png` — generated, but stages 1 & 2 are placeholders.

## 9.3 What you must provide / run manually

1. **Run Stage 1** (or supply its log / `trainer_state.json` + `text_data`) to get
   the pretraining loss curve and perplexity before/after.
2. **Compute perplexity** of base vs pretrained model on a held-out Mongolian text
   set (`src/1_pretrain_text.py:compute_metrics` already returns it — just run eval).
3. **Compute EM / F1** for all 5 checkpoints on `qa_data` test:

   ```bash
   # adjust configs/saved_model_location.yaml step2 path, then:
   python test/test_qa_model.py        # prints EM% and F1% for that checkpoint
   ```
   Repeat per model version; record into `report_assets/data/eval_results.csv`
   and `08_summary_table.md`.
4. **Generate sample outputs** for the 5 prompts in `INFERENCE_NEEDED.md` across
   base / pretrained / QA / DPO / instruction checkpoints; fill `05_sample_outputs.md`.
5. **Retrieve base model architecture** from the HF model card / `config.json` of
   `Qwen/Qwen3.5-2B-Base` to complete §1.1 (hidden size, layers, heads, exact params).
6. **Pull Stage-1/Stage-2 adapters from HF** (`Ganaa0614/...text-ver_0.1`,
   `...qa-ver_0.3`) if local checkpoints are needed for evaluation.

## 9.4 Suggested gap-filling commands

```bash
# EM/F1 for a checkpoint (edit step2 local_path/hub_id in the yaml first)
python test/test_qa_model.py

# Perplexity-only eval of pretrained vs base (sketch — reuse compute_metrics)
#   load model, run trainer.evaluate() on qa/text validation, read 'perplexity'

# Re-extract this report's data after new logs appear
venv/bin/python report_assets/_extract.py
venv/bin/python report_assets/_figures.py
```

## 9.5 Data that IS solid (no gap)

- DPO training: full step-by-step loss/reward log + best checkpoint (Stage 3).
- Stage-4 search training: full loss log + checkpoint (Stage 4).
- All three present datasets (QA, DPO, instruction): exact row counts, sizes,
  token statistics, samples, preprocessing code.
- All hyperparameters from training scripts + run scripts + logs.
- Full environment / dependency list.
