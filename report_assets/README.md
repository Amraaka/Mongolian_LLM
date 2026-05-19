# Report Assets — Mongolian LLM 4-Stage Pipeline

Auto-extracted, read-only analysis of the training repo for the academic report.
No training file was modified; everything here is generated from repo contents.

## Documents (paste-ready for LaTeX, UTF-8)

| File | Section |
|------|---------|
| `01_overview.md` | Base model, architecture, tokenizer, repo structure, pipeline order, README |
| `02_datasets.md` | Per-stage dataset stats, samples, preprocessing |
| `03_hyperparameters.md` | Per-stage hyperparameter tables |
| `04_metrics.md` | Loss/eval tables (DPO & Stage-4); pretrain/QA = not in repo |
| `05_sample_outputs.md` | Available QA-tuned generations |
| `06_code_snippets.md` | Methodology code excerpts (LaTeX `listings`) |
| `07_environment.md` | Python, libraries, CUDA, GPU, env vars |
| `08_summary_table.md` | Final comparison table (EM/F1 = TBD) |
| `09_GAPS.md` | Everything missing + how to fill it |
| `INFERENCE_NEEDED.md` | 5 prompts to run per checkpoint |

## Data & figures

- `samples/` — `pretrain_samples.json` (missing-marker), `qa_samples.json`,
  `dpo_samples.json`, `instruction_samples.json` (5 each).
- `data/loss_curves.csv` — `stage,step,loss,eval_loss,learning_rate` (DPO + Stage 4).
- `data/eval_results.csv` — `model_version,metric_name,value` (EM/F1/ppl = "not found in repo").
- `data/_log_history.json`, `data/_dataset_report.json`, `data/instruction_token_lengths.json` — raw.
- `figures/` — `fig_03_dpo_loss`, `fig_03b_dpo_rewards`, `fig_04_instruction_loss`,
  `fig_05_all_losses_combined`, `fig_08_token_distribution`, `fig_09_pipeline_diagram`
  (300 DPI PNG). `figures/MISSING.md` lists figs skipped for lack of data.

## Reproduce

```bash
venv/bin/python report_assets/_extract.py    # datasets, samples, CSVs
venv/bin/python report_assets/_figures.py    # figures (needs matplotlib in venv)
```

## Key caveats (read `09_GAPS.md`)

- Stages 1 (pretraining) & 2 (QA) have **no logs, metrics, or local models** — no
  loss curves, no perplexity, no EM/F1 anywhere in the repo.
- Only DPO (Stage 3) and web-search SFT (Stage 4) have full training data.
- EM/F1/perplexity must be computed manually (`test/test_qa_model.py`); table cells
  are marked **TBD**, never fabricated.
