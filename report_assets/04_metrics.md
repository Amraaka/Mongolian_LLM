# 4. Training Metrics

Extracted from `models/*/checkpoint-*/trainer_state.json` (`log_history`) and the
training logs in `logs/`. Raw data is in `report_assets/data/loss_curves.csv`,
`report_assets/data/eval_results.csv`, and `report_assets/data/_log_history.json`.

## 4.1 Stage 1 — Pretraining

**Not found in repo.** No `trainer_state.json`, no training log, no
`models/..._text_*` directory. Initial/final loss, loss curve, perplexity
**before/after pretraining**: *not found in repo*.

> `src/1_pretrain_text.py` defines `compute_metrics` returning `perplexity` and
> `eval_loss_recalculated`, but no evaluation output was saved anywhere.

## 4.2 Stage 2 — QA Fine-tuning

**Not found in repo.** No `trainer_state.json`, no QA training log, no
`models/..._qa_0.3` directory. Loss curve, eval loss, **EM/F1**: *not found in repo*.

> `src/2_finetune_qa.py` and `test/test_qa_model.py` /
> `test/test_qa_model_v0.3.ipynb` implement EM & F1, but **no computed score is
> stored** (notebook score cell has no saved output). See `09_GAPS.md`.

## 4.3 Stage 3 — DPO

| Metric | Value |
|--------|-------|
| Total training steps | 1,500 (max_steps; best at step 1,400) |
| Initial train loss (step 50) | 9.2661 |
| Final train loss (step 1,500) | 0.003015 |
| Mean train loss (`train_loss`) | 0.42118 |
| Initial learning rate (step 50) | 4.90e-06 |
| Final learning rate (step 1,500) | 5.87e-12 |
| First eval_loss (step 200) | 0.133163 |
| Best/last eval_loss | **0.035385** (step 1,400, best) / 0.035677 (step 1,500) |
| Final eval rewards/accuracies | 0.9904 |
| Final eval rewards/margins | 48.009 |
| Total training time | **20,206.76 s ≈ 5.61 h** (`train_runtime`) |
| Throughput | 1.188 samples/s, 0.074 steps/s |

### Stage 3 loss per logged step (train, logging_steps=50)

| Step | Loss | LR | rewards/acc | rewards/margins |
|-----:|------|----|-------------|-----------------|
| 50 | 9.26609 | 4.90e-06 | 0.4525 | -1.257 |
| 100 | 2.58039 | 4.986e-06 | 0.7825 | 19.589 |
| 150 | 0.22244 | 4.943e-06 | 0.9688 | 38.242 |
| 200 | 0.16082 | 4.871e-06 | 0.9775 | 45.826 |
| 250 | 0.09396 | 4.771e-06 | 0.9775 | 42.163 |
| 300 | 0.01264 | 4.645e-06 | 0.9925 | 44.497 |
| 350 | 0.07935 | 4.494e-06 | 0.9838 | 45.397 |
| 400 | 0.03227 | 4.319e-06 | 0.9900 | 46.654 |
| 450 | 0.03926 | 4.123e-06 | 0.9888 | 46.591 |
| 500 | 0.02675 | 3.907e-06 | 0.9937 | 46.056 |
| 550 | 0.01022 | 3.676e-06 | 0.9975 | 48.142 |
| 600 | 0.01306 | 3.430e-06 | 0.9950 | 45.245 |
| 650 | 0.00563 | 3.174e-06 | 0.9962 | 44.612 |
| 700 | 0.01724 | 2.910e-06 | 0.9962 | 46.803 |
| 750 | 0.01998 | 2.641e-06 | 0.9962 | 48.246 |
| 800 | 0.01595 | 2.370e-06 | 0.9962 | 49.020 |
| 850 | 0.00081 | 2.101e-06 | 1.0000 | 49.678 |
| 900 | 0.00131 | 1.836e-06 | 1.0000 | 48.903 |
| 950 | 0.00122 | 1.580e-06 | 0.9987 | 48.879 |
| 1000 | 0.00123 | 1.334e-06 | 1.0000 | 46.340 |
| 1050 | 0.00425 | 1.102e-06 | 0.9987 | 47.960 |
| 1100 | 0.00646 | 8.857e-07 | 0.9987 | 49.608 |
| 1150 | 0.00021 | 6.887e-07 | 1.0000 | 46.578 |
| 1200 | 0.00913 | 5.130e-07 | 0.9987 | 50.328 |
| 1250 | 0.00043 | 3.607e-07 | 1.0000 | 48.878 |
| 1300 | 0.00225 | 2.333e-07 | 0.9987 | 47.636 |
| 1350 | 0.00017 | 1.326e-07 | 1.0000 | 48.721 |
| 1400 | 0.00127 | 5.962e-08 | 1.0000 | 48.434 |
| 1450 | 0.00768 | 1.525e-08 | 0.9987 | 49.347 |
| 1500 | 0.00302 | 5.87e-12 | 0.9987 | 49.552 |

### Stage 3 eval loss (eval_steps=200)

| Step | eval_loss | eval rewards/acc | eval rewards/margins |
|-----:|-----------|------------------|----------------------|
| 200 | 0.133163 | 0.9808 | 42.686 |
| 400 | 0.049669 | 0.9888 | 45.314 |
| 600 | 0.045689 | 0.9904 | 45.637 |
| 800 | 0.038908 | 0.9904 | 47.175 |
| 1000 | 0.035727 | 0.9904 | 47.508 |
| 1200 | 0.035410 | 0.9904 | 47.496 |
| **1400** | **0.035385** | 0.9904 | 47.969 |
| 1500 | 0.035677 | 0.9904 | 48.009 |

## 4.4 Stage 4 — Instruction (Web-Search Function-Calling)

| Metric | Value |
|--------|-------|
| Total training steps | 57 (1 epoch) |
| Initial train loss (step 10) | 1.55560 |
| Final train loss (step 50) | 0.96917 |
| Mean train loss (`train_loss`) | 1.11560 |
| Initial learning rate (step 10) | 1.80e-04 |
| Final logged learning rate (step 50) | 1.396e-05 |
| First eval_loss (step 20) | 1.023124 |
| Best/last eval_loss | **0.947077** (step 57, best) |
| Total training time | **424.998 s ≈ 7.08 min** (`train_runtime`) |
| Throughput | 2.118 samples/s, 0.134 steps/s |
| total_flos | 1.0518e+16 |

### Stage 4 loss per logged step (logging_steps=10)

| Step | Train loss | eval_loss | LR |
|-----:|-----------|-----------|----|
| 10 | 1.55560 | — | 1.800e-04 |
| 20 | 1.09854 | 1.023124 | 1.824e-04 |
| 30 | 1.03977 | — | 1.296e-04 |
| 40 | 0.99595 | 0.958193 | 6.406e-05 |
| 50 | 0.96917 | — | 1.396e-05 |
| 57 | — | **0.947077** | — |

> A second log entry shows an accidental re-run resuming from checkpoint-57
> (`train_runtime=0.0085s`, `train_loss=0.0`) — ignored; not real training.

## 4.5 Cross-stage metrics

| Metric | Status |
|--------|--------|
| Perplexity (base vs pretrained) | **not found in repo** |
| EM / F1 (base / pretrained / QA / DPO / instruction) | **not found in repo** |
| eval_loss (DPO best) | 0.035385 |
| eval_loss (Stage-4 best) | 0.947077 |

## 4.6 Raw data files

- `report_assets/data/loss_curves.csv` — columns `stage,step,loss,eval_loss,learning_rate`
  (stages `dpo`, `instruction_search`; pretraining & QA rows absent — no data).
- `report_assets/data/eval_results.csv` — columns `model_version,metric_name,value`
  (EM/F1/perplexity rows = "not found in repo"; two `eval_loss_best` rows populated).
- `report_assets/data/_log_history.json` — full raw `log_history` for DPO & Stage 4.
