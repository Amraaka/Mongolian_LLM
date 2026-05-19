# 8. Summary Table

Final comparison table for the report. EM/F1 and perplexity were **not computed or
saved anywhere in the repo** (the eval code exists in `src/2_finetune_qa.py`,
`test/test_qa_model.py`, and `test/test_qa_model_v0.3.ipynb`, but no numeric results
are persisted). Those cells are marked **TBD** — run `INFERENCE_NEEDED.md` /
`test/test_qa_model.py` per checkpoint to fill them.

## 8.1 Primary metrics (report requirement)

| Model | Task | Metric | Result |
|-------|------|--------|--------|
| Base model | QA | EM | TBD |
| Base model | QA | F1 | TBD |
| Base model | LM | Perplexity | TBD |
| Pretrained model (Stage 1) | LM | Perplexity | TBD |
| Pretrained model (Stage 1) | QA | EM | TBD |
| Pretrained model (Stage 1) | QA | F1 | TBD |
| QA-tuned model (Stage 2) | QA | EM | TBD |
| QA-tuned model (Stage 2) | QA | F1 | TBD |
| DPO model (Stage 3) | QA | EM | TBD |
| DPO model (Stage 3) | QA | F1 | TBD |
| Instruction-tuned (Stage 4) | QA | EM | TBD |
| Instruction-tuned (Stage 4) | QA | F1 | TBD |

## 8.2 Metrics that ARE available (from logs / trainer_state)

| Model | Task | Metric | Result |
|-------|------|--------|--------|
| DPO model (Stage 3) | DPO | Best eval_loss | **0.035385** (step 1,400) |
| DPO model (Stage 3) | DPO | Final train loss | 0.003015 (step 1,500) |
| DPO model (Stage 3) | DPO | Mean train loss | 0.42118 |
| DPO model (Stage 3) | DPO | Eval reward accuracy | 0.9904 |
| DPO model (Stage 3) | DPO | Eval reward margin | 48.009 |
| DPO model (Stage 3) | DPO | Wall-clock training time | ≈ 5.61 h (20,206.76 s) |
| Instruction-tuned (Stage 4) | SFT | Best eval_loss | **0.947077** (step 57) |
| Instruction-tuned (Stage 4) | SFT | Final train loss | 0.96917 (step 50) |
| Instruction-tuned (Stage 4) | SFT | Mean train loss | 1.11560 |
| Instruction-tuned (Stage 4) | SFT | Wall-clock training time | ≈ 7.08 min (424.998 s) |
| Pretrained (Stage 1) | LM | Loss / Perplexity | not found in repo |
| QA-tuned (Stage 2) | SFT | Loss / eval_loss | not found in repo |

## 8.3 Dataset summary

| Stage | Dataset | Train | Val | Test | Avg tokens/sample |
|-------|---------|------:|----:|-----:|------------------:|
| 1 Pretrain | `text_data` | — | — | — | not in repo |
| 2 QA | `qa_data` | 98,000 | 1,025 | 10,000 | 363.7 (train) |
| 3 DPO | `dpo_data` | 7,496 | 625 | 1,874 | chosen 328 / rejected 497 |
| 4 Instruction | `instruction_data` | 900 | 50 | 50 | 982.7 (train) |

## 8.4 Model summary

| Stage | Output | Trainer | Key HP | Steps | Best eval_loss |
|-------|--------|---------|--------|------:|----------------|
| 1 | `..._text_0.1` | SFTTrainer | lr 2e-4, eff.bs 8, len 1024 | (not run/logged) | not in repo |
| 2 | `..._qa_0.3` | SFTTrainer | lr 5e-5→3e-5, eff.bs 8, len 1024 | (not logged) | not in repo |
| 3 | `..._dpo_0.1` | DPOTrainer | β 0.1, lr 5e-6, eff.bs 16 | 1,500 | 0.035385 |
| 4 | `..._search_0.1` | SFTTrainer | lr 2e-4, eff.bs 16, len 2048 | 57 | 0.947077 |

All stages: base `Qwen/Qwen3.5-2B-Base`, QLoRA 4-bit, LoRA r=16/α=16/dropout=0,
bf16, adamw_8bit, cosine LR, seed 3407.
