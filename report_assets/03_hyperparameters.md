# 3. Hyperparameters per Stage

Sources: `src/{1,2,3,4}_*.py` (`SFTConfig`/`DPOConfig`), `run_pipeline.sh`,
`run_dpo.sh`, the training logs (actual CLI args), and `checkpoint-*/trainer_state.json`.
`training_args.bin` could not be deserialized (requires the Unsloth package), so
optimizer sub-parameters not set explicitly in code are reported as TRL/HF defaults.

**Common to all stages:** QLoRA (4-bit) via Unsloth `FastVisionModel`; LoRA
**r=16, α=16, dropout=0, bias=none**, `task_type=CAUSAL_LM`, `init_lora_weights=true`,
no RSLoRA/DoRA; target modules = attention + MLP projections
(`q/k/v/o_proj`, `gate/up/down_proj`) via regex (see `models/*/adapter_config.json`);
precision **bf16=True, fp16=False**; **gradient_checkpointing="unsloth"** (on);
optimizer **`adamw_8bit`** (bitsandbytes 8-bit AdamW); LR scheduler **cosine**;
**seed=3407** (trainer); LoRA `random_state=42`; dataset split seed `42`;
`report_to=tensorboard`; `load_best_model_at_end=True`, `save_total_limit=2`.
Optimizer β1/β2/ε and weight_decay are **not set in code** → TRL/HF defaults
(β1=0.9, β2=0.999, ε=1e-8, weight_decay=0.0, max_grad_norm=1.0).

---

## 3.1 Stage 1 — Continued Pretraining

⚠️ Stage 1 was **not executed/logged** in this repo (the `run_pipeline.sh` line is
commented out; no log or `trainer_state.json`). Values below are the *intended*
configuration from `src/1_pretrain_text.py` + the commented `run_pipeline.sh` command.

| Hyperparameter | Value |
|----------------|-------|
| Trainer | `trl.SFTTrainer` / `SFTConfig` (CLM) |
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Warmup steps | 700 |
| Per-device train batch size | 2 |
| Per-device eval batch size | 1 |
| Gradient accumulation | 4 |
| **Effective batch size** | **8** |
| Max steps | 25,000 |
| Epochs | (`num_train_epoch` — see note) |
| Max sequence length | 1,024 |
| Precision | bf16 |
| Gradient checkpointing | on (unsloth) |
| Optimizer | adamw_8bit |
| eval/save steps | 1,000 / 1,000 |
| logging steps | 100 |
| Seed | 3407 |
| LoRA | r=16, α=16, dropout=0 |

> **Bug note:** `SFTConfig` is passed `num_train_epoch=args.epochs` (misspelled;
> correct key is `num_train_epochs`). The argument is silently ignored, so training
> would have been governed solely by `max_steps=25000`. `run_pipeline.sh` comment
> states "~4 epochs Trained ~51 hours".

---

## 3.2 Stage 2 — QA Fine-tuning

⚠️ No Stage-2 log or `trainer_state.json` is stored (model dir
`models/..._qa_0.3` absent). Values from `src/2_finetune_qa.py` + the active
`run_pipeline.sh` commands.

| Hyperparameter | Run A | Run B |
|----------------|-------|-------|
| Trainer | `SFTTrainer`/`SFTConfig` | same |
| Learning rate | 5e-5 | 3e-5 |
| LR scheduler | cosine | cosine |
| Warmup steps | 150 | 150 |
| Per-device train batch | 2 | 2 |
| Per-device eval batch | 1 | 1 |
| Gradient accumulation | 4 | 4 |
| **Effective batch size** | **8** | **8** |
| Epochs | 4 | 3 |
| Max steps | (not set → epoch-driven) | (epoch-driven) |
| Max sequence length | 1,024 | 1,024 |
| Precision | bf16 | bf16 |
| eval/save steps | 250 / 250 | 250 / 250 |
| logging steps | 100 | 100 |
| Optimizer | adamw_8bit | adamw_8bit |
| Seed | 3407 | 3407 |
| LoRA | r=16, α=16, dropout=0 | same |
| save_version | 0.3 | 0.3 |

`SFTConfig` here correctly uses `num_train_epochs`. `compute_metrics` computes
**exact_match** and **f1** during eval (token-decoded), but no eval metrics are
persisted in the repo.

---

## 3.3 Stage 3 — DPO

Source: `src/3_train_dpo.py` (`DPOConfig`) + `run_dpo.sh` + DPO log +
`checkpoint-1500/trainer_state.json`.

| Hyperparameter | Value |
|----------------|-------|
| Trainer | `trl.DPOTrainer` / `DPOConfig` (`PatchDPOTrainer()`) |
| **DPO β (beta)** | **0.1** |
| **Reference model** | `ref_model=None` → implicit reference = base model with LoRA adapter disabled (PEFT/Unsloth DPO) |
| Loss type | TRL default (sigmoid) |
| Learning rate | 5e-6 |
| LR scheduler | cosine |
| Warmup steps | 50 |
| Per-device train batch | 1 |
| Per-device eval batch | 1 |
| Gradient accumulation | 16 |
| **Effective batch size** | **16** |
| Max steps | 1,500 |
| `num_train_epochs` (config) | 4 (but `max_steps=1500` governs → ran to step 1500, epoch ≈ 3.20) |
| max_length / max_prompt_length | 1,024 / 256 |
| Precision | bf16 |
| Gradient checkpointing | on |
| eval/save steps | 200 / 200 |
| logging steps | 50 |
| Optimizer | adamw_8bit |
| Seed | 3407 |
| LoRA | r=16, α=16, dropout=0 |
| Train data | 7,496 train / 625 val (from log) |
| Best checkpoint | step 1,400 (best eval_loss = 0.03538) |

---

## 3.4 Stage 4 — Instruction (Web-Search Function-Calling)

Source: `src/4_instruction_fine_tuning.py` (`SFTConfig`) + search log +
`checkpoint-57/trainer_state.json`.

| Hyperparameter | Value |
|----------------|-------|
| Trainer | `SFTTrainer` / `SFTConfig` (SFT) |
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Warmup steps | 10 |
| Per-device train batch | 2 |
| Per-device eval batch | 1 |
| Gradient accumulation | 8 |
| **Effective batch size** | **16** |
| Epochs | 1 |
| Max steps | -1 (epoch-driven → 57 steps total) |
| Max sequence length | 2,048 |
| Precision | bf16 |
| Gradient checkpointing | on |
| eval/save steps | 20 / 20 |
| logging steps | 10 |
| Optimizer | adamw_8bit |
| Seed | 3407 |
| LoRA | r=16, α=16, dropout=0 |
| Train data | 900 train / 50 val (from log) |
| Best checkpoint | step 57 (best eval_loss = 0.9471) |

> The search log shows a **second run** that produced
> `train_runtime=0.0085s, train_loss=0.0` — i.e., it immediately resumed from the
> existing checkpoint-57 and did no real training (an accidental re-invoke). The
> first run (step 10→57) is the real Stage-4 training.
