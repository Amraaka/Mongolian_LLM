# 1. Project Overview

## 1.1 Base Model

| Item | Value |
|------|-------|
| Base model (main training) | **Qwen3.5-2B-Base** |
| HuggingFace ID | `Qwen/Qwen3.5-2B-Base` |
| Debug/demo model | `Qwen/Qwen3.5-0.8B` |
| Base model class | `Qwen3_5ForConditionalGeneration` (per adapter `auto_mapping`) |
| Parameter scale | ~2B (from model name; full base `config.json` is **not stored in repo**) |
| Adaptation method | QLoRA (4-bit NF4) throughout all 4 stages |

Source: `src/1_pretrain_text.py` (`model_choices`), `configs/saved_model_location.yaml`,
`models/*/adapter_config.json` (`base_model_name_or_path = Qwen/Qwen3.5-2B-Base`).

### Architecture details

The base model weights/`config.json` are **not present in the repository** (only QLoRA
adapters are saved). Therefore the following cannot be read from the repo and must be
taken from the upstream HF model card:

| Field | Value |
|-------|-------|
| Parameter count | not found in repo (≈2B by name) |
| Hidden size | not found in repo |
| Number of layers | not found in repo |
| Number of attention heads | not found in repo |
| Vocab size | **248,044** (base) / **248,077** including 33 added special tokens — from local `tokenizer.json` |
| Context length | **262,144** (`model_max_length` in `tokenizer_config.json`); training capped much lower (see below) |

Max sequence length actually used during training (from scripts):
Stage 1 = 1024, Stage 2 = 1024, Stage 3 = 1024 (prompt ≤256), Stage 4 = 2048.

## 1.2 Tokenizer

| Item | Value |
|------|-------|
| Tokenizer class | `TokenizersBackend` (fast tokenizers, `tokenizer.json`) |
| Vocabulary size | 248,044 |
| Total with added tokens | 248,077 (33 added special tokens, ChatML-style: `<|im_start|>`, `<|im_end|>`, `<tool_call>`, etc.) |
| `model_max_length` | 262,144 |
| Pad token | set to EOS token in every training script when `pad_token is None` |

A `chat_template.jinja` (ChatML format) is bundled with each saved model.

## 1.3 Repository Structure (top level)

| Path | Contents |
|------|----------|
| `configs/` | YAML configs: `data_locations.yaml`, `saved_model_location.yaml`, `data_split_partition.yaml` |
| `data/` | Datasets (HF Arrow on disk): `qa_data/`, `dpo_data/`, `instruction_data/`. `text_data/`, `vl_data/`, `iam_data/` referenced in configs but **not present** |
| `data_generation/` | Step-4 dataset builders: `generate_questions.py`, `build_search_dataset.py` (Tavily), `push_instruction_dataset.py`, `tavily_cache.jsonl`, `seed_questions.jsonl` |
| `dataset/` | DPO dataset prep: `dpo-dataset.py` + notebooks (`push_dpo_to_hub.ipynb`, `test_dpo_*.ipynb`) |
| `src/` | Training scripts (the pipeline) — see §1.4 |
| `models/` | Saved QLoRA adapters: `*_mongolian_dpo_0.1`, `*_mongolian_search_0.1` (Stage 1/2 model dirs **not present**) |
| `logs/` | Training logs: DPO, Step-4 search, DPO-dataset prep |
| `test/` | Evaluation scripts: `test_qa_model.py`, `test_qa_model_v0.3.ipynb`, `test_gradio.py` |
| `utils/` | `utils.py` — `setup_logging`, `CustomLogCallback`, `CustomDataLoader` (dataset formatting) |
| `unsloth_compiled_cache/` | Auto-generated Unsloth trainer caches (not project code) |
| `venv/` | Python virtual environment |
| root | `run_pipeline.sh`, `run_dpo.sh`, `requirements.txt`, `Readme.md`, `tavily_test.py`, `nccl_test.py` |

## 1.4 Training Pipeline Order

The pipeline is a 4-stage sequential chain; each stage loads the previous stage's
adapter (path resolved from `configs/saved_model_location.yaml`).

| Stage | Script | Method | Input model | Output model dir |
|-------|--------|--------|-------------|------------------|
| 1 | `src/1_pretrain_text.py` | Continued pretraining (CLM, `SFTTrainer`) | `Qwen/Qwen3.5-2B-Base` | `models/..._mongolian_text_0.1` |
| 2 | `src/2_finetune_qa.py` | QA SFT (`SFTTrainer`, ChatML) | Stage 1 adapter | `models/..._mongolian_qa_0.3` |
| 3 | `src/3_train_dpo.py` | DPO alignment (`DPOTrainer`, β=0.1) | Stage 2 adapter | `models/..._mongolian_dpo_0.1` |
| 4 | `src/4_instruction_fine_tuning.py` | Web-search function-calling SFT | Stage 3 adapter | `models/..._mongolian_search_0.1` |

`src/0.preprocess.py` and `src/5_vl_fine_tuning.py` exist but are **empty (0 lines)**.
Stage 4 was reframed from generic instruction tuning to a **single-tool web-search
(Tavily) function-calling** task that reuses `qa_data` answers as gold responses.

Orchestration: `run_pipeline.sh` invokes Stage 2 (Stage 1 line is commented out;
the active line trains QA at lr 5e-5 for 4 epochs, then 3 epochs at lr 3e-5).
`run_dpo.sh` invokes Stage 3.

## 1.5 README Content (repo `Readme.md`)

The repo `Readme.md` is the **course assignment brief** (ШУТИС / MUST, course
`F.CS332 Гүн Сургалт`). Key points:

- Goal: pick a 0.8B–4B open LLM, train it for Mongolian through 4 sequential stages
  (continued pretraining → QA fine-tuning → DPO → instruction fine-tuning), then
  compare pre/post performance.
- Suggested pretraining corpora: Wikipedia MN, Common Crawl MN, OSCAR Mongolian,
  news, OpenSubtitles, a course SharePoint dataset.
- Evaluation: **Perplexity & Loss** for pretraining; **EM / F1** for QA.
- Grading: Pretraining 20%, QA 20%, DPO 20%, Instruction 20%, Results & analysis 20%.
- Deliverables: code (GitHub), PDF report (MUST research-paper standard),
  model checkpoints, a 7–10 min presentation.

The model-card `README.md` files inside `models/*/` are auto-generated by TRL
(framework versions: TRL 0.24.0, Transformers 5.5.0, PyTorch 2.10.0,
Datasets 4.3.0, Tokenizers 0.22.2) and include the DPO citation (Rafailov et al., 2023).
