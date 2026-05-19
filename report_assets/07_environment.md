# 7. Hardware & Environment

## 7.1 Python

| Item | Value |
|------|-------|
| Python version | **3.12.3** (`venv/pyvenv.cfg`; system `/usr/bin/python3.12`) |
| Virtual env | `venv/` (system site-packages disabled) |

## 7.2 Key library versions (`requirements.txt`)

| Library | Version |
|---------|---------|
| torch | **2.10.0** |
| transformers | **5.5.0** |
| trl | **0.24.0** |
| peft | **0.18.1** |
| accelerate | **1.13.0** |
| datasets | **4.3.0** |
| tokenizers | 0.22.2 |
| unsloth | 2026.4.4 |
| unsloth_zoo | 2026.4.3 |
| bitsandbytes | 0.49.2 |
| xformers | 0.0.35 |
| triton | 3.6.0 |
| sentence-transformers | 3.3.1 (DPO data prep) |
| sacrebleu | 2.4.3 (chrF in DPO data prep) |
| numpy | 2.4.3 |
| sentencepiece | 0.2.1 |
| tensorboard | 2.20.0 |

Model-card READMEs report the runtime stack: TRL 0.24.0, Transformers 5.5.0,
PyTorch 2.10.0, Datasets 4.3.0, Tokenizers 0.22.2.

> `matplotlib` is **not** in `requirements.txt`; it was installed into the venv
> solely to render the report figures (it is not used by any training code).

## 7.3 CUDA

| Item | Value |
|------|-------|
| CUDA toolkit (pip wheels) | **CUDA 12.8** (`nvidia-cuda-runtime-cu12==12.8.90`, `nvidia-cublas-cu12==12.8.4.1`, `nvidia-cudnn-cu12==9.10.2.21`, `nvidia-nccl-cu12==2.27.5`) |
| torch build | 2.10.0; `torchaudio==2.4.0+cu121` (cu121 tag) |
| GPU driver (current machine) | 595.58.03 |

## 7.4 GPU

GPU info is **not recorded in any training log**. Detected on the *current* machine
(may differ from the original training workstation):

| Item | Value |
|------|-------|
| GPU | **NVIDIA GeForce RTX 5080** |
| VRAM | 16,303 MiB (~16 GB) |
| Driver | 595.58.03 |

Context from `Readme.md` / `run_pipeline.sh`: training targeted a "317-room GPU
workstation / server"; a comment notes Stage-1 pretraining ≈ "4 epochs, ~51 hours"
and Stage-2 QA ≈ "6 epochs, ~1.3 hours". The DPO log shows ~5.61 h wall time and
Stage-4 ~7 min. The low effective batch sizes (8–16) and QLoRA 4-bit are consistent
with a single ~16 GB consumer GPU.

## 7.5 Environment variables

Set in code / shell scripts (names only; values come from `.env`, not committed):

| Variable | Where | Purpose |
|----------|-------|---------|
| `HF_TOKEN` | `.env`, used in all `src/*` (`huggingface_hub.login`) | HF auth / push_to_hub |
| `HUGGINGFACE_TOKEN` | referenced as alt name | HF auth |
| `TAVILY_API_KEY` | `.env`, `data_generation/build_search_dataset.py` | Tavily web search (Stage-4 data) |
| `WANDB_DISABLED="true"` | `src/1_pretrain_text.py` | disable Weights & Biases |
| `PYTORCH_ALLOC_CONF=expandable_segments:True` | `run_dpo.sh` | CUDA allocator (fragmentation) |
| `UNSLOTH_RETURN_HIDDEN_STATES` | Unsloth internal | hidden-state return toggle |
| `RANK` / `LOCAL_RANK` / `WORLD_SIZE` | distributed launch | (single-GPU runs observed) |

Also set in scripts: `torch.backends.cuda.matmul.allow_tf32 = True` and
`torch.backends.cudnn.allow_tf32 = True` (TF32 enabled). W&B disabled;
logging via TensorBoard (`report_to=["tensorboard"]`, runs in `models/*/runs/`).
