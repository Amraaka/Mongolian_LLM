# 6. Code Snippets for the Report

Curated excerpts demonstrating the methodology. Each block is paste-ready for the
LaTeX `listings` package, e.g.:

```latex
\begin{lstlisting}[language=Python, caption={Stage 1 — SFT pretraining setup}]
... snippet ...
\end{lstlisting}
```

---

### 6.1 Stage 1 — Continued pretraining: SFT setup (`src/1_pretrain_text.py:266`)

```python
sft_config = SFTConfig(
    output_dir=save_dir,
    dataset_text_field="text",
    max_length=MAX_SEQ_LEN,                 # 1024
    eos_token=tokenizer.eos_token,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum_step,
    warmup_steps=args.warmup_step,
    max_steps=args.steps,                   # 25000
    gradient_checkpointing=True,
    bf16=True, fp16=False,
    eval_strategy="steps", eval_steps=1000, save_steps=1000, logging_steps=100,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    seed=3407,
)
trainer = SFTTrainer(model=model, args=sft_config, processing_class=tokenizer,
                      train_dataset=train_set, data_collator=collator,
                      eval_dataset=validation_set, compute_metrics=compute_metrics)
```

### 6.2 Stage 1 — Perplexity metric (`src/1_pretrain_text.py:41`)

```python
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    if isinstance(logits, tuple): logits = logits[0]
    shift_logits = torch.tensor(logits)[..., :-1, :].contiguous()
    shift_labels = torch.tensor(labels)[..., 1:].contiguous()
    loss = CrossEntropyLoss(ignore_index=-100)(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    try:    perplexity = math.exp(loss.item())
    except OverflowError: perplexity = float("inf")
    return {"perplexity": perplexity, "eval_loss_recalculated": loss.item()}
```

### 6.3 Stage 2 — QA fine-tuning + EM/F1 (`src/2_finetune_qa.py:45`)

```python
def exact_match_metric(prediction, reference):
    return 1.0 if prediction.strip() == reference.strip() else 0.0

def f1_score_metric(prediction, reference):
    pred_tokens = normalize_text(prediction).split()
    ref_tokens  = normalize_text(reference).split()
    common = collections.Counter(pred_tokens) & collections.Counter(ref_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 1.0 if pred_tokens == ref_tokens else 0.0
    if num_same == 0: return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(ref_tokens)
    return (2 * precision * recall) / (precision + recall)
```

### 6.4 Stage 3 — DPO trainer config (`src/3_train_dpo.py:92`)

```python
dpo_config = DPOConfig(
    beta=0.1,
    max_length=1024, max_prompt_length=256,
    output_dir=save_dir,
    per_device_train_batch_size=args.batch_size,      # 1
    gradient_accumulation_steps=args.grad_accum_step,  # 16
    warmup_steps=args.warmup_step,                     # 50
    max_steps=args.steps,                              # 1500
    learning_rate=args.lr,                             # 5e-6
    gradient_checkpointing=True, bf16=True,
    eval_steps=200, save_steps=200, logging_steps=50,
    lr_scheduler_type="cosine", optim="adamw_8bit", seed=3407,
)
trainer = DPOTrainer(model=model, ref_model=None, args=dpo_config,
                      processing_class=tokenizer,
                      train_dataset=train_set, eval_dataset=validation_set)
```

### 6.5 Stage 4 — Web-search function-calling SFT (`src/4_instruction_fine_tuning.py:91`)

```python
sft_config = SFTConfig(
    output_dir=save_dir, dataset_text_field="text",
    max_length=2048, eos_token=tokenizer.eos_token,
    per_device_train_batch_size=args.batch_size,       # 2
    gradient_accumulation_steps=args.grad_accum_step,  # 8
    warmup_steps=args.warmup_step,                     # 10
    max_steps=args.steps if args.steps is not None else -1,
    num_train_epochs=args.epochs,                      # 1
    bf16=True, gradient_checkpointing=True,
    eval_steps=20, save_steps=20, logging_steps=10,
    lr_scheduler_type="cosine", optim="adamw_8bit",
    learning_rate=args.lr, seed=3407,                  # 2e-4
)
trainer = SFTTrainer(model=model, args=sft_config, processing_class=tokenizer,
                      train_dataset=train_set, eval_dataset=validation_set,
                      data_collator=collator)
```

### 6.6 Dataset preprocessing — ChatML formatters (`utils/utils.py:65`)

```python
def format_qa(self, batch):
    return {"text": [
        f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"
        for q, a in zip(batch["question"], batch["answer"])]}

def format_dpo(self, batch):
    return {
      "prompt":  [f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
                  for p in batch["prompt"]],
      "chosen":  [f"{c}<|im_end|>" for c in batch["chosen"]],
      "rejected":[f"{r}<|im_end|>" for r in batch["rejected"]]}
```

### 6.7 Stage 4 — Multi-turn tool-use template (`utils/utils.py:88`)

```python
def format_instruction(self, batch):
    formatted = []
    for tools, q, call, resp, ans in zip(
        batch["tools_json"], batch["question"],
        batch["tool_call_json"], batch["tool_response_json"], batch["answer"]):
        sys_msg = f"Та хэрэглэгчид туслахын тулд дараах хэрэгслүүдийг ашиглаж болно.\n<tools>\n{tools}\n</tools>"
        formatted.append(
            f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
            f"<|im_start|>user\n{q}<|im_end|>\n"
            f"<|im_start|>assistant\n<tool_call>{call}</tool_call><|im_end|>\n"
            f"<|im_start|>tool\n<tool_response>{resp}</tool_response><|im_end|>\n"
            f"<|im_start|>assistant\n{ans}<|im_end|>")
    return {"text": formatted}
```

### 6.8 DPO data construction — preference-pair filtering (`dataset/dpo-dataset.py:127`)

```python
# rejected = sampled generation from the QA-tuned model; chosen = gold answer
emb_c = embedder.encode(chosen, normalize_embeddings=True)     # Qwen3-Embedding-0.6B
emb_r = embedder.encode(rejected, normalize_embeddings=True)
cos_scores  = np.einsum("ij,ij->i", emb_c, emb_r)
chrf_scores = np.array([sentence_chrf(r, [c]).score / 100.0
                        for c, r in zip(chosen, rejected)])
score = 0.7 * cos_scores + 0.3 * chrf_scores                   # COSINE_WEIGHT / CHRF_WEIGHT
# drop a pair when chosen ≈ rejected (no learning signal)
if score >= 0.85:  drop()                                      # THRESHOLD
else:              keep({"prompt": q, "chosen": ch, "rejected": rj})
```

### 6.9 Tokenization / collation (used by Stages 1, 2, 4)

```python
# tokenizer obtained from Unsloth: tokenizer = processor.tokenizer  (Qwen3.5, ChatML)
if tokenizer.pad_token is None:
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
# SFTConfig(dataset_text_field="text", max_length=MAX_SEQ_LEN) tokenizes internally;
# preprocess_logits_for_metrics returns argmax pred_ids for EM/F1 decoding.
```
