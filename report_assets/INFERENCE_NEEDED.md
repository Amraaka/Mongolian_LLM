# Inference Needed — Manual Generation Required

No model generations are stored for the **base, pretrained, DPO, or
instruction-tuned** checkpoints (only QA-tuned has captured notebook outputs, see
`05_sample_outputs.md`). To complete the report's side-by-side comparison, run the
5 prompts below through **each** checkpoint and record the responses.

## Checkpoints to evaluate

| Version | Source |
|---------|--------|
| Base model | `Qwen/Qwen3.5-2B-Base` (HF) |
| Pretrained (Stage 1) | `Ganaa0614/Qwen3.5-2B-Base-qlora-mongolian-text-ver_0.1` (HF; local dir absent) |
| QA-tuned (Stage 2) | `Ganaa0614/Qwen3.5-2B-Base-qlora-mongolian-qa-ver_0.3` (HF; local dir absent) |
| DPO (Stage 3) | `models/Qwen3.5-2B-Base_qlora_mongolian_dpo_0.1` (local ✓) |
| Instruction/search (Stage 4) | `models/Qwen3.5-2B-Base_qlora_mongolian_search_0.1` (local ✓) |

## 5 suggested test prompts (Mongolian)

1. **Factual / knowledge** —
   `Монгол улсын нийслэл хот хаана байрладаг вэ, түүний түүхийг товч өгүүлнэ үү?`
2. **Reasoning / math** —
   `Нэг дэлгүүрт 12 алим байсан. 5 алим зарж, дараа нь 8 алим нэмж авав. Одоо хэдэн алим байна вэ? Алхам алхмаар бод.`
3. **Open-ended generation** —
   `Намрын улирлын тухай богино шүлэг бичээрэй.`
4. **Instruction following** —
   `Дараах өгүүлбэрийг англи хэл рүү орчуул: "Би өнөөдөр сургуульдаа явсан."`
5. **Web-search function-calling** (specifically tests Stage 4) —
   `2024 оны зуны олимп хаана болсон бэ?` — expect the Stage-4 model to emit a
   `<tool_call>{"name":"web_search",...}</tool_call>` then a Mongolian answer after a
   `<tool_response>`.

## Suggested runner (read-only; do not commit into training dirs)

```bash
# For each model id/path:
python - <<'PY'
from unsloth import FastVisionModel
import torch
mp = "models/Qwen3.5-2B-Base_qlora_mongolian_dpo_0.1"   # swap per checkpoint
model, proc = FastVisionModel.from_pretrained(model_name=mp, load_in_4bit=True)
FastVisionModel.for_inference(model)
tok = proc.tokenizer
prompts = [...]  # the 5 prompts above
for q in prompts:
    p = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    ids = tok(p, return_tensors="pt").to("cuda")
    out = model.generate(**ids, max_new_tokens=256, do_sample=False,
                          pad_token_id=tok.eos_token_id)
    print(q, "→", tok.decode(out[0], skip_special_tokens=True).split("assistant\n")[-1])
PY
```

Save the collected responses to `report_assets/samples/inference_outputs.json`
(suggested schema: `{prompt: {base, pretrained, qa, dpo, instruction}}`) and then
populate the comparison tables in `05_sample_outputs.md` and the EM/F1 rows in
`08_summary_table.md` / `data/eval_results.csv`.
