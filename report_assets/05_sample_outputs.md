# 5. Sample Model Outputs

**No dedicated inference-output directory exists** (`outputs/`, `predictions/`,
`generations/` are absent; no model-response `.json` files). The **only** saved
generations in the repo are inside `test/test_qa_model_v0.3.ipynb` — captured cell
outputs for the **Stage-2 QA model** (`Ganaa0614/Qwen3.5-2B-Base-qlora-mongolian-qa-ver_0.3`),
greedy decoding, `max_new_tokens=256`.

A full side-by-side comparison across **base / pretrained / QA-tuned / DPO /
instruction-tuned** is therefore **not possible from the repo** — only the QA-tuned
column has data. See `report_assets/INFERENCE_NEEDED.md` for the 5 prompts to run
through every checkpoint to complete this section.

## 5.1 Available generations — QA-tuned model (Stage 2, ver_0.3)

### Example 1 — Geometry (triangle area)
- **Q:** `A (2, 3), B (1, −1) ба C (−2, 4) оройтой ABC гурвалжны талбайг тооцоол.`
- **GOLD:** Uses the shoelace formula, substitutes, result = **0.5** sq. units.
- **QA-tuned PRED:** Reproduces the correct formula and substitution
  (`1/2·|2(-1-4)+1(4-3)+(-2)(3-(-1))|`), but the generation is **truncated** before
  the final numeric answer. Methodologically correct, incomplete.

### Example 2 — Explanation (big data vs ML)
- **Q:** `Том өгөгдөл болон машин сургалтын хоорондын хамаарлыг тайлбарла.`
- **GOLD:** ~156-word thorough explanation.
- **QA-tuned PRED:** ~68-word coherent, on-topic but shorter explanation — correct
  relationship described, less detailed than gold.

### Example 3 — Definition (circuit breaker)
- **Q:** `Хэлхээ таслагч гэж юу болохыг тайлбарла.`
- **GOLD:** Correct: a safety device that interrupts current on overload/short circuit.
- **QA-tuned PRED:** Partially incorrect — describes it as a power
  transmission/distribution device; **factual drift**, and truncated.

> Observation (relevant to the DPO stage rationale): the QA model is fluent in
> Mongolian but exhibits truncation, occasional factual drift, and shorter-than-gold
> answers — the error modes DPO (Stage 3) targets.

## 5.2 Base / Pretrained / DPO / Instruction-tuned outputs

**Not found in repo** — these checkpoints' generations were never saved. The
base and Stage-1/Stage-2 model directories are also absent locally
(only DPO and Stage-4 adapters are present). To produce the comparison table the
report requires, run the prompts in `INFERENCE_NEEDED.md` against each checkpoint.
