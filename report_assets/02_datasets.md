# 2. Dataset Information

All datasets are stored as **HuggingFace `datasets` Arrow files on disk** (loaded via
`load_from_disk`). Token counts use the local Qwen3.5 tokenizer
(`models/Qwen3.5-2B-Base_qlora_mongolian_dpo_0.1`, `add_special_tokens=False`) over the
exact training text. For large splits, token statistics are computed on a random sample
of 4,000 rows (seed 42) and the total is **estimated** (flagged below); row counts and
file sizes are always exact.

Configured train/val/test sizes are in `configs/data_split_partition.yaml`. Actual
on-disk split sizes differ where noted (splits were produced from a single shuffled
dataset with seed 42 via `CustomDataLoader.load_data`).

---

## 2.1 Stage 1 — Pretraining dataset (`text_data`)

**STATUS: NOT PRESENT IN REPOSITORY.**

- Configured source (`configs/data_locations.yaml`): local `data/text_data/mn_61785_test`
  → mapped `data/text_data/mn_mapped`; HF hub `Ganaa0614/mongolian-text-dataset`.
- Configured split (`configs/data_split_partition.yaml`): train 60,000 / validation 785 / test 1,000.
- Format expected by `utils.format_text`: a single `text` column; each row gets EOS appended.
- The `data/text_data/` directory does **not exist** on disk → no samples, no token
  counts, no size can be computed. `samples/pretrain_samples.json` records this gap.
- Preprocessing (`CustomDataLoader.format_text`): `text → text + <eos>`. No cleaning
  code beyond EOS appending is present (`src/0.preprocess.py` is empty).

---

## 2.2 Stage 2 — QA dataset (`qa_data`)

- **Source:** local `data/qa_data/qa_mapped/{train,validation,test}_set`; HF hub
  fallback `Ganaa0614/mongolian-qa-dataset`.
- **Format:** HF Arrow (mapped). Fields: `question`, `answer`, `text`.
- **Preprocessing** (`CustomDataLoader.format_qa`): each row →
  `<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>`
  stored in the `text` column (the field `SFTTrainer` trains on).

| Split | Rows | Size (MB) | Avg tok | Min | Max | Median | Total tokens |
|-------|------|-----------|---------|-----|-----|--------|--------------|
| train | 98,000 | 265.49 | 363.7 | 25 | 1,560 | 255 | ≈35,642,208 (est., 4k sample) |
| validation | 1,025 | 2.79 | 367.7 | 30 | 1,383 | 253 | 376,942 (exact) |
| test | 10,000 | 27.03 | 364.8 | 27 | 1,576 | 260 | ≈3,648,159 (est., 4k sample) |

Configured split (data_split_partition.yaml): train 98,000 / val 1,025 / test 10,000
— **matches on-disk**. (Note: only train+validation are used for training;
`test` is consumed by Stage-3 DPO data generation and by QA evaluation.)

**3 example samples** (`text` field, the trained sequence):

```text
<|im_start|>user
1-ээс 10 хүртэлх оноогоор авч үзвэл та ажилдаа хэр сэтгэл хангалуун байдаг вэ?<|im_end|>
<|im_start|>assistant
AI-ийн туслах хүний ​​хувьд надад ямар ч мэдрэмж, хувийн хүсэл сонирхол байдаггүй тул
ажлын сэтгэл ханамжийн тухай ойлголт надад хамаарахгүй. Миний зорилго бол програмчлагдсан
чадвараараа хэрэглэгчдэд хэрэгтэй хариулт, үйлчилгээг үзүүлэх явдал юм.<|im_end|>
```

```text
<|im_start|>user
 Өгөгдсөн дарааллын дараагийн элементийг таагаарай<|im_end|>
<|im_start|>assistant
3, 5, 7, 9, __ дарааллын дараагийн элемент нь 11 байх болно.<|im_end|>
```

```text
<|im_start|>user
Дараах үгийг дууд: 'conscientious'<|im_end|>
<|im_start|>assistant
 Энэ үгийг дууддаг: kahn-shee-EN-shuhs<|im_end|>
```

Full 5 examples: `report_assets/samples/qa_samples.json`.

---

## 2.3 Stage 3 — DPO dataset (`dpo_data`)

- **Source:** local `data/dpo_data/{train,validation,test}_set`; HF hub
  `Bokhbat/mongolian-dpo-dataset`.
- **Format:** HF Arrow. Fields: `prompt`, `chosen`, `rejected`.
- **Construction** (`dataset/dpo-dataset.py`): for each QA `test` row, `prompt` =
  question, `chosen` = gold answer, `rejected` = a generation sampled from the Stage-2
  QA model (`temperature=0.8, top_p=0.9, do_sample=True, max_new_tokens=512`). Pairs
  whose `chosen`/`rejected` are **too similar** are dropped, using a combined score
  `0.7·cosine(Qwen3-Embedding-0.6B) + 0.3·chrF`, threshold **0.85**.
  Log: total 10,000 → dropped 5 (too similar) → **kept 9,995**, then split.
- **Preprocessing at load** (`CustomDataLoader.format_dpo`): wraps to ChatML —
  prompt → `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`;
  chosen/rejected get `<|im_end|>` appended.

Token statistics computed per field on the **raw** (pre-wrap) text:

| Split | Rows | Size MB | prompt avg/min/max | chosen avg/min/max | rejected avg/min/max |
|-------|------|---------|--------------------|--------------------|----------------------|
| train | 7,496 | 201.65 | 38.8 / 16 / 460 | 328.0 / 2 / 1,538 | 496.9 / 7 / 517 |
| validation | 625 | 14.30 | 38.5 / 20 / 108 | 294.6 / 5 / 1,440 | 492.9 / 23 / 515 |
| test | 1,874 | 5.94 | 39.2 / 17 / 1,014 | 341.1 / 3 / 1,445 | 494.2 / 14 / 515 |

Total ≈ 9,995 pairs (matches prep log). Configured split
(data_split_partition.yaml: train 8,995 / val 500 / test 500) **does not match
on-disk** (actual 7,496 / 625 / 1,874); the on-disk Arrow splits are authoritative.
Training log confirms `train_size=7496 | val_size=625`.

> Note: `rejected` length clusters near ~507 tokens (the 512-token generation cap),
> while `chosen` (gold) is shorter on average — a notable length bias.

**3 example samples** (raw fields, truncated for readability):

```text
[prompt]   "Тэр сайхан шөнөдөө зөөлөн бүү ор" шүлгийг хураангуйлан бичээрэй.
[chosen]   "Тэр сайхан шөнөдөө зөөлөн бүү яв" гэдэг нь Дилан Томасын 1951 онд бичсэн
           шүлэг юм. ... (Дилан Томасын шүлгийн зөв тайлбар)
[rejected] "Тэр сайхан шөнөдөө зөөлөн бүү ороорой" шүлэг нь нэгэн эмэгтэйн хайрын
           мөчүүдэд зориулсан үг хэллэг юм. ... (буруу тайлбар, давталттай)
```

```text
[prompt]   Ажлынхаа ердийн нэг өдрийг дүрсэл.
[chosen]   Хиймэл оюун ухааны туслах хүний хувьд хэрэглэгч миний тусламж авах бүрд
           миний өдөр эхэлдэг. ... (тодорхой, холбогдох хариулт)
[rejected] Хиймэл оюун ухааны хувьд надад өдөр тутмын ажлын уулзалт ... 7:00-9:00 ...
           (гол асуултаас хазайсан хариулт)
```

```text
[prompt]   Долоо хоногийн үнэ цэнэтэй хоолны жагсаалт гарга.
[chosen]   Хүнсний жагсаалт: Спагетти гоймон / Маринара соус / ... (цэгцтэй жагсаалт)
[rejected] Долоо хоногийн үнэ цэнэтэй хоолны жагсаалтыг энд оруулав: 1. ... 6. ...
           (бүтэц муутай, давталттай)
```

Full 5 examples (with full prompt/chosen/rejected): `report_assets/samples/dpo_samples.json`.

---

## 2.4 Stage 4 — Instruction / Web-Search dataset (`instruction_data`)

- **Source:** local `data/instruction_data/raw` (single Dataset, 1,000 rows) →
  mapped `data/instruction_data/mapped/{train,validation,test}_set`. HF hub id empty.
- **Format:** HF Arrow. Raw fields: `tools_json`, `question`, `tool_call_json`,
  `tool_response_json`, `answer`. Mapped adds a `text` field.
- **Construction** (`data_generation/`): `generate_questions.py` builds ~1,000
  curated Mongolian factual questions (people / places / culture / science / events ×
  templates → `seed_questions.jsonl`). `build_search_dataset.py` calls **Tavily** web
  search per question (cached to `tavily_cache.jsonl`), keeps top-3 results
  (≤600 chars each), and packs the 5 fields. Answers come from QA gold or, in
  questions-file mode, are derived from Tavily's synthesized answer / top snippet.
- **Preprocessing at load** (`CustomDataLoader.format_instruction`): builds a full
  multi-turn ChatML conversation — system (tool spec) → user (question) →
  assistant (`<tool_call>…</tool_call>`) → tool (`<tool_response>…</tool_response>`) →
  assistant (final Mongolian answer) — stored in `text`.

| Split | Rows | Size (MB) | Avg tok | Min | Max | Median | Total tokens |
|-------|------|-----------|---------|-----|-----|--------|--------------|
| raw   | 1,000 | 13.54 | — | — | — | — | (unmapped) |
| train | 900 | 9.85 | 982.7 | 440 | 2,311 | 960 | 884,412 (exact) |
| validation | 50 | 0.53 | 920.2 | 625 | 1,443 | 885 | 46,010 (exact) |
| test | 50 | 0.31 | 1,038.8 | 677 | 2,249 | 995 | 51,941 (exact) |

Configured split (data_split_partition.yaml: train 900 / val 50 / test 100) — train/val
match; on-disk `test` = 50 (not 100). Training log confirms `train_size=900 | val_size=50`.

**3 example samples** (raw fields, `tool_response_json` truncated):

```text
[question]   Сэргэн мандалт хэзээ болсон бэ?
[tool_call]  {"name": "web_search", "arguments": {"query": "Сэргэн мандалт хэзээ болсон бэ?"}}
[tool_resp]  [{"title": "Сэргэн мандалтын үе - History ...", "url": "http://...",
              "content": "Сэргэн мандалт нь 13-р зууны сүүл үеийн Флоренцоос ..."}]
[answer]     Сэргэн мандалт нь XIV зууны эхээс XVI зууны сүүл хүртэл Итали дахь соёл,
             урлагийн хөгжлийн үе юм. ...
```

```text
[question]   Увс аймаг хаана байрладаг вэ?
[tool_call]  {"name": "web_search", "arguments": {"query": "Увс аймаг хаана байрладаг вэ?"}}
[tool_resp]  [{"title": "Увс аймаг :: touristinfocenter.mn", "url": "https://...",
              "content": "Аймгийн төв Улаангом хот ... Нутаг 69.6 мянган км²."}]
[answer]     Увс аймаг Монгол улсын баруун хязгаарт, Улаангом хотоор нийслэл,
             69.6 мянган км² нутаг дэвсгэртэй байрладаг.
```

```text
[tools_json] [{"name": "web_search", "description": "Интернетээс хайлт хийж, шинэлэг
             мэдээлэл авах...", "parameters": {"type":"object","properties":
             {"query":{"type":"string","description":"Хайх асуулт (Монгол эсвэл
             Англи хэлээр)"}},"required":["query"]}}]
```
(`tools_json` is identical for every row — the single `web_search` tool spec.)

Full 5 examples (instruction/input/output style): `report_assets/samples/instruction_samples.json`
(saved with keys `question`, `tool_call_json`, `tool_response_json`, `answer`, `tools_json`).

---

## 2.5 Raw sample files

| File | Contents |
|------|----------|
| `samples/pretrain_samples.json` | `{"status": "MISSING - data/text_data/ not present in repo"}` |
| `samples/qa_samples.json` | 5 QA rows (question / answer / text) |
| `samples/dpo_samples.json` | 5 DPO rows (prompt / chosen / rejected) |
| `samples/instruction_samples.json` | 5 web-search rows (question / tool_call_json / tool_response_json / answer / tools_json) |
