"""
Read-only extraction for the academic report. Loads datasets/tokenizer,
computes stats, dumps samples + CSVs. Writes ONLY into report_assets/.
"""
import os, json, glob, statistics, random
from datasets import load_from_disk
from transformers import AutoTokenizer

random.seed(42)
ROOT = "/home/toru2/Amara/Mongolian_LLM"
RA = os.path.join(ROOT, "report_assets")
os.makedirs(os.path.join(RA, "samples"), exist_ok=True)
os.makedirs(os.path.join(RA, "data"), exist_ok=True)

TOK_DIR = os.path.join(ROOT, "models/Qwen3.5-2B-Base_qlora_mongolian_dpo_0.1")
tok = AutoTokenizer.from_pretrained(TOK_DIR)
print("tokenizer:", type(tok).__name__, "vocab_size=", tok.vocab_size,
      "len=", len(tok), "model_max_length=", tok.model_max_length)

EOS = tok.eos_token
TOK_CAP = 4000  # cap rows used for token stats on big splits


def dir_size_mb(path):
    total = 0
    for dp, _, fs in os.walk(path):
        for f in fs:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total / 1e6


def tok_stats(texts):
    n = len(texts)
    sample = texts if n <= TOK_CAP else random.sample(texts, TOK_CAP)
    lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in sample]
    return {
        "n_rows": n,
        "n_tokenized": len(lens),
        "is_estimate": n > TOK_CAP,
        "total_tokens_in_sample": sum(lens),
        "est_total_tokens": int(sum(lens) / len(lens) * n) if lens else 0,
        "avg": round(statistics.mean(lens), 1) if lens else 0,
        "min": min(lens) if lens else 0,
        "max": max(lens) if lens else 0,
        "median": statistics.median(lens) if lens else 0,
    }


def fmt_qa(q, a):
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"


report = {}

# ---------- QA dataset (mapped: train/validation/test) ----------
qa = {}
for split in ["train", "validation", "test"]:
    p = os.path.join(ROOT, "data/qa_data/qa_mapped", f"{split}_set")
    ds = load_from_disk(p)
    texts = ds["text"] if "text" in ds.column_names else [fmt_qa(x["question"], x["answer"]) for x in ds]
    qa[split] = {
        "path": os.path.relpath(p, ROOT),
        "rows": ds.num_rows,
        "columns": ds.column_names,
        "size_mb": round(dir_size_mb(p), 2),
        "tokens": tok_stats(list(texts)),
    }
report["qa_data"] = qa
qa_tr = load_from_disk(os.path.join(ROOT, "data/qa_data/qa_mapped/train_set"))
qa_samples = [{k: qa_tr[i][k] for k in qa_tr.column_names} for i in range(5)]
json.dump(qa_samples, open(f"{RA}/samples/qa_samples.json", "w"), ensure_ascii=False, indent=2)

# ---------- DPO dataset (train/validation/test) ----------
dpo = {}
for split in ["train", "validation", "test"]:
    p = os.path.join(ROOT, "data/dpo_data", f"{split}_set")
    ds = load_from_disk(p)
    pl = tok_stats(list(ds["prompt"]))
    cl = tok_stats(list(ds["chosen"]))
    rl = tok_stats(list(ds["rejected"]))
    dpo[split] = {
        "path": os.path.relpath(p, ROOT),
        "rows": ds.num_rows,
        "columns": ds.column_names,
        "size_mb": round(dir_size_mb(p), 2),
        "prompt_tokens": pl, "chosen_tokens": cl, "rejected_tokens": rl,
    }
report["dpo_data"] = dpo
dpo_tr = load_from_disk(os.path.join(ROOT, "data/dpo_data/train_set"))
dpo_samples = [{k: dpo_tr[i][k] for k in dpo_tr.column_names} for i in range(5)]
json.dump(dpo_samples, open(f"{RA}/samples/dpo_samples.json", "w"), ensure_ascii=False, indent=2)

# ---------- Instruction / search dataset (raw + mapped) ----------
instr = {}
praw = os.path.join(ROOT, "data/instruction_data/raw")
ds_raw = load_from_disk(praw)
instr["raw"] = {
    "path": os.path.relpath(praw, ROOT),
    "rows": ds_raw.num_rows,
    "columns": ds_raw.column_names,
    "size_mb": round(dir_size_mb(praw), 2),
}
for split in ["train", "validation", "test"]:
    p = os.path.join(ROOT, "data/instruction_data/mapped", f"{split}_set")
    ds = load_from_disk(p)
    instr[split] = {
        "path": os.path.relpath(p, ROOT),
        "rows": ds.num_rows,
        "columns": ds.column_names,
        "size_mb": round(dir_size_mb(p), 2),
        "tokens": tok_stats(list(ds["text"])),
    }
report["instruction_data"] = instr
instr_tr = load_from_disk(os.path.join(ROOT, "data/instruction_data/mapped/train_set"))
instr_samples = [{k: instr_tr[i][k] for k in instr_tr.column_names if k != "text"} for i in range(5)]
json.dump(instr_samples, open(f"{RA}/samples/instruction_samples.json", "w"), ensure_ascii=False, indent=2)

# token length list for instruction histogram
instr_all_lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in instr_tr["text"]]
json.dump(instr_all_lens, open(f"{RA}/data/instruction_token_lengths.json", "w"))

# pretrain samples: text_data is absent
json.dump({"status": "MISSING - data/text_data/ not present in repo"},
          open(f"{RA}/samples/pretrain_samples.json", "w"), ensure_ascii=False, indent=2)

json.dump(report, open(f"{RA}/data/_dataset_report.json", "w"), ensure_ascii=False, indent=2)
print("DATASET REPORT:\n", json.dumps(report, ensure_ascii=False, indent=1))


# ---------- Metrics: parse trainer_state.json log_history ----------
def load_logh(p):
    return json.load(open(p)).get("log_history", [])


dpo_lh = load_logh(os.path.join(ROOT, "models/Qwen3.5-2B-Base_qlora_mongolian_dpo_0.1/checkpoint-1500/trainer_state.json"))
search_lh = load_logh(os.path.join(ROOT, "models/Qwen3.5-2B-Base_qlora_mongolian_search_0.1/checkpoint-57/trainer_state.json"))

rows = [("stage", "step", "loss", "eval_loss", "learning_rate")]
for e in dpo_lh:
    if "loss" in e or "eval_loss" in e:
        rows.append(("dpo", e.get("step"), e.get("loss", ""), e.get("eval_loss", ""), e.get("learning_rate", "")))
for e in search_lh:
    if "loss" in e or "eval_loss" in e:
        rows.append(("instruction_search", e.get("step"), e.get("loss", ""), e.get("eval_loss", ""), e.get("learning_rate", "")))
with open(f"{RA}/data/loss_curves.csv", "w") as f:
    for r in rows:
        f.write(",".join(str(x) for x in r) + "\n")

json.dump({"dpo": dpo_lh, "search": search_lh},
          open(f"{RA}/data/_log_history.json", "w"), ensure_ascii=False, indent=1)

# eval_results.csv — EM/F1/perplexity not found anywhere in repo
with open(f"{RA}/data/eval_results.csv", "w") as f:
    f.write("model_version,metric_name,value\n")
    f.write("base,QA_EM,not found in repo\n")
    f.write("base,QA_F1,not found in repo\n")
    f.write("pretrained,QA_EM,not found in repo\n")
    f.write("pretrained,QA_F1,not found in repo\n")
    f.write("pretrained,perplexity,not found in repo\n")
    f.write("base,perplexity,not found in repo\n")
    f.write("qa_tuned,QA_EM,not found in repo\n")
    f.write("qa_tuned,QA_F1,not found in repo\n")
    f.write("dpo,QA_EM,not found in repo\n")
    f.write("dpo,QA_F1,not found in repo\n")
    f.write("instruction,QA_EM,not found in repo\n")
    f.write("instruction,QA_F1,not found in repo\n")
    f.write("dpo,eval_loss_best,0.035384997725486755\n")
    f.write("instruction_search,eval_loss_best,0.9470770359039307\n")

print("DONE")
