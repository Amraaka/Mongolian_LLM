#=====================================================================================
# DPO DATASET PREPARATION
#=====================================================================================
import unsloth
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gc
import json
import yaml
import logging
import numpy as np
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import login
from unsloth import FastVisionModel
from sentence_transformers import SentenceTransformer
from sacrebleu import sentence_chrf
from tqdm import tqdm

from utils.utils import setup_logging, CustomDataLoader

load_dotenv()
login(token=os.getenv("HF_TOKEN"))


COSINE_WEIGHT = 0.7
CHRF_WEIGHT = 0.3
THRESHOLD = 0.85
GEN_BATCH_SIZE = 128
EMBED_BATCH_SIZE = 128


@torch.inference_mode()
def generate_rejected_batch(model, tokenizer, questions, max_new_tokens=1024):
    prompts = [
        f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        for q in questions
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    generated = outputs[:, prompt_len:]
    return [t.strip() for t in tokenizer.batch_decode(generated, skip_special_tokens=True)]


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _load_yaml(name):
        with open(os.path.join(current_dir, "configs", name)) as f:
            return yaml.safe_load(f)

    model_configs = _load_yaml("saved_model_location.yaml")
    data_configs = _load_yaml("data_locations.yaml")

    model_path = model_configs["step2"]["hub_id"]

    setup_logging(current_dir, "dpo_dataset")

    model, processor = FastVisionModel.from_pretrained(
        model_name=model_path,
        load_in_4bit=True,
    )
    FastVisionModel.for_inference(model)
    tokenizer = processor.tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"   # required for batched decoder-only generation

    dataloader = CustomDataLoader(current_dir=current_dir, tokenizer=tokenizer, dataset_name="qa_data")
    test_set = dataloader.load_data(only_test=True)

    logging.info(f"DPO dataset prep started | model={model_path} | test_size={len(test_set)} | gen_batch={GEN_BATCH_SIZE}")

    questions = [item["question"] for item in test_set]
    chosen = [item["answer"] for item in test_set]
    rejected = []

    for i in tqdm(range(0, len(questions), GEN_BATCH_SIZE)):
        batch_q = questions[i:i + GEN_BATCH_SIZE]
        rejected.extend(generate_rejected_batch(model, tokenizer, batch_q, max_new_tokens=1024))

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    embedder = SentenceTransformer(
        "Qwen/Qwen3-Embedding-0.6B",
        device="cuda",
        model_kwargs={"torch_dtype": torch.float16},
    )

    emb_c = embedder.encode(chosen, batch_size=EMBED_BATCH_SIZE, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=True)
    emb_r = embedder.encode(rejected, batch_size=EMBED_BATCH_SIZE, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=True)
    cos_scores = np.einsum("ij,ij->i", emb_c, emb_r)

    chrf_scores = np.array([
        sentence_chrf(r, [c]).score / 100.0
        for c, r in tqdm(list(zip(chosen, rejected)))
    ])

    score = COSINE_WEIGHT * cos_scores + CHRF_WEIGHT * chrf_scores

    kept = []
    metadata = []
    drop_reason_key = f"too_similar (score>={THRESHOLD})"
    drop_counts = {drop_reason_key: 0}

    for q, ch, rj, cos, cf, sc in zip(questions, chosen, rejected, cos_scores, chrf_scores, score):
        cos, cf, sc = float(cos), float(cf), float(sc)
        meta = {
            "prompt": q,
            "chosen": ch,
            "rejected": rj,
            "cosine": cos,
            "chrf": cf,
            "score": sc,
            "kept": False,
            "drop_reason": None,
        }

        if sc >= THRESHOLD:
            meta["drop_reason"] = drop_reason_key
            drop_counts[drop_reason_key] += 1
        else:
            meta["kept"] = True
            kept.append({"prompt": q, "chosen": ch, "rejected": rj})

        metadata.append(meta)

    save_dir = os.path.join(current_dir, data_configs["dpo_data"]["local_path"]["raw"])
    os.makedirs(save_dir, exist_ok=True)

    Dataset.from_list(kept).save_to_disk(save_dir)

    metadata_path = os.path.join(save_dir, "scoring_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logging.info(f"DPO dataset prep finished | total={len(questions)} | "
                 f"dropped({drop_reason_key})={drop_counts[drop_reason_key]} | "
                 f"kept={len(kept)} | saved_to={save_dir}")
