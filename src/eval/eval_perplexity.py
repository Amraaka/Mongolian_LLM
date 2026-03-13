from __future__ import annotations

import argparse
import math

from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from src.utils.io import read_yaml
from src.utils.logging import get_logger

LOGGER = get_logger("eval_perplexity")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate perplexity on validation set.")
    parser.add_argument("--model-config", default="configs/model/qwen2b-continued-pretrain.yaml")
    parser.add_argument("--train-config", default="configs/train/train_base.yaml")
    parser.add_argument("--data-config", default="configs/data/data_mongolian.yaml")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint or model path to evaluate.")
    args = parser.parse_args()

    model_cfg = read_yaml(args.model_config)
    train_cfg = read_yaml(args.train_config)
    data_cfg = read_yaml(args.data_config)
    model_name = args.checkpoint or train_cfg["output_dir"] or model_cfg["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=bool(model_cfg.get("trust_remote_code", True)))
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=bool(model_cfg.get("trust_remote_code", True)))

    val_ds = load_from_disk(data_cfg["tokenized_val_path"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    targs = TrainingArguments(output_dir="outputs/eval_tmp", per_device_eval_batch_size=1, report_to="none")
    trainer = Trainer(model=model, args=targs, eval_dataset=val_ds, tokenizer=tokenizer, data_collator=collator)
    metrics = trainer.evaluate()

    eval_loss = metrics.get("eval_loss")
    ppl = math.exp(eval_loss) if eval_loss is not None else float("nan")
    LOGGER.info("Eval metrics: %s", metrics)
    LOGGER.info("Perplexity: %.4f", ppl)


if __name__ == "__main__":
    main()
