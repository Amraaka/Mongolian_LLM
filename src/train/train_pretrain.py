from __future__ import annotations

import argparse

from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from src.train.callbacks import ThroughputCallback
from src.train.checkpointing import latest_checkpoint
from src.utils.io import ensure_dir, read_yaml
from src.utils.logging import get_logger
from src.utils.seed import set_seed

LOGGER = get_logger("train_pretrain")


def to_dtype(dtype_name: str):
    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_name.lower(), torch.bfloat16)


def main() -> None:
    parser = argparse.ArgumentParser(description="Continued pretraining for Qwen2B Mongolian.")
    parser.add_argument("--model-config", default="configs/model/qwen2b-continued-pretrain.yaml")
    parser.add_argument("--train-config", default="configs/train/train_base.yaml")
    parser.add_argument("--data-config", default="configs/data/data_mongolian.yaml")
    parser.add_argument("--deepspeed-config", default="configs/deepspeed/ds_zero2.json")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if available.")
    args = parser.parse_args()

    model_cfg = read_yaml(args.model_config)
    train_cfg = read_yaml(args.train_config)
    data_cfg = read_yaml(args.data_config)

    set_seed(int(train_cfg.get("seed", 42)))
    ensure_dir(train_cfg["output_dir"])
    ensure_dir(train_cfg["logging_dir"])

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model"], trust_remote_code=bool(model_cfg.get("trust_remote_code", True))
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        torch_dtype=to_dtype(str(model_cfg.get("torch_dtype", "bfloat16"))),
    )
    model.config.use_cache = False

    train_ds = load_from_disk(data_cfg["tokenized_train_path"])
    val_ds = load_from_disk(data_cfg["tokenized_val_path"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    targs = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        logging_dir=train_cfg["logging_dir"],
        report_to=train_cfg.get("report_to", "none"),
        max_steps=int(train_cfg["max_steps"]),
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        lr_scheduler_type=str(train_cfg["lr_scheduler_type"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        weight_decay=float(train_cfg["weight_decay"]),
        logging_steps=int(train_cfg["logging_steps"]),
        eval_steps=int(train_cfg["eval_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        save_total_limit=int(train_cfg["save_total_limit"]),
        evaluation_strategy=str(train_cfg["evaluation_strategy"]),
        save_strategy=str(train_cfg["save_strategy"]),
        bf16=bool(train_cfg.get("bf16", False)),
        fp16=bool(train_cfg.get("fp16", False)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 4)),
        remove_unused_columns=bool(train_cfg.get("remove_unused_columns", False)),
        deepspeed=args.deepspeed_config,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=[ThroughputCallback()],
    )

    checkpoint = latest_checkpoint(train_cfg["output_dir"]) if args.resume else None
    if checkpoint:
        LOGGER.info("Resuming from checkpoint: %s", checkpoint)
    else:
        LOGGER.info("Starting fresh training run.")

    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])
    LOGGER.info("Training complete. Artifacts saved to %s", train_cfg["output_dir"])


if __name__ == "__main__":
    main()
