#=====================================================================================
# STEP4 WEB-SEARCH FUNCTION-CALLING FINE TUNING
#
# Trains the Step 3 (DPO) checkpoint to emit <tool_call> JSON for a single tool
# (web_search), then produce a Mongolian answer after a <tool_response> turn.
#=====================================================================================
import unsloth
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import DataCollatorForLanguageModeling
from transformers.trainer_utils import get_last_checkpoint
from huggingface_hub import login
import gc
from dotenv import load_dotenv
from trl import SFTTrainer, SFTConfig
from unsloth import FastVisionModel
import argparse
import logging
import yaml
from utils.utils import setup_logging, CustomDataLoader, CustomLogCallback

load_dotenv()
login(token=os.getenv("HF_TOKEN"))


def args_parse():
    parser = argparse.ArgumentParser(description="STEP 4 TRAINING HYPERPARAMETERS")
    parser.add_argument("--trainer", default="Bokhbat", required=True)
    parser.add_argument("--peft", choices=["qlora", "fft", "lora"], default="qlora", required=True)
    parser.add_argument("--batch_size", default=2, type=int, required=True)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--steps", default=None, type=int, help="(debug) override max_steps")
    parser.add_argument("--eval_batch", default=1, type=int, required=True)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--grad_accum_step", default=8, type=int)
    parser.add_argument("--warmup_step", default=10, type=int)
    parser.add_argument("--save_version", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = args_parse()
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.join(current_dir, "configs", "saved_model_location.yaml")

    with open(yaml_path, "r") as file:
        configs = yaml.safe_load(file)

    step3_model_path = configs["step3"]["local_path"]
    if not os.path.exists(step3_model_path):
        step3_model_path = configs["step3"]["hub_id"]

    if args.peft.lower() in ["lora", "qlora"]:
        is_4bit = (args.peft.lower() == "qlora")
        model, processor = FastVisionModel.from_pretrained(
            model_name=step3_model_path,
            load_in_4bit=is_4bit,
            use_gradient_checkpointing="unsloth",
        )
    elif args.peft.lower() in ["fft"]:
        model, processor = FastVisionModel.from_pretrained(
            model_name=step3_model_path,
            use_gradient_checkpointing="unsloth",
        )

    model_name = configs["trained_model"]

    save_dir = os.path.join(current_dir, "models", f"{model_name}_{args.peft}_mongolian_search_{args.save_version}")
    hub_model_id = f"{args.trainer}/{model_name}-{args.peft}-mongolian-search-ver_{args.save_version}"
    log_file = setup_logging(current_dir, f"{model_name}-{args.peft}-mongolian-search-ver_{args.save_version}")

    os.makedirs(save_dir, exist_ok=True)

    tokenizer = processor.tokenizer
    MAX_SEQ_LEN = 2048

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataloader = CustomDataLoader(current_dir=current_dir, tokenizer=tokenizer, dataset_name="instruction_data")
    train_set, validation_set = dataloader.load_data()

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    sft_config = SFTConfig(
        output_dir=save_dir,
        dataset_text_field="text",
        max_length=MAX_SEQ_LEN,
        eos_token=tokenizer.eos_token,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch,
        gradient_accumulation_steps=args.grad_accum_step,
        warmup_steps=args.warmup_step,
        max_steps=args.steps if args.steps is not None else -1,
        num_train_epochs=args.epochs,
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=20,
        save_steps=20,
        logging_steps=10,
        load_best_model_at_end=True,
        greater_is_better=False,
        save_total_limit=2,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        lr_scheduler_type="cosine",
        push_to_hub=True,
        hub_model_id=hub_model_id,
        report_to=["tensorboard"],
        optim="adamw_8bit",
        learning_rate=args.lr,
        seed=3407,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        processing_class=tokenizer,
        train_dataset=train_set,
        eval_dataset=validation_set,
        data_collator=collator,
        callbacks=[CustomLogCallback()],
    )

    logging.info(
        f"STEP 4 (search FC) training started | model={step3_model_path} | "
        f"train_size={len(train_set)} | val_size={len(validation_set)} | "
        + ", ".join(f"{k}={v}" for k, v in vars(args).items())
    )

    last_checkpoint = None
    if os.path.exists(save_dir) and os.listdir(save_dir):
        last_checkpoint = get_last_checkpoint(save_dir)

    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(save_dir)
    trainer.push_to_hub("Step 4 web-search function-calling training completed!")

    relative_save_dir = os.path.relpath(save_dir, current_dir)
    config_data = {
        "step4": {
            "local_path": relative_save_dir,
            "hub_id": hub_model_id,
        }
    }

    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as file:
            existing_data = yaml.safe_load(file) or {}
            existing_data.update(config_data)
            config_data = existing_data

    yaml_str = yaml.dump(config_data, default_flow_style=False, sort_keys=False)
    yaml_str = yaml_str.replace("\nstep1:", "\n\nstep1:")
    yaml_str = yaml_str.replace("\nfine tuned model:", "\n\nfine tuned model:")
    yaml_str = yaml_str.replace("\nstep2:", "\n\nstep2:")
    yaml_str = yaml_str.replace("\nstep3:", "\n\nstep3:")
    yaml_str = yaml_str.replace("\nstep4:", "\n\nstep4:")

    with open(yaml_path, "w") as file:
        file.write(yaml_str)

    logging.info(f"STEP 4 training finished | saved={save_dir} | hub={hub_model_id}")

    del model, trainer, processor
    gc.collect()
    torch.cuda.empty_cache()
