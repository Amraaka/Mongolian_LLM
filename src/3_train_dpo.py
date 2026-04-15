#=====================================================================================
# STEP3 DPO TRAINING
#=====================================================================================

import torch 
from transformers.trainer_utils import get_last_checkpoint
from huggingface_hub import login 
import gc 
import os 
from dotenv import load_dotenv
from trl import DPOTrainer, DPOConfig
from unsloth import FastVisionModel, PatchDPOTrainer
import argparse
import logging 
import yaml 
from Mongolian_LLM.utils.utils import setup_logging, CustomDataLoader, CustomLogCallback

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

PatchDPOTrainer() 

def args_parse():
    parser = argparse.ArgumentParser(description="TRAINING HYPERPARAMETERS")
    parser.add_argument("--trainer", default="Ganaa0614", choices=["Ganaa0614"], required=True)
    parser.add_argument("--peft", choices=["qlora", "fft", "lora"], default="qlora", required=True)
    parser.add_argument("--batch_size", default=4, type=int, required=True) 
    parser.add_argument("--steps", default=3000, type=int, required=True) 
    parser.add_argument("--eval_batch", default=2, type=int, required=True)
    parser.add_argument("--lr", default=5e-6, type=float) 
    parser.add_argument("--grad_accum_step", default=8, type=int)
    parser.add_argument("--warmup_step", default=10, type=int)
    parser.add_argument("--save_version", required=True)
    return parser.parse_args()



if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True

    args = args_parse()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, "configs", "saved_model_location.yaml")

    with open(yaml_path, "r") as file:
        configs = yaml.safe_load(file)

    step2_model_path = configs["step2"]["local_path"]
    if not os.path.exists(step2_model_path):
        step2_model_path = configs["step2"]["hub_id"]

    if args.peft.lower() in ["lora", "qlora"]:
        is_4bit = (args.peft.lower() == "qlora")
        model, processor = FastVisionModel.from_pretrained(
            model_name=step2_model_path,
            load_in_4bit=is_4bit,
            use_gradient_checkpointing="unsloth"
        )
    elif args.peft.lower() in ["fft"]:
        model, processor = FastVisionModel.from_pretrained(
            model_name=step2_model_path,
            use_gradient_checkpointing="unsloth"
        )

    model_name = configs["trained_model"] 

    save_dir = os.path.join(current_dir, "models", f"{model_name}_{args.peft}_mongolian_dpo") 
    
    hub_model_id = f"{args.trainer}/{model_name}-{args.peft}-mongolian-dpo-ver_{args.save_version}"
    
    log_file = setup_logging(current_dir, f"{model_name}-{args.peft}-mongolian-dpo-ver_{args.save_version}")

    os.makedirs(save_dir, exist_ok=True)

    tokenizer = processor.tokenizer 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
        tokenizer.pad_token_id = tokenizer.eos_token_id


    dataloader = CustomDataLoader(current_dir=current_dir, tokenizer=tokenizer, dataset_name="dpo_data")
    train_set, test_set = dataloader.load_data()

    dpo_config = DPOConfig(
        beta=0.1, 
        output_dir=save_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch, 
        gradient_accumulation_steps=args.grad_accum_step,
        warmup_steps=args.warmup_step,
        max_steps=args.steps,
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=50,
        load_best_model_at_end=True,
        greater_is_better=False,
        save_total_limit=2,
        lr_scheduler_type="cosine",
        dataloader_num_workers=4, 
        dataloader_pin_memory=True,
        push_to_hub=True,
        hub_model_id=hub_model_id,
        report_to=["tensorboard"],
        optim="adamw_8bit",
        seed=3407,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None, 
        args=dpo_config,
        tokenizer=tokenizer,
        train_dataset=train_set,
        eval_dataset=test_set,
        callbacks=[CustomLogCallback()]
    )

    logging.info(f"DPO Training started with\n" + ", ".join(f"{k}: {v}" for k, v in vars(args).items()) + "\n")
    
    last_checkpoint = get_last_checkpoint(save_dir) if os.path.exists(save_dir) else None
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(save_dir)
    trainer.push_to_hub("DPO Training completed!")

    config_data = {
        "step3": {
            "local_path": save_dir,
            "hub_id": hub_model_id
        }
    }

    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as file:
            existing_data = yaml.safe_load(file) or {} 
            existing_data.update(config_data)
            config_data = existing_data
    
    with open(yaml_path, "w") as file:
        yaml.dump(config_data, file, default_flow_style=False)


    logging.info(f"DPO Training finished\nModel saved: {save_dir}\nModel configs saved: {yaml_path}\n\n\n")

    del model, trainer, processor
    gc.collect() 
    torch.cuda.empty_cache()





    





