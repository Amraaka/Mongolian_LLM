#=====================================================================================
# FOR REAL TRAINING Qwen/Qwen3.5-2B-Base
# FOR SCRIPT DEMONSTRATION Qwen/Qwen3.5-0.8B
#=====================================================================================
# STEP1 CONTINUED PRETRIANING ON MONGOLIAN LANGUAGE
#=====================================================================================
import unsloth
import torch 
from huggingface_hub import login 
import gc 
from dotenv import load_dotenv
import os 
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import (
    AutoProcessor, 
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig
import argparse
from transformers.trainer_utils import get_last_checkpoint
import yaml
from utils.utils import setup_logging, CustomLogCallback, CustomDataLoader

import logging
import math 
import numpy as np 
from torch.nn import CrossEntropyLoss



load_dotenv()
login(token=os.getenv("HF_TOKEN"))
os.environ["WANDB_DISABLED"] = "true"

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    
    if isinstance(logits, tuple):
        logits = logits[0]

    logits_tensor = torch.tensor(logits)
    labels_tensor = torch.tensor(labels)

    shift_logits = logits_tensor[..., :-1, :].contiguous()
    shift_labels = labels_tensor[..., 1:].contiguous()
    
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    try:
        perplexity = math.exp(loss.item())
    except OverflowError:
        perplexity = float("inf")

    return {
        "perplexity": perplexity,
        "eval_loss_recalculated": loss.item() 
    }

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]

    pred_ids = torch.argmax(logits, dim=-1)

    return pred_ids 


def args_parse():
    parser = argparse.ArgumentParser(description="TRAINGING HYPERPARAMETERS")
    parser.add_argument(
        "--trainer",
        help="Who is traininglog_setup this model enter your HUGGINGFACE_USER_NAME (default Ganaa0614)",
        default="Ganaa0614",
        choices=["Ganaa0614"] # Enter hugginface neree oruulah (AMARAA)
    )
    parser.add_argument(
        "--model", 
        help="For code demonstration (debug) or real training", 
        choices=["debug", "real"], 
        default="debug",
        required=True
    )
    parser.add_argument(
        "--peft", 
        help="Training method FFT or QLoRA", 
        choices=["qlora", "fft", "lora"],
        default="fft",
        required=True
    )
    parser.add_argument(
        "--batch_size",
        help="batch_size: 8, 16, etc (default 1)",
        default=1,
        type=int,
        required=True
    )
    parser.add_argument(
        "--steps",
        help="steps 6000, 7000, 8000, .etc (default 6000)",
        default=6000,
        type=int,
        required=True
    )
    parser.add_argument(
        "--eval_batch",
        help="eval batch size 4, 8, .etc (default 4)",
        default=4,
        type=int,
        required=True
    )
    parser.add_argument(
        "--lr",
        help="Learning rate 2e-4, 5e-5, .etc (default 2e-4)",
        default=2e-4,
        type=float,
    )
    parser.add_argument(
        "--grad_accum_step",
        help="Gradient accumalation steps 4, 8, .etc (default 4)",
        default=4, 
        type=int,
    )
    parser.add_argument(
        "--warmup_step",
        help="Warm up steps 3, 4, .etc (default 3)",
        default=3,
        type=int,
    )
    parser.add_argument(
        "--save_version",
        help="model saved verison name (0.1, 0.2, .etc)",
        required=True
    )
    return parser.parse_args()

# def tokenize_function(examples):
#         return tokenizer(
#             examples["text"], 
#             truncation=True, 
#             max_length=MAX_SEQ_LEN
#     )


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True

    args = args_parse()

    model_choices = {
        "debug": "Qwen/Qwen3.5-0.8B",
        "real": "Qwen/Qwen3.5-2B-Base"
    }

    model_name = model_choices[args.model].split("/")[-1]
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(current_dir, "models", f"{model_name}_{args.peft}_mongolian_text_{args.save_version}") 

    hub_model_id = f"{args.trainer}/{model_name}-{args.peft}-mongolian-text-ver_{args.save_version}"
    log_file = setup_logging(current_dir, f"{model_name}-{args.peft}-mongolian-text-ver_{args.save_version}")

    os.makedirs(save_dir, exist_ok=True)

    prev_hub_id = f"{args.trainer}/{model_name}-{args.peft}-mongolian-text-ver_{args.save_version}"

    if args.peft.lower() in ["lora", "qlora"]:
        is_4bit = (args.peft.lower() == "qlora")
        model, processor = FastVisionModel.from_pretrained(
            model_name=model_choices[args.model],
            load_in_4bit=is_4bit if args.peft.lower() == "qlora" else None,
            use_gradient_checkpointing="unsloth"
        )

        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision=False,
            finetune_language=True,
            finetune_attention_modules=True,
            finetune_mlp=True,
            r=16,
            lora_alpha=16,
            lora_dropout=0,
            random_state=42,
        )
    
    elif args.peft.lower() in ["fft"]:
        model, processor = FastVisionModel.from_pretrained(
            model_name=model_choices[args.model],
            use_gradient_checkpointing="unsloth",
            load_in_4bit=False
        )

    # processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-2B-Base")
    # model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-2B-Base")

    tokenizer = processor.tokenizer 
    MAX_SEQ_LEN = 1024

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    
    logging.info(f"Loaded model and processer with {args.peft}!")

    dataloader = CustomDataLoader(current_dir=current_dir,tokenizer=tokenizer, dataset_name="text_data")
    train_set, test_set = dataloader.load_data()
    # test_set = test_set.shuffle(seed=42).select(range(2))    # for 16gb system ram machiine
    
    # training_args = TrainingArguments(
    #     output_dir=save_dir,
    #     per_device_train_batch_size=args.batch_size,
    #     per_device_eval_batch_size=args.eval_batch,
    #     gradient_accumulation_steps=args.grad_accum_step,
    #     warmup_steps=args.warmup_step,
    #     max_steps=args.steps,
    #     gradient_checkpointing=True,
    #     fp16=False,
    #     bf16=True,
    #     eval_strategy="steps",
    #     save_strategy="steps",
    #     eval_steps=1000,
    #     save_steps=1000,
    #     logging_steps=100,
    #     load_best_model_at_end=True,
    #     greater_is_better=False,
    #     save_total_limit=2,
    #     dataloader_num_workers=0,        
    #     dataloader_pin_memory=True,
    #     push_to_hub=True,
    #     hub_model_id=hub_model_id,
    #     report_to=["tensorboard"],
    #     optim="adamw_8bit",
    #     seed=3407,
    #     remove_unused_columns=False,
        
    # )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,                  
    #     processing_class=tokenizer,
    #     train_dataset=train_set,
    #     eval_dataset=test_set,  
    #     compute_metrics=compute_metrics,
    #     callbacks=[CustomLogCallback()],
    #     data_collator=data_collator
    # )



    sft_config = SFTConfig(
        output_dir=save_dir,
        dataset_text_field="text",
        max_length=MAX_SEQ_LEN,   
        eos_token=tokenizer.eos_token,              
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
        eval_steps=1000,
        save_steps=1000,
        logging_steps=100,
        load_best_model_at_end=True,
        greater_is_better=False,
        save_total_limit=2,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        push_to_hub=True,
        hub_model_id=hub_model_id,
        lr_scheduler_type="cosine",      # Added cosine lr scheduler
        report_to=["tensorboard"],
        optim="adamw_8bit",
        seed=3407,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,                  
        processing_class=tokenizer,
        train_dataset=train_set,
        data_collator=collator,
        eval_dataset=test_set,  
        compute_metrics=compute_metrics,
        callbacks=[CustomLogCallback()],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

   
    logging.info(f"Trainnig started with\n" + ", ".join(f"{k}: {v}" for k, v in vars(args).items()) + "\n")

    last_checkpoint = None 

    if os.path.exists(save_dir) and os.listdir(save_dir):
        last_checkpoint = get_last_checkpoint(save_dir)

    trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.save_model(save_dir)
    trainer.push_to_hub("Training completed!") 
    print(f"Model saved at {save_dir}")

    configs_dir = os.path.join(current_dir, "configs")
    os.makedirs(configs_dir, exist_ok=True)

    yaml_path = os.path.join(configs_dir, "saved_model_location.yaml")
    relative_save_dir = os.path.relpath(save_dir, current_dir)

    config_data = {
        "trained_model": model_name,
        "step1": {
            "local_path": relative_save_dir,
            "hub_id": hub_model_id
        }
    }

    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f: 
            existing_data = yaml.safe_load(f) or {}
            existing_data.update(config_data) 
            config_data = existing_data
    
    yaml_str = yaml.dump(config_data, default_flow_style=False, sort_keys=False)

    yaml_str = yaml_str.replace("\nstep1:", "\n\nstep1:")
    yaml_str = yaml_str.replace("\nfine tuned model:", "\n\nfine tuned model:")
    yaml_str = yaml_str.replace("\nstep2:", "\n\nstep2:")

    with open(yaml_path, "w") as f:
        f.write(yaml_str)

        
    logging.info(f"Training finished\nModel saved: {save_dir}\nModel configs saved: {yaml_path}\n\n\n")

    del model 
    del trainer 
    del processor

    gc.collect() 
    torch.cuda.empty_cache()    

 



