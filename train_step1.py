#=====================================================================================
# FOR REAL TRAINING Qwen/Qwen3.5-2B-Base
# FOR SCRIPT DEMONSTRATION Qwen/Qwen3.5-0.8B
#=====================================================================================
import torch 
from datasets import load_from_disk, load_dataset, concatenate_datasets
from transformers import (
    TrainingArguments,
    Trainer,
)
import evaluate 
from huggingface_hub import login 
import gc 
from dotenv import load_dotenv
import os 
from trl import SFTTrainer, SFTConfig
import bitsandbytes as bnb 
from unsloth import FastVisionModel
import argparse
from transformers.trainer_utils import get_last_checkpoint

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

def args_parse():
    parser = argparse.ArgumentParser(description="TRAINGING HYPERPARAMETERS")
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
        help="batch_size: 8, 16, etc (default 8)",
        default=8,
        type=int,
        required=True
    )
    parser.add_argument(
        "--epochs",
        help="epochs 10, 20, 30, .etc (default 10)",
        default=10,
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


def format_text(batch):
    formatted_text = []

    for text in batch["text"]:
        formatted_text.append(text + EOS_TOKEN)
    
    return {"text": formatted_text}


if __name__ == "__main__":
    args = args_parse()

    model_choices = {
        "debug": "Qwen/Qwen3.5-0.8B",
        "real": "Qwen/Qwen3.5-2B-Base"
    }
    model_name = model_choices[args.model].split("/")[-1].lower()

    save_dir = f"Mongolian_LLM/models/{model_name}_mongolian"
    cache_dir = "data/cache"

    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(f"{cache_dir}/train_set") and os.path.exists(f"{cache_dir}/test_set"):
        train_set = load_from_disk(f"{cache_dir}/train_set")
        test_set = load_from_disk(f"{cache_dir}/test_set")
    else:
        fulldataset = load_dataset("Ganaa0614/mongolian-text-dataset")
        splitted_dataset = fulldataset.train_test_split(test_size=0.1, seed=42)

        train_set = splitted_dataset["train"].map(format_text)
        test_set = splitted_dataset["trest"].map(format_text)

        train_set.save_to_disk(f"{cache_dir}/train_set")
        test_set.save_to_disk(f"{cache_dir}/test_set")

        del fulldataset
        del splitted_dataset
        gc.collect()


    MAX_SEQ_LEN = 2048

    
    if args.peft.lower() in ["lora", "qlora"]:
        is_4bit = (args.peft.lower() == "qlora")
        model, processor = FastVisionModel.from_pretrained(
            model_name=model_choices[args.model],
            load_in_4bit=is_4bit,
            use_gradient_checkpointing="unsloth"
        )

        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=16,
            lora_alpha=16,
            lora_dropout=0,
            random_state=42,
        )
    
    elif args.peft.lower() in ["fft"]:
        model, processor = FastVisionModel.from_pretrained(
            model_name=model_choices[args.model],
            use_gradient_checkpointing="unsloth"
        )

    tokenizer = processor.tokenizer 
    EOS_TOKEN = tokenizer.eos_token 


    sft_config = SFTConfig(
        output_dir=save_dir,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch,
        gradient_accumulation_steps=args.grad_accum_step,
        warmup_steps=args.warmup_step,
        num_train_epochs=args.epochs,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        save_steps=100,
        eval_accumulation_steps=100,
        logging_steps=50,
        load_best_model_at_end=True,
        greater_is_better=False,
        save_total_limit=2,
        push_to_hub=True,
        hub_model_id=f"Ganaa0614/{model_choices[args.model]}-mongolian-ver_{args.save_version}",
        report_to=["tensorboard"],
        dataset_num_proc=1,
        optim="adamw_8bit",
        seed=3407
    )

    last_checkpoint = None 

    if os.path.exists(save_dir) and os.listdir(save_dir):
        last_checkpoint = get_last_checkpoint(save_dir)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        tokenizer=tokenizer, 
        train_dataset=train_set,
        eval_dataset=test_set,
        processing_class=processor
    )

    trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.save_model(save_dir)
    trainer.push_to_hub("Training completed!")

    print(f"Model saved at {save_dir}")

    del model 
    del trainer 
    del processor

    gc.collect() 
    torch.cuda.empty_cache()    



