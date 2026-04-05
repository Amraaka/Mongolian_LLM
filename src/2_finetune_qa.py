#=====================================================================================
# STEP2 QA FINE TUNING
#=====================================================================================

import torch 
from datasets import load_from_disk, load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from huggingface_hub import login 
import gc 
from dotenv import load_dotenv
import os 
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from unsloth import FastVisionModel
import argparse
from transformers.trainer_utils import get_last_checkpoint
import string
import collections
import numpy as np 
import logging
import yaml
from utils.utils import setup_logging, CustomDataLoader, CustomLogCallback

load_dotenv()
login(token=os.getenv("HF_TOKEN"))


def normalize_text(text):
    if not isinstance(text, str): 
        return ""
    
    def remove_punc(t):
        exclude = set(string.punctuation)
        return " ".join(char for char in t if char not in exclude)
    
    return " ".join(remove_punc(text.lower().split()))


#=====================================================================================
# EVALUATION FUNCTION EM, F1-SCORE

def exact_match_metric(prediction, reference):
    return 1.0 if prediction.strip() == reference.strip() else 0.0

def f1_score_metric(prediction, reference):
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    common = collections.Counter(pred_tokens) & collections.Counter(ref_tokens)
    num_same = sum(common.values())

    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 1.0 if pred_tokens == ref_tokens else 0.0
    if num_same == 0:
        return 0.0 
    
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(ref_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_metrics(eval_preds):
    preds, labels = eval_preds 

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    em_scores = [] 
    f1_scores = [] 

    for pred, label in zip(decoded_preds, decoded_labels):
        em_scores.append(exact_match_metric(pred, label))
        f1_scores.append(f1_score_metric(pred, label))
    
    return {
        "exact_match": em_scores,
        "f1": np.mean(f1_scores) 
    }

#=====================================================================================


def args_parse():
    parser = argparse.ArgumentParser(description="TRAINGING HYPERPARAMETERS")
    parser.add_argument(
        "--trainer",
        help="Who is training this model enter your HUGGINGFACE_USER_NAME (default Ganaa0614)",
        default="Ganaa0614",
        choices=["Ganaa0614"] # Enter hugginface neree oruulah (AMARAA)
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



if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True

    args = args_parse()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, "configs", "saved_model_location.yaml")

    with open(yaml_path, "r") as file:
        configs = yaml.safe_load(file)
    
    step1_model_path = configs["step1"]["local_path"]

    if not os.path.exists(step1_model_path):
        step1_model_path = configs["step1"]["hub_id"]


    if args.peft.lower() in ["lora", "qlora"]:
        is_4bit = (args.peft.lower() == "qlora")
        model, processor = FastVisionModel.from_pretrained(
            model_name=step1_model_path,
            load_in_4bit=is_4bit,
            use_gradient_checkpointing="unsloth"
        )
    
    elif args.peft.lower() in ["fft"]:
        model, processor = FastVisionModel.from_pretrained(
            model_name=step1_model_path,
            use_gradient_checkpointing="unsloth"
        )

    model_name = configs["trained_model"]

    save_dir = os.path.join(current_dir, "models", f"{model_name}_{args.peft}_mongolian_qa") 
    hub_model_id = f"{args.trainer}/{model_name}-{args.peft}-mongolian-qa-ver_{args.save_version}"
    log_file = setup_logging(current_dir, f"{model_name}-{args.peft}-mongolian-qa-ver_{args.save_version}")

    os.makedirs(save_dir, exist_ok=True)


    tokenizer = processor.tokenizer 
    EOS_TOKEN = tokenizer.eos_token 

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataloader = CustomDataLoader(current_dir=current_dir, tokenizer=tokenizer, dataset_name="qa_data")
    train_set, test_set = dataloader.load_data()

    MAX_SEQ_LEN = 2048

    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer
    )

    seq2seq_config = Seq2SeqTrainingArguments(
        output_dir=save_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1, 
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
        dataloader_num_workers=4, 
        dataloader_pin_memory=True,
        push_to_hub=True,
        hub_model_id=hub_model_id,
        report_to=["tensorboard"],
        optim="adamw_8bit",
        seed=3407,
        predict_with_generate=True,     
        generation_max_length=128,      
        generation_num_beams=1,         
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=seq2seq_config,
        tokenizer=tokenizer,
        train_dataset=train_set,
        eval_dataset=test_set,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[CustomLogCallback()]
    )

    logging.info(f"Trainnig started with\n" + ", ".join(f"{k}: {v}" for k, v in vars(args).items()) + "\n")
    
    last_checkpoint = None 
    
    if os.path.exists(save_dir) and os.listdir(save_dir):
        last_checkpoint = get_last_checkpoint(save_dir)

    trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.save_model(save_dir)
    trainer.push_to_hub("Training completed!")

    print(f"Model saved at {save_dir}")

    config_data  = {
        "step2": {
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

    logging.info(f"Training finished\nModel saved: {save_dir}\nModel configs saved: {yaml_path}\n\n\n")
    

    del model 
    del trainer 
    del processor

    gc.collect() 
    torch.cuda.empty_cache()    



