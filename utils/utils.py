import os 
from transformers import TrainerCallback
import logging
import gc 
import os 
from datasets import load_dataset, load_from_disk
import yaml 
from typing import Any, List, Dict


def setup_logging(current_dir, train_session_name):
    log_dir = os.path.join(current_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{train_session_name}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True, 
        handlers=[
            logging.FileHandler(log_file, mode='a'), 
            logging.StreamHandler()                  
        ]
    )
    
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    
    return log_file


class CustomLogCallback(TrainerCallback):

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            log_str = f"Step {state.global_step}: " + ", ".join([f"{k}={v}" for k, v in logs.items()])
            logging.info(log_str)


class CustomDataLoader():
    
    def __init__(self, current_dir, tokenizer: Any, dataset_name: str):
        self.current_dir = current_dir
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name.lower()
        self.yaml_path = os.path.join(self.current_dir, "configs", "data_locations.yaml")
        self.EOS_TOKEN = self.tokenizer.eos_token
        self.maps = {"text_data": self.format_text, 
                     "qa_data": self.format_qa, 
                     "dpo_data": self.format_dpo, 
                     "instruction_data": None, 
                     "iam_data": None
                    }
        

    def format_text(self, batch):
        formatted_text = []
        for text in batch["text"]:
            formatted_text.append(text + self.EOS_TOKEN) #"<|endoftext|>"
        return {"text": formatted_text}


    def format_qa(self, batch):
        formatted_texts = []

        for question, answer in zip(batch["instruction"], batch["output"]):
            text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
            formatted_texts.append(text)

        return {"text": formatted_texts}
        

    def format_dpo(self, batch):
        prompts = [] 
        chosens = []
        rejecteds = []

        for prompt, chosen, rejected in zip(batch["prompt"], batch["chosen"], batch["rejected"]):
            prompts.append(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
            chosens.append(f"{chosen}<|im_end|>")
            rejecteds.append(f"{rejected}<|im_end|>")
        
        return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds} 


    def load_data(self):
        with open(self.yaml_path, "r") as file:
            configs = yaml.safe_load(file)

        mapped_dir = os.path.join(self.current_dir, configs[self.dataset_name]["local_path"]["mapped"])

        if os.path.exists(f"{mapped_dir}/train_set") and os.path.exists(f"{mapped_dir}/test_set"):
            train_set = load_from_disk(f"{mapped_dir}/train_set")
            test_set = load_from_disk(f"{mapped_dir}/test_set")
            return train_set, test_set
        else:
            local_dir = configs[self.dataset_name]["local_path"]["raw"]

            if os.path.exists(local_dir):
                fulldataset = load_from_disk(local_dir)
            else:
                fulldataset = load_dataset(configs[self.dataset_name]["hub_id"])
                
            if self.dataset_name == "qa_data":
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            train_set = fulldataset["train"].map(self.maps[self.dataset_name], batched=True)
            test_set = fulldataset["validation"].map(self.maps[self.dataset_name], batched=True)

            train_set.save_to_disk(f"{mapped_dir}/train_set")
            test_set.save_to_disk(f"{mapped_dir}/test_set")

            del fulldataset
            gc.collect()

            return train_set, test_set


