import torch 
from unsloth import FastVisionModel
import os 
import string
import collections
import numpy as np 
import yaml
from Mongolian_LLM.utils.utils import CustomDataLoader
from tqdm import tqdm

def normalize_text(text):
    if not isinstance(text, str): return ""
    def remove_punc(t):
        exclude = set(string.punctuation)
        return " ".join(char for char in t if char not in exclude)
    return " ".join(remove_punc(text.lower()).split())

def exact_match_metric(prediction, reference):
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0

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


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, "configs", "saved_model_location.yaml")

    with open(yaml_path, "r") as file:
        configs = yaml.safe_load(file)
    
    stage2_model_path = configs["step2"]["local_path"]

    model, processor = FastVisionModel.from_pretrained(
        model_name=stage2_model_path,
        load_in_4bit=True,
    )
    FastVisionModel.for_inference(model) 
    tokenizer = processor.tokenizer

    dataloader = CustomDataLoader(current_dir=current_dir, tokenizer=tokenizer, dataset_name="qa_data")
    _, test_set = dataloader.load_data() 

    em_scores = []
    f1_scores = []

    print("Starting Generative Evaluation...")
    
    for item in tqdm(test_set):
        prompt = item["text"].split("<|im_start|>assistant\n")[0] + "<|im_start|>assistant\n"
        true_answer = item["output"]

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128, 
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_answer = full_output.split("assistant\n")[-1].strip()
        
        em_scores.append(exact_match_metric(generated_answer, true_answer))
        f1_scores.append(f1_score_metric(generated_answer, true_answer))

    print("\n=== FINAL STAGE 2 QA SCORES ===")
    print(f"Exact Match (EM): {np.mean(em_scores) * 100:.2f}%")
    print(f"F1 Score:         {np.mean(f1_scores) * 100:.2f}%")