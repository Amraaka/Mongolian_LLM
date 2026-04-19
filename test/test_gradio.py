import os
import yaml
import torch
import gradio as gr
from unsloth import FastLanguageModel, FastVisionModel
from transformers import AutoTokenizer

print("[1] Step 1 Model (Continued Pre-Training)")
print("[2] Step 2 Model (QA Fine-Tuned)")
choice = input("Enter 1 or 2: ").strip()

step_key = "step2" if choice == "2" else "step1"

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
yaml_path = os.path.join(current_dir, "configs", "saved_model_location.yaml")

with open(yaml_path, "r") as f:
    configs = yaml.safe_load(f)

model_path = configs[step_key]["local_path"]
print(f"\nLoading {step_key.upper()} from: {model_path}...\n")


tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    use_fast=False 
)

model, _ = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 1024,
    load_in_4bit = True, 
)

FastLanguageModel.for_inference(model) 


def generate_response(message, history):
    messages = [
        {"role": "user", "content": message}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True 
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens = 256,
        use_cache = True,
        temperature = 0.3,    
        top_p = 0.85,
        repetition_penalty = 1.15,
        pad_token_id = tokenizer.eos_token_id
    )
    
    full_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    clean_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    clean_prompt_decoded = tokenizer.decode(tokenizer.encode(clean_prompt), skip_special_tokens=True)
    
    response = full_text.replace(clean_prompt_decoded, "").strip()
    
    return response


demo = gr.ChatInterface(
    fn=generate_response,
    title=f"Mongolian Qwen 3.5 Test ({step_key.upper()})",
    examples=["Монгол улсын нийслэл хот бол", "Хиймэл оюун ухаан гэж юу вэ?"],
)

cert_path = os.path.join(current_dir, "keysforce_https", "gantumur-desktop.tail981298.ts.net.crt")
key_path = os.path.join(current_dir, "keysforce_https", "gantumur-desktop.tail981298.ts.net.key")

if __name__ == "__main__":
    demo.launch(share=False)


