import torch
import gradio as gr
from unsloth import FastLanguageModel, FastVisionModel
from transformers import AutoTokenizer

print("[1] Step 1 Model (Continued Pre-Training)")
print("[2] Step 2 Model (QA Fine-Tuned)")
choice = input("Enter number:").strip()

step_key = "step2" if choice == "2" else "step1"

models = {
    "1": "Ganaa0614/Qwen3.5-2B-Base-qlora-mongolian-text-ver_0.1",
    "2": "Ganaa0614/Qwen3.5-2B-Base-qlora-mongolian-qa-ver_0.2",
    "3": "Ganaa0614/Qwen3.5-2B-Base-qlora-mongolian-qa-ver_0.3",
    "4": "Bokhbat/Qwen3.5-2B-Base-qlora-mongolian-dpo-ver_0.1"
}

tokenizer = AutoTokenizer.from_pretrained(
    models[choice], 
    use_fast=False 
)

model, _ = FastLanguageModel.from_pretrained(
    model_name = models[choice],
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


if __name__ == "__main__":
    demo.launch(share=True)


