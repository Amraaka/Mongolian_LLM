"""
STEP 4 GRADIO DEMO — web-search Mongolian assistant with a browser UI.

Wraps the same Round 1 / Round 2 pipeline as run_search_demo.py in a Gradio
interface so you can demo it visually in a browser.

Setup (if gradio is not installed):
    pip install gradio

Usage:
    python src/run_search_demo_gradio.py                # local only (localhost:7860)
    python src/run_search_demo_gradio.py --share        # public link via Gradio tunnel
"""
import unsloth
import argparse
import json
import os
import re
import sys
from pathlib import Path

import gradio as gr
import torch
import yaml
from dotenv import load_dotenv
from huggingface_hub import login
from tavily import TavilyClient
from unsloth import FastVisionModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()
if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))

CURRENT_DIR = Path(__file__).resolve().parent.parent

WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": "Интернетээс хайлт хийж, шинэлэг мэдээлэл авах. Хайлтын асуултыг богино, ойлгомжтой бичих.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Хайх асуулт (Монгол эсвэл Англи хэлээр)",
            }
        },
        "required": ["query"],
    },
}
TOOLS_JSON = json.dumps([WEB_SEARCH_TOOL], ensure_ascii=False)
SYSTEM_PROMPT = (
    f"Та хэрэглэгчид туслахын тулд дараах хэрэгслүүдийг ашиглаж болно.\n"
    f"<tools>\n{TOOLS_JSON}\n</tools>"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true", help="Expose via Gradio public tunnel")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()


def load_step4_model():
    yaml_path = CURRENT_DIR / "configs" / "saved_model_location.yaml"
    with yaml_path.open() as f:
        configs = yaml.safe_load(f)
    if "step4" not in configs:
        sys.exit("step4 entry missing from configs/saved_model_location.yaml — train first.")

    path = configs["step4"]["local_path"]
    if not os.path.isabs(path):
        path = str(CURRENT_DIR / path)
    if not os.path.exists(path):
        path = configs["step4"]["hub_id"]
        print(f"Local Step 4 model not found, loading from Hub: {path}")
    else:
        print(f"Loading Step 4 model from {path}")

    model, processor = FastVisionModel.from_pretrained(
        model_name=path,
        load_in_4bit=True,
    )
    FastVisionModel.for_inference(model)
    return model, processor.tokenizer


def build_round1_prompt(question: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_round2_prompt(round1_prompt: str, tool_call_text: str, tool_response_json: str) -> str:
    return (
        f"{round1_prompt}{tool_call_text}<|im_end|>\n"
        f"<|im_start|>tool\n<tool_response>{tool_response_json}</tool_response><|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float = 0.3) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)


def extract_tool_call(text: str):
    m = re.search(r"<tool_call>\s*(\{.+?\})\s*</tool_call>", text, re.DOTALL)
    if not m:
        return None, None
    raw = m.group(0)
    try:
        return raw, json.loads(m.group(1))
    except json.JSONDecodeError:
        return raw, None


def format_tool_response(tavily_response: dict, top_k: int = 3, content_chars: int = 600):
    snippets = []
    sources = []
    for r in (tavily_response.get("results") or [])[:top_k]:
        content = (r.get("content") or "").strip().replace("\n", " ")
        if len(content) > content_chars:
            content = content[:content_chars].rstrip() + "..."
        url = r.get("url", "")
        title = r.get("title", "")
        snippets.append({"title": title, "url": url, "content": content})
        if url:
            sources.append((title or url, url))
    return json.dumps(snippets, ensure_ascii=False), sources


def strip_after_imend(text: str) -> str:
    return text.split("<|im_end|>")[0].strip()


def make_pipeline(model, tokenizer, tavily):
    def pipeline(question: str):
        if not question or not question.strip():
            return "", "", ""
        question = question.strip()

        # ROUND 1 — model emits tool call
        prompt = build_round1_prompt(question)
        round1 = generate(model, tokenizer, prompt, max_new_tokens=200)
        raw_call, parsed = extract_tool_call(round1)

        if not raw_call or not parsed:
            answer = "❌ Загвар хүчинтэй tool_call гаргаж чадсангүй."
            internals = f"**Raw model output (Round 1):**\n```\n{round1[:500]}\n```"
            return answer, internals, ""

        try:
            query = parsed["arguments"]["query"]
        except (KeyError, TypeError):
            return f"❌ tool_call дотор query байхгүй: `{parsed}`", "", ""

        # PYTHON — call Tavily
        try:
            tavily_resp = tavily.search(query=query, search_depth="basic", max_results=5, include_answer=True)
        except Exception as e:
            return f"❌ Tavily call failed: {e}", "", ""

        tool_response_json, sources = format_tool_response(tavily_resp)

        # ROUND 2 — model emits final answer
        round2_prompt = build_round2_prompt(prompt, raw_call, tool_response_json)
        round2 = generate(model, tokenizer, round2_prompt, max_new_tokens=500, temperature=0.4)
        answer = strip_after_imend(round2)

        internals_md = (
            f"**🛠 Tool call (Round 1 output):**\n```json\n{json.dumps(parsed, ensure_ascii=False, indent=2)}\n```\n\n"
            f"**🔎 Search query sent to Tavily:** `{query}`\n\n"
            f"**📄 Top snippets returned:**\n```json\n{json.dumps(json.loads(tool_response_json), ensure_ascii=False, indent=2)[:1500]}\n```"
        )
        sources_md = "\n".join([f"- [{title}]({url})" for title, url in sources]) if sources else "_No sources_"

        return answer, internals_md, sources_md

    return pipeline


def main():
    args = parse_args()

    if not os.getenv("TAVILY_API_KEY"):
        sys.exit("TAVILY_API_KEY not set in .env")
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    print("Loading model...")
    model, tokenizer = load_step4_model()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    pipeline = make_pipeline(model, tokenizer, tavily)

    with gr.Blocks(title="Mongolian Search Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🔍 Монгол хайлтын туслах\n"
            "Qwen3.5-2B загвар Tavily-тэй холбогдож интернетээс хайлт хийж монгол хэлээр хариулна."
        )

        with gr.Row():
            question_box = gr.Textbox(
                label="Асуулт",
                placeholder="Жишээ: Чингис хаан хэн бэ?",
                lines=2,
                scale=4,
            )
            submit_btn = gr.Button("🔍 Хайх", variant="primary", scale=1)

        answer_box = gr.Markdown(label="💬 Хариулт")

        with gr.Accordion("📎 Эх сурвалж", open=True):
            sources_box = gr.Markdown()

        with gr.Accordion("🔧 Загвар хэрхэн ажилласан (Round 1 → Tavily → Round 2)", open=False):
            internals_box = gr.Markdown()

        submit_btn.click(
            fn=pipeline,
            inputs=question_box,
            outputs=[answer_box, internals_box, sources_box],
        )
        question_box.submit(
            fn=pipeline,
            inputs=question_box,
            outputs=[answer_box, internals_box, sources_box],
        )

        gr.Examples(
            examples=[
                "Чингис хаан хэн бэ?",
                "Улаанбаатар хотын талаар мэдээлэл өгнө үү.",
                "Лионел Месси ямар алдартай вэ?",
                "Хиймэл оюун ухаан гэж юу вэ?",
                "Эверест уул хаана байрладаг вэ?",
            ],
            inputs=question_box,
            label="Жишээ асуултууд (товшоод турших)",
        )

    demo.launch(server_port=args.port, share=True)


if __name__ == "__main__":
    main()
