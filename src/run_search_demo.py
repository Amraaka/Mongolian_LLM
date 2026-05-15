"""
STEP 4 INFERENCE DEMO — live web-search Mongolian assistant.

Loads the Step 4 fine-tuned model and connects it to live Tavily.
Two model.generate() calls with a Python orchestrator in between:

  Round 1: user question -> model emits <tool_call>{...}</tool_call>
  Python : parse JSON, call Tavily, format <tool_response>
  Round 2: model emits the final Mongolian answer

Usage:
    python src/run_search_demo.py
    python src/run_search_demo.py --verbose   # show tool call + raw snippets
    python src/run_search_demo.py --question "Чингис хаан хэн бэ?"
"""
import unsloth
import argparse
import json
import os
import re
import sys
from pathlib import Path

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

# Must match the WEB_SEARCH_TOOL used at training time (data_generation/build_search_dataset.py).
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
    p = argparse.ArgumentParser(description="Step 4 web-search demo")
    p.add_argument("--question", default=None, help="One-shot question (otherwise interactive REPL)")
    p.add_argument("--verbose", action="store_true", help="Print tool call + raw snippets for inspection")
    p.add_argument("--max-new-tokens-call", type=int, default=200)
    p.add_argument("--max-new-tokens-answer", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--top-k", type=int, default=3, help="Tavily results to feed back to model")
    p.add_argument("--content-chars", type=int, default=600, help="Truncate each snippet")
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
        use_gradient_checkpointing=False,
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
    """Append the model's tool call + Python's tool response + a fresh assistant turn."""
    return (
        f"{round1_prompt}{tool_call_text}<|im_end|>\n"
        f"<|im_start|>tool\n<tool_response>{tool_response_json}</tool_response><|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)


def extract_tool_call(text: str):
    """Find the first <tool_call>...</tool_call> block and parse its JSON.
    Returns (raw_call_text_including_tags, parsed_dict) or (None, None)."""
    m = re.search(r"<tool_call>\s*(\{.+?\})\s*</tool_call>", text, re.DOTALL)
    if not m:
        return None, None
    raw = m.group(0)
    try:
        parsed = json.loads(m.group(1))
        return raw, parsed
    except json.JSONDecodeError:
        return raw, None


def format_tool_response(tavily_response: dict, top_k: int, content_chars: int) -> tuple[str, list]:
    snippets = []
    sources = []
    for r in (tavily_response.get("results") or [])[:top_k]:
        content = (r.get("content") or "").strip().replace("\n", " ")
        if len(content) > content_chars:
            content = content[:content_chars].rstrip() + "..."
        url = r.get("url", "")
        snippets.append({"title": r.get("title", ""), "url": url, "content": content})
        if url:
            sources.append(url)
    return json.dumps(snippets, ensure_ascii=False), sources


def strip_after_imend(text: str) -> str:
    return text.split("<|im_end|>")[0].strip()


def answer_one(question: str, model, tokenizer, tavily: TavilyClient, args) -> None:
    print(f"\n🔍 Хайж байна...")

    # === ROUND 1: model emits tool call ===
    prompt = build_round1_prompt(question)
    round1 = generate(model, tokenizer, prompt, args.max_new_tokens_call, args.temperature)
    raw_call, parsed = extract_tool_call(round1)

    if not raw_call or not parsed:
        print("❌ Загвар хүчинтэй tool_call гаргаж чадсангүй.")
        if args.verbose:
            print("--- Model output ---")
            print(round1)
        return

    try:
        query = parsed["arguments"]["query"]
    except (KeyError, TypeError):
        print(f"❌ tool_call дотор 'arguments.query' талбар байхгүй: {parsed}")
        return

    if args.verbose:
        print(f"\n[TOOL CALL] {raw_call}")
        print(f"[SEARCH QUERY] {query}")

    # === Python orchestrator: call Tavily for real ===
    try:
        tavily_resp = tavily.search(query=query, search_depth="basic", max_results=5, include_answer=True)
    except Exception as e:
        print(f"❌ Tavily call failed: {e}")
        return

    tool_response_json, sources = format_tool_response(tavily_resp, args.top_k, args.content_chars)

    if args.verbose:
        print(f"\n[TOOL RESPONSE] {tool_response_json[:400]}...")

    # === ROUND 2: model emits the final answer ===
    round2_prompt = build_round2_prompt(prompt, raw_call, tool_response_json)
    round2 = generate(model, tokenizer, round2_prompt, args.max_new_tokens_answer, args.temperature)
    answer = strip_after_imend(round2)

    print(f"\n💬 {answer}")
    if sources:
        print(f"\nЭх сурвалж:")
        for u in sources:
            print(f"  - {u}")


def main():
    args = parse_args()

    if not os.getenv("TAVILY_API_KEY"):
        sys.exit("TAVILY_API_KEY not set in .env")
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    model, tokenizer = load_step4_model()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("\n🟢 Монгол хэлээр асуулт тавина уу (гарахдаа 'exit' эсвэл Ctrl-D)\n")

    if args.question:
        answer_one(args.question, model, tokenizer, tavily, args)
        return

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nБаяртай!")
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "гарах"):
            print("Баяртай!")
            break
        answer_one(question, model, tokenizer, tavily, args)


if __name__ == "__main__":
    main()
