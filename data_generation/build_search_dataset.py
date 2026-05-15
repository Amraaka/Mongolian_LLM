"""
STEP 4 DATASET BUILDER — web-search function-calling dataset.

Pipeline:
  1. Load Mongolian QA pairs (from data/qa_data/qa_mapped or HF Hub fallback).
  2. For each question, call Tavily web search and cache the response.
  3. Format each row as the 5 fields the chat template consumes:
       tools_json, question, tool_call_json, tool_response_json, answer
  4. Save as a HuggingFace dataset to data/instruction_data/raw.

Cache strategy: every Tavily response is appended to tavily_cache.jsonl as we
go. On rerun, cached queries are skipped — a crash or rate-limit halt loses no
progress and burns no extra quota.

Usage:
  python data_generation/build_search_dataset.py --limit 1000
  python data_generation/build_search_dataset.py --limit 3000 --top-k 3
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from dotenv import load_dotenv
from tavily import TavilyClient
from tqdm import tqdm

load_dotenv()

CURRENT_DIR = Path(__file__).resolve().parent.parent
QA_LOCAL_DIR = CURRENT_DIR / "data" / "qa_data" / "qa_mapped"
QA_HUB_ID = "Ganaa0614/mongolian-qa-dataset"
OUTPUT_DIR = CURRENT_DIR / "data" / "instruction_data" / "raw"
CACHE_PATH = CURRENT_DIR / "data_generation" / "tavily_cache.jsonl"

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


def parse_args():
    p = argparse.ArgumentParser(description="Build web-search function-calling dataset")
    p.add_argument("--limit", type=int, default=1000,
                   help="Max number of (question, search) pairs to build (default 1000)")
    p.add_argument("--top-k", type=int, default=3,
                   help="Top-k Tavily results to keep per query (default 3)")
    p.add_argument("--content-chars", type=int, default=600,
                   help="Truncate each result's content to this many chars (default 600)")
    p.add_argument("--sleep", type=float, default=1.1,
                   help="Seconds to sleep between Tavily calls (default 1.1)")
    p.add_argument("--search-depth", choices=["basic", "advanced"], default="basic",
                   help="Tavily search depth — 'advanced' costs 2 credits (default basic)")
    p.add_argument("--questions-file", type=str, default=None,
                   help="JSONL of {\"question\": ...} rows. If set, used instead of qa_data; "
                        "answer is filled from Tavily synthesized answer or top snippet.")
    p.add_argument("--answer-chars", type=int, default=400,
                   help="Max chars for snippet-derived answer when no ground truth exists (default 400)")
    p.add_argument("--split", choices=["train", "validation", "test", "all"], default="train",
                   help="(qa_data source only) Which QA split to draw questions from")
    p.add_argument("--shuffle-seed", type=int, default=42,
                   help="Seed for shuffling questions before slicing --limit (default 42)")
    return p.parse_args()


def load_qa_questions(split: str):
    """Return a HF Dataset of QA rows with at least 'question' and 'answer' columns."""
    if QA_LOCAL_DIR.exists():
        print(f"Loading QA data from local: {QA_LOCAL_DIR}")
        parts = []
        for s in (["train", "validation", "test"] if split == "all" else [split]):
            path = QA_LOCAL_DIR / f"{s}_set"
            if path.exists():
                parts.append(load_from_disk(str(path)))
            else:
                print(f"  WARN: split '{s}' not found at {path}, skipping")
        if not parts:
            sys.exit(f"No splits loaded from {QA_LOCAL_DIR}")
        return concatenate_datasets(parts) if len(parts) > 1 else parts[0]

    print(f"Local QA not found, falling back to HF Hub: {QA_HUB_ID}")
    ds = load_dataset(QA_HUB_ID)
    if split == "all":
        return concatenate_datasets([ds[s] for s in ds.keys()])
    return ds[split]


def load_questions_from_file(path: str):
    """Return a HF Dataset with at least a 'question' column. 'answer' will be empty."""
    p = Path(path)
    if not p.is_absolute():
        p = CURRENT_DIR / p
    if not p.exists():
        sys.exit(f"Questions file not found: {p}")
    print(f"Loading questions from file: {p}")
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "question" not in obj:
                continue
            rows.append({"question": obj["question"], "answer": obj.get("answer", "")})
    if not rows:
        sys.exit(f"No usable rows in {p}")
    return Dataset.from_list(rows)


def load_cache():
    """Cache shape: {query: tavily_response_dict}. Built from a jsonl append log."""
    cache = {}
    if not CACHE_PATH.exists():
        return cache
    with CACHE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                cache[entry["query"]] = entry["response"]
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(cache)} cached Tavily responses from {CACHE_PATH.name}")
    return cache


def append_cache(query: str, response: dict):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"query": query, "response": response}, ensure_ascii=False) + "\n")


def tavily_search_with_retry(client: TavilyClient, query: str, depth: str, max_retries: int = 3):
    """Call Tavily with simple exponential backoff. Returns response dict or None on failure."""
    for attempt in range(max_retries):
        try:
            return client.search(
                query=query,
                search_depth=depth,
                max_results=5,
                include_answer=True,
            )
        except Exception as e:
            wait = 5 * (2 ** attempt)
            print(f"  ERROR on query '{query[:50]}...': {e} — retrying in {wait}s")
            time.sleep(wait)
    print(f"  GIVING UP on query '{query[:50]}...' after {max_retries} retries")
    return None


def derive_answer(response: dict, max_chars: int) -> str:
    """When the question source has no ground-truth answer, derive one from Tavily.
    Prefer Tavily's synthesized 'answer' field; fall back to the top result's content."""
    tavily_answer = (response.get("answer") or "").strip()
    if tavily_answer:
        return tavily_answer[:max_chars].rstrip()
    results = response.get("results") or []
    if results:
        top = (results[0].get("content") or "").strip().replace("\n", " ")
        return top[:max_chars].rstrip()
    return ""


def format_tool_response(response: dict, top_k: int, content_chars: int) -> str:
    """Extract the fields the model will see — keep it compact."""
    snippets = []
    for r in (response.get("results") or [])[:top_k]:
        content = (r.get("content") or "").strip().replace("\n", " ")
        if len(content) > content_chars:
            content = content[:content_chars].rstrip() + "..."
        snippets.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": content,
        })
    return json.dumps(snippets, ensure_ascii=False)


def main():
    args = parse_args()

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        sys.exit("TAVILY_API_KEY not set in .env")
    client = TavilyClient(api_key=api_key)

    if args.questions_file:
        qa_ds = load_questions_from_file(args.questions_file)
        oversample = 1  # file is already curated, no need to oversample heavily
    else:
        qa_ds = load_qa_questions(args.split)
        oversample = 2  # qa_data is mixed quality — oversample to absorb filter rate
    qa_ds = qa_ds.shuffle(seed=args.shuffle_seed).select(range(min(args.limit * oversample, len(qa_ds))))
    print(f"Sampled {len(qa_ds)} candidate rows (will keep first {args.limit} that produce results)")

    cache = load_cache()
    rows = []
    cache_hits = 0
    api_calls = 0
    skipped_empty = 0
    skipped_dupe = 0
    seen_questions = set()

    pbar = tqdm(total=args.limit, desc="Building examples")
    for ex in qa_ds:
        if len(rows) >= args.limit:
            break

        question = (ex.get("question") or "").strip()
        answer = (ex.get("answer") or "").strip()
        if not question:
            continue
        # When loading from --questions-file, answer is empty and we'll derive it
        # from Tavily below. When loading from qa_data, we require ground-truth answer.
        if not answer and not args.questions_file:
            continue
        if question in seen_questions:
            skipped_dupe += 1
            continue
        seen_questions.add(question)

        if question in cache:
            response = cache[question]
            cache_hits += 1
        else:
            response = tavily_search_with_retry(client, question, args.search_depth)
            api_calls += 1
            if response is not None:
                append_cache(question, response)
                cache[question] = response
            time.sleep(args.sleep)

        if not response or not response.get("results"):
            skipped_empty += 1
            continue

        # If we don't have a ground-truth answer (questions-file mode), derive one.
        if not answer:
            answer = derive_answer(response, args.answer_chars)
            if not answer:
                skipped_empty += 1
                continue

        tool_response_json = format_tool_response(response, args.top_k, args.content_chars)
        tool_call_json = json.dumps(
            {"name": "web_search", "arguments": {"query": question}},
            ensure_ascii=False,
        )

        rows.append({
            "tools_json": TOOLS_JSON,
            "question": question,
            "tool_call_json": tool_call_json,
            "tool_response_json": tool_response_json,
            "answer": answer,
        })
        pbar.update(1)
    pbar.close()

    print(
        f"\nDone: kept={len(rows)}, cache_hits={cache_hits}, api_calls={api_calls}, "
        f"skipped_empty={skipped_empty}, skipped_dupe={skipped_dupe}"
    )

    if not rows:
        sys.exit("No rows produced — check Tavily key and quota.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_list(rows)
    ds.save_to_disk(str(OUTPUT_DIR))
    print(f"Saved {len(ds)} rows to {OUTPUT_DIR}")
    print("\nExample row preview:")
    sample = ds[0]
    for k, v in sample.items():
        preview = v if len(v) < 200 else v[:200] + "..."
        print(f"  {k}: {preview}")


if __name__ == "__main__":
    main()
