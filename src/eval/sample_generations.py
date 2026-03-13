from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.io import read_yaml
from src.utils.logging import get_logger

LOGGER = get_logger("sample_generations")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample Mongolian outputs.")
    parser.add_argument("--model-config", default="configs/model/qwen2b-continued-pretrain.yaml")
    parser.add_argument("--train-config", default="configs/train/train_base.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--prompt",
        default="Монгол хэл дээр богино тайлбар бичнэ үү:",
        help="Prompt text.",
    )
    args = parser.parse_args()

    model_cfg = read_yaml(args.model_config)
    train_cfg = read_yaml(args.train_config)
    model_name = args.checkpoint or train_cfg["output_dir"] or model_cfg["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=bool(model_cfg.get("trust_remote_code", True)))
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=bool(model_cfg.get("trust_remote_code", True)))
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    LOGGER.info("Prompt: %s", args.prompt)
    LOGGER.info("Generation: %s", text)


if __name__ == "__main__":
    main()
