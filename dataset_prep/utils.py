import os
import re
import torch
from typing import List
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


NLLB_SRC_LANG = "eng_Latn"
NLLB_TGT_LANG = "khk_Cyrl"

CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")


class NLLBTranslator:
    def __init__(self, model_name: str = "facebook/nllb-200-3.3B",
                 device: str = "cuda", dtype=torch.float16,
                 num_beams: int = 2, max_input_tokens: int = 512,
                 max_new_tokens: int = 512):
        self.device = device
        self.num_beams = num_beams
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=NLLB_SRC_LANG)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, dtype=dtype
        ).to(device).eval()

        self.forced_bos = self.tokenizer.convert_tokens_to_ids(NLLB_TGT_LANG)

    @torch.inference_mode()
    def translate_batch(self, texts: List[str]) -> List[str]:
        clean = [t if isinstance(t, str) and t.strip() else "" for t in texts]
        enc = self.tokenizer(
            clean, return_tensors="pt", padding=True,
            truncation=True, max_length=self.max_input_tokens,
        ).to(self.device)

        out = self.model.generate(
            **enc,
            forced_bos_token_id=self.forced_bos,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
        )
        decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        return [d if src else "" for d, src in zip(decoded, clean)]


def cyrillic_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if CYRILLIC_RE.match(c)) / len(letters)


def row_is_valid(row: dict, min_cyr: float = 0.5,
                 min_len: int = 3, max_len_ratio: float = 5.0) -> bool:
    for k in ("prompt", "chosen", "rejected"):
        v = row.get(k, "")
        if not isinstance(v, str) or len(v.strip()) < min_len:
            return False
        if cyrillic_ratio(v) < min_cyr:
            return False

    if row["chosen"].strip() == row["rejected"].strip():
        return False

    lens = [len(row["chosen"]), len(row["rejected"])]
    if min(lens) == 0 or max(lens) / min(lens) > max_len_ratio:
        return False
    return True


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cache_path(*parts: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", *parts)
