# brahmaanu_llm/app/infer.py
from __future__ import annotations
import os
from typing import Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from configs.app_config import AppCfg  # your config module
from configs.sft_config import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN, UNK_TOKEN, MODEL_MAX_LENGTH, MODEL_PADDING_SIDE

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def resolve_cache_dir(cfg: dict) -> str:
    return (
        os.getenv("HF_HUB_CACHE")
        or os.getenv("TRANSFORMERS_CACHE")
        or cfg.get("cache_dir")
        or "./hf_cache"
    )


def init_infer(cfg: AppCfg, mode: str = "SFT") -> Tuple[AutoTokenizer, Dict[str, AutoModelForCausalLM]]:
    """
    Load tokenizer + models once at startup.

    Returns:
        tok: shared tokenizer
        models: {"BASE": base_model, "SFT": sft_model}
    """
    base_id = cfg.model.base_id
    dtype = _to_dtype(cfg.model.torch_dtype)
    CACHE = resolve_cache_dir(cfg)
    print(f"Cache dir : {CACHE}")

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    tok.pad_token = PAD_TOKEN or tok.eos_token or "</s>"
    tok.eos_token = EOS_TOKEN or tok.eos_token or tok.pad_token
    tok.bos_token = BOS_TOKEN or tok.bos_token
    tok.unk_token = UNK_TOKEN or tok.unk_token
    tok.padding_side = MODEL_PADDING_SIDE

    models_dict = {}

    # ---- Load BASE ----
    if "BASE" in mode or "SFT" in mode:
        print("Loading the BASE model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_id,
            torch_dtype=dtype,
            device_map=cfg.model.device_map,
            cache_dir=CACHE,
            attn_implementation="sdpa",
            token=os.getenv("HF_TOKEN"),
        ).eval()
        models_dict["BASE"] = base_model
        print("Loaded the BASE model")

    # ---- Load SFT via LoRA ----
    if "SFT" in mode:
        lora_dir = getattr(cfg.model, "lora_dir", None)
        if not lora_dir:
            raise ValueError("cfg.model.lora_dir must be set for SFT mode")

        print(f"Loading LoRA adapter from: {lora_dir}")
        # If remote HF path with subfolder
        if "/" in lora_dir:
            parts = lora_dir.strip("/").split("/")
            repo_id = "/".join(parts[:2])
            subfolder = "/".join(parts[2:]) if len(parts) > 2 else None
            sft_model = PeftModel.from_pretrained(
                base_model,
                repo_id,
                subfolder=subfolder,
                torch_dtype=dtype,
                token=os.getenv("HF_TOKEN"),
            ).eval()
        else:
            # Local dir
            sft_model = PeftModel.from_pretrained(
                base_model,
                lora_dir,
                torch_dtype=dtype,
            ).eval()

        print("Loaded SFT (LoRA) model")
        models_dict["SFT"] = sft_model

    print("Returning tokenizer and models_dict")
    return tok, models_dict



def generate_text(
    models: Dict[str, AutoModelForCausalLM],
    tok: AutoTokenizer,
    prompt: str,
    mode: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    timeout_s: int = 12,
) -> str:
    """
    Run a single completion with a soft timeout (max_time).
    mode: one of {"SFT_RAG","SFT","BASE_RAG","BASE"} → selects SFT or BASE weights.
    Returns raw decoded text (model output only).
    """
    model = _select_model(models, mode)
    
    # compute a safe context length
    ctx_max = _safe_ctx_max(tok, model, fallback=2048)   # <- pass model
    max_new = int(max_new_tokens)
    prompt_budget = max(16, ctx_max - max_new - 8)       # leave headroom
    
    # tokenize and cap prompt explicitly; do NOT pass a huge max_length to HF
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"][:, -prompt_budget:]
    attn_mask = enc["attention_mask"][:, -prompt_budget:] if "attention_mask" in enc else None
    
    inputs = {"input_ids": input_ids.to(model.device)}
    if attn_mask is not None:
        inputs["attention_mask"] = attn_mask.to(model.device)
    
    do_sample = bool(temperature and temperature > 1e-6)
    
    gen_kwargs = dict(
        max_new_tokens=max_new,
        do_sample=do_sample,
        top_p=1.0,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        use_cache=True,
        max_time=max(1.0, float(timeout_s) * 0.9),
    )
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)   # only set when sampling
    
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    
    new_ids = out[0, inputs["input_ids"].shape[1]:]
    text = tok.decode(new_ids, skip_special_tokens=True)
    return text


def count_tokens(text: str, tok: AutoTokenizer | None = None) -> int:
    """Quick token estimate for budgeting."""
    if tok is None:
        return max(1, len((text or "").strip()) // 4)
    return len(tok.encode(text, add_special_tokens=False))


# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------

def _select_model(models: Dict[str, AutoModelForCausalLM], mode: str) -> AutoModelForCausalLM:
    return models["SFT"] if mode in ("SFT_RAG", "SFT") else models["BASE"]

def _safe_ctx_max(tok : AutoTokenizer, model, fallback=2048) -> int :
    x = getattr(tok, "model_max_length", None)
    if x is None or x == float("inf") or (isinstance(x, int) and x > 1_000_000):
        y = getattr(getattr(model, "config", None), "max_position_embeddings", None)
        if isinstance(y, int) and 0 < y <= 65536:
            return y
        return fallback
    return int(x)

def _to_dtype(name: str):
    name = (name or "float16").lower()
    if name in ("fp16", "float16", "half"): return torch.float16
    if name in ("bf16", "bfloat16"):        return torch.bfloat16
    return torch.float16
