# brahmaanu_llm/app/infer.py
from __future__ import annotations
import os
from typing import Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from configs.app_config import AppCfg  # your config module
from configs.sft_config import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN, UNK_TOKEN, MODEL_MAX_LENGTH, MODEL_PADDING_SIDE

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def init_infer(cfg: AppCfg, mode : str = "SFT") -> Tuple[AutoTokenizer, Dict[str, AutoModelForCausalLM]]:
    """
    Load tokenizer + models once at startup.

    Returns:
        tok: shared tokenizer
        models: {"BASE": base_model, "SFT": sft_model}
    """
    base_id = cfg.model.base_id
    dtype = _to_dtype(cfg.model.torch_dtype)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    tok.pad_token = PAD_TOKEN
    tok.eos_token = EOS_TOKEN
    tok.bos_token = BOS_TOKEN
    tok.unk_token = UNK_TOKEN
    tok.padding_side = MODEL_PADDING_SIDE
    
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "</s>"
    if tok.eos_token is None:
        tok.eos_token = tok.sep_token or tok.pad_token
    ## Output
    models_dict = {}
    
    # BASE model
    if "BASE" in mode:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_id,
            torch_dtype=dtype,
            device_map=cfg.model.device_map,
            attn_implementation="sdpa",
        ).eval()
        models_dict["BASE"] = base_model

    # SFT model (prefer merged for simplicity & speed)
    if cfg.model.use_merged and "SFT" in mode:
        sft_repo = "/".join(cfg.model.merged_repo.split("/")[:2])
        subfolder = cfg.model.merged_repo.split("/")[-1]
        sft_model = AutoModelForCausalLM.from_pretrained(
            sft_repo,
            subfolder = subfolder ,
            torch_dtype=dtype,
            device_map=cfg.model.device_map,
            attn_implementation="sdpa",
        ).eval()
        models_dict["SFT"] = sft_model
    elif not cfg.model.use_merged and "SFT" in mode:
        # If you ever need runtime LoRA, enable PEFT path here.
        raise NotImplementedError("Set model.use_merged: true in config for MVP.")
    
    
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
