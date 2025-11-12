# brahmaanu_llm/app/infer.py
from __future__ import annotations
import os, glob, json
import logging
from typing import Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging
from peft import PeftModel

from configs.app_config import AppCfg
from configs.sft_config import (
    PAD_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
    UNK_TOKEN,
    MODEL_MAX_LENGTH,
    MODEL_PADDING_SIDE,
)

# ---------------------------------------------------------------------
# global logging setup (so container logs show HF activity)
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
hf_logging.set_verbosity_info()
hf_logging.enable_default_handler()
hf_logging.enable_explicit_format()

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def resolve_cache_dir(cfg) -> str:
    return (
        os.getenv("HF_HUB_CACHE")
        or os.getenv("TRANSFORMERS_CACHE")
        or getattr(cfg, "cache_dir", None)
        or "./hf_cache"
    )


    

def _to_dtype(name: str):
    name = (name or "float16").lower()
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float16


def _env_debug():
    print("[infer] ===== ENV DEBUG =====")
    for k in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_TOKEN"]:
        print(f"[infer] {k} = {os.getenv(k)}")
    keys = ["BASE_ID","BASE_MODEL_PATH","HF_HOME","TRANSFORMERS_CACHE",
            "HF_HUB_OFFLINE","HF_TOKEN","CUDA_VISIBLE_DEVICES"]
    print("[debug] env:", {k: os.getenv(k, "NOT SET") for k in keys})
    root = os.getenv("TRANSFORMERS_CACHE", "/workspace/hf")
    print("[debug] cache root exists:", os.path.isdir(root))
    print("[debug] top dirs:", [p for p in glob.glob(f"{root}/*")][:20])
    print("[debug] models--*/snapshots:", [p for p in glob.glob(f"{root}/models--*/*/snapshots/*")][:20])
    print("[infer] ======================")


def _dump_hf_repo(cache_dir: str, repo_id: str):
    """
    Print local HF cache layout for a repo like 'Srikasi/bro-sft'
    and return ([(snap_id, snap_path), ...], current_ref_hash_or_None).
    """
    owner, name = repo_id.split("/", 1)
    repo_root = os.path.join(cache_dir, f"models--{owner}--{name}")
    print(f"[infer] HF repo root: {repo_root}")
    if not os.path.isdir(repo_root):
        print("[infer] HF repo root not found locally")
        return [], None

    snaps_dir = os.path.join(repo_root, "snapshots")
    refs_dir = os.path.join(repo_root, "refs")

    snaps = []
    if os.path.isdir(snaps_dir):
        for s in os.listdir(snaps_dir):
            full = os.path.join(snaps_dir, s)
            print(f"[infer]  snapshot: {s}")
            snaps.append((s, full))
            try:
                inner = os.listdir(full)
                print(f"[infer]    contents: {inner}")
            except Exception:
                pass

    current_ref = None
    if os.path.isdir(refs_dir):
        main_ref = os.path.join(refs_dir, "main")
        if os.path.isfile(main_ref):
            with open(main_ref, "r") as f:
                current_ref = f.read().strip()
            print(f"[infer] refs/main -> {current_ref}")

    return snaps, current_ref


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def init_infer(cfg: AppCfg, mode: str = "SFT") -> Tuple[AutoTokenizer, Dict[str, AutoModelForCausalLM]]:
    print("[infer] init_infer() called")
    _env_debug()

    base_id = cfg.model.base_id
    dtype = _to_dtype(cfg.model.torch_dtype)
    cache = resolve_cache_dir(cfg)

    print(f"[infer] base_id        : {base_id}")
    print(f"[infer] dtype          : {dtype}")
    print(f"[infer] device_map     : {cfg.model.device_map}")
    print(f"[infer] cache_dir      : {cache}")
    print(f"[infer] requested mode : {mode}")

    # 1) tokenizer
    print("[infer] loading tokenizer ...")
    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    tok.pad_token = PAD_TOKEN or tok.eos_token or "</s>"
    tok.eos_token = EOS_TOKEN or tok.eos_token or tok.pad_token
    tok.bos_token = BOS_TOKEN or tok.bos_token
    tok.unk_token = UNK_TOKEN or tok.unk_token
    tok.padding_side = MODEL_PADDING_SIDE
    tok.model_max_length = MODEL_MAX_LENGTH
    print(f"[infer] tokenizer loaded, model_max_length={tok.model_max_length}")

    models: Dict[str, AutoModelForCausalLM] = {}

    # 2) BASE MODEL
    print("[infer] ----- BASE LOAD START -----")
    try:
        print("[infer] calling AutoModelForCausalLM.from_pretrained(...)")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_id,
            torch_dtype=dtype,
            device_map=cfg.model.device_map,   # e.g. "cuda:0"
            cache_dir=cache,
            local_files_only=True,             # stay inside /workspace/hf
            token=os.getenv("HF_TOKEN"),
        )
        print("[infer] from_pretrained returned")
        base_model = base_model.eval()
        models["BASE"] = base_model
        print("[infer] ----- BASE LOAD OK -----")
    except Exception as e:
        print(f"[infer] ***** BASE LOAD FAILED *****: {type(e).__name__}: {e}")
        return tok, models

    # 3) LoRA MODEL
    if "SFT" in mode and "BASE" in models:
        lora_dir = getattr(cfg.model, "lora_dir", None)
        print(f"[infer] SFT requested, lora_dir={lora_dir}")
        if lora_dir:
            # expected: "Srikasi/bro-sft/lora-zero3-4gpu/last"
            parts = lora_dir.strip("/").split("/")
            repo_id = "/".join(parts[:2])       # "Srikasi/bro-sft"
            subfolder = "/".join(parts[2:])     # "lora-zero3-4gpu/last"
            print(f"[infer] repo_id={repo_id}, subfolder={subfolder}")

            snaps, current_ref = _dump_hf_repo(cache, repo_id)

            chosen_revision = None
            for snap_id, snap_path in snaps:
                maybe = os.path.join(snap_path, subfolder)
                if os.path.isdir(maybe):
                    chosen_revision = snap_id
                    print(f"[infer] found LoRA in snapshot {snap_id} at {maybe}")
                    break

            print(f"[infer] chosen_revision={chosen_revision} (refs/main={current_ref})")
            print("[infer] ----- LORA LOAD START -----")

            # load a SECOND base and apply LoRA on that
            sft_base = AutoModelForCausalLM.from_pretrained(
                base_id,
                torch_dtype=dtype,
                device_map=cfg.model.device_map,
                cache_dir=cache,
                local_files_only=True,
                token=os.getenv("HF_TOKEN"),
            ).eval()
            try:
                if chosen_revision:
                    print("[infer] ----- LORA LOAD START FROM CACHE -----")
                    sft_model = PeftModel.from_pretrained(
                        sft_base,
                        repo_id,
                        subfolder=subfolder,
                        revision=chosen_revision,
                        cache_dir=cache,
                        local_files_only=True,
                        token=os.getenv("HF_TOKEN"),
                    )
                    
                else:
                    print("[infer] ----- LORA LOAD START FROM HF HUB -----")
                    sft_model = PeftModel.from_pretrained(
                        sft_base,
                        repo_id,
                        subfolder=subfolder,
                        cache_dir=cache,
                        token=os.getenv("HF_TOKEN"),
                    )
                print("[infer] PeftModel.from_pretrained returned")
                sft_model = sft_model.eval()
                models["SFT"] = sft_model
                print("[infer] ----- LORA LOAD OK -----")
            except Exception as e:
                print(f"[infer] ***** LORA LOAD FAILED *****: {type(e).__name__}: {e}")
        else:
            print("[infer] SFT requested but cfg.model.lora_dir is empty; skipping.")

    print("[infer] returning tokenizer + models")
    return tok, models



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
