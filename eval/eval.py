# brahmaanu_llm/eval/eval.py
# Usage:
#   python -m brahmaanu_llm.eval.eval --modes base sft base_rag sft_rag
# Writes: metrics_summary.json in brahmaanu_llm/eval/outputs - Currently already files exist

import os, json, argparse
import pandas as pd
from importlib.resources import files

from .metrics import validate_json_schema, evaluate_items
from brahmaanu_llm.configs.eval_config import *

PKG_OUTPUTS = "brahmaanu_llm.eval.outputs"


def _outputs_dir() -> str:
    try:
        return str(files(PKG_OUTPUTS))
    except Exception:
        return PKG_OUTPUTS.replace(".", os.sep)

def _resolve_parquet(stem: str) -> str:
    base = _outputs_dir()
    for name in (stem, f"{stem}.parquet", f"{stem}.pq"):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Parquet not found: {stem} under {base}")

def _augment_with_schema(df: pd.DataFrame, pred_col: str, prefix: str) -> pd.DataFrame:
    vals = df[pred_col].astype(str).apply(validate_json_schema).tolist()
    add = pd.DataFrame(vals).add_prefix(f"{prefix}_")
    return pd.concat([df, add], axis=1)

def _build_items(df: pd.DataFrame, pred_col: str, prompt_col: str, target_col: str):
    return [
        {
            "prompt": p,
            "gold": g,
            "pred_text": pred,
        }
        for p, g, pred in zip(df[prompt_col].astype(str), df[target_col].astype(str), df[pred_col].astype(str))
    ]

def run(modes):
    out_dir = _outputs_dir()
    os.makedirs(out_dir, exist_ok=True)

    summary = {}

    # ---------- Non-RAG (base, sft) ----------
    if any(m in modes for m in ("base", "sft")):
        p_full = _resolve_parquet(STEM_FULL) ## Currently already has all the fields
        df_full = pd.read_parquet(p_full)

        # expected columns
        # prompt, target_json, base_model_response, sft_model_response
        if "base" in modes and "base_model_response" in df_full.columns:
            df_full = _augment_with_schema(df_full, "base_model_response", "base")
            items = _build_items(df_full, "base_model_response", "prompt", "target_json")
            summary["base"] = evaluate_items(items)

        if "sft" in modes and "sft_model_response" in df_full.columns:
            df_full = _augment_with_schema(df_full, "sft_model_response", "sft")
            items = _build_items(df_full, "sft_model_response", "prompt", "target_json")
            summary["sft"] = evaluate_items(items)

        # write back augmented parquet
        df_full.to_parquet(os.path.join(out_dir, "df_eval_full_with_schema.parquet"), index=False)

    # ---------- RAG (base_rag, sft_rag) ----------
    if any(m in modes for m in ("base_rag", "sft_rag")):
        p_rag = _resolve_parquet(STEM_RAG)
        df_rag = pd.read_parquet(p_rag)

        # expected columns
        # rag_prompt, target_json, base_rag_response, sft_rag_response
        if "base_rag" in modes and "base_rag_response" in df_rag.columns:
            df_rag = _augment_with_schema(df_rag, "base_rag_response", "base_rag")
            items = _build_items(df_rag, "base_rag_response", "rag_prompt", "target_json")
            summary["base_rag"] = evaluate_items(items)

        if "sft_rag" in modes and "sft_rag_response" in df_rag.columns:
            df_rag = _augment_with_schema(df_rag, "sft_rag_response", "sft_rag")
            items = _build_items(df_rag, "sft_rag_response", "rag_prompt", "target_json")
            summary["sft_rag"] = evaluate_items(items)

        # write back augmented parquet
        df_rag.to_parquet(os.path.join(out_dir, "rag_df_eval_full_with_schema.parquet"), index=False) ## Currently already has all the fields

    # ---------- summary ----------
    with open(METRICS_SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["base", "sft", "base_rag", "sft_rag"],
                    help="Any of: base sft base_rag sft_rag")
    args = ap.parse_args()
    run(set(args.modes))
