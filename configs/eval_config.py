
from pathlib import Path
from brahmaanu_llm.configs.sft_config import *
import os


F_ROOT = Path(__file__).resolve().parents[1]

## ------------------------------------------------------------------------------------------------
## DATA SCHEMA
ALLOWED_STATUS = {"OK", "UNKNOWN", "PARTIAL", "CORRECTED"}

## OUTPUTS

STEM_FULL = Path(os.getenv("STEM_FULL", F_ROOT / "eval" / "outputs" / "df_eval_full_final.parquet")).resolve()       # expected parquet with base + sft columns
STEM_RAG  = Path(os.getenv("STEM_RAG", F_ROOT / "eval" / "outputs" / "rag_df_eval_full_final.parquet")).resolve()       # expected parquet with rag columns
EVAL_OUT_DIR = Path(os.getenv("OUT_EVAL_DIR", F_ROOT / "eval" / "outputs")).resolve()  
METRICS_SUMMARY_JSON = os.path.join(str(OUT_EVAL_DIR), "metrics_summary.json")

# Public LoRA on HF Hub
LORA_REPO_ID   = "Srikasi/bro-sft"
LORA_SUBFOLDER = "lora-zero3-4gpu/last"   # path inside the repo
LORA_REVISION  = "main"                   # tag/branch/commit; keep None for default
LORA_LOCAL_DIR = "./checkpoints/lora"     # where to cache locally

# Optional caches (uncomment if you want)
# os.environ["HF_HOME"] = "./.hf_cache"
# os.environ["HF_HUB_CACHE"] = "./.hf_cache"
# os.environ["TRANSFORMERS_CACHE"] = "./.hf_cache"