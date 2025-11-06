# app/config.py
from __future__ import annotations
import os, yaml
from dataclasses import dataclass, field
from typing import List, Optional

# ---------- Dataclasses ----------
@dataclass
class ModelCfg:
    base_id: str = "mistralai/Mistral-7B-Instruct-v0.3"
    merged_repo: str = "Srikasi/bro-sft/merged-hf"     # your merged SFT repo on HF
    use_merged: bool = True                         # prefer merged to avoid PEFT at runtime
    sft_repo: str = "Srikasi/bro-sft"               # only used if use_merged=False
    sft_subfolder: str = "lora-zero3-4gpu/last"
    torch_dtype: str = "float16"                    # float16 | bfloat16 | auto
    device_map: str = "auto"
    max_new_tokens: int = 256
    temperature: float = 0.0
    ctx_max_tokens: int = 2048

@dataclass
class RagCfg:
    docs_folder: str = "brahmaanu_llm/data/raw/docs" 
    chunk_strategy: str = "fact_mode"               # fact_mode | token_mode
    n_units: int = 1
    index_pkl : Optional[str] = "brahmaanu_llm/rag/outputs/rag_index.pkl.gz" 
    top_k : int = 10

@dataclass
class MemoryCfg:
    max_turns: int = 12
    max_session_tokens: int = 1200                  # budget for memory window

@dataclass
class GuardsCfg:
    timeout_s: int = 12
    per_request_max_gbp: float = 0.10
    daily_soft_gbp: float = 1.0
    monthly_gbp: float = 20.0
    gpu_gbp_per_hour: float = 0.80                  # set for your GPU tier; used for estimate

@dataclass
class FallbackCfg:
    order: List[str] = field(default_factory=lambda: ["SFT_RAG","SFT","BASE_RAG","BASE"])

@dataclass
class CacheCfg:
    enabled: bool = True
    ttl_min: int = 1440

@dataclass
class GgufCfg:
    enabled: bool = False                           # default off for this project
    server_url: Optional[str] = None                # e.g., "http://localhost:8081"
    model_path_or_url: Optional[str] = None
    n_threads: int = 8

@dataclass
class LoggingCfg:
    debug: bool = False
    sample_rate: float = 0.1                        # fraction of successful requests to log fully

@dataclass
class AppCfg:
    model: ModelCfg = field(default_factory=ModelCfg)
    rag: RagCfg = field(default_factory=RagCfg)
    memory: MemoryCfg = field(default_factory=MemoryCfg)
    guards: GuardsCfg = field(default_factory=GuardsCfg)
    fallback: FallbackCfg = field(default_factory=FallbackCfg)
    cache: CacheCfg = field(default_factory=CacheCfg)
    gguf: GgufCfg = field(default_factory=GgufCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)

# ---------- Loader ----------
def _merge_dict(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge_dict(dst[k], v)
        else:
            dst[k] = v
    return dst

def load_cfg() -> AppCfg:
    cfg = AppCfg()

    # Load YAML if provided
    path = os.getenv("CONFIG_PATH")
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        # Manual nested merge into dataclasses
        base = {
            "model": cfg.model.__dict__,
            "rag": cfg.rag.__dict__,
            "memory": cfg.memory.__dict__,
            "guards": cfg.guards.__dict__,
            "fallback": {"order": cfg.fallback.order},
            "cache": cfg.cache.__dict__,
            "gguf": cfg.gguf.__dict__,
            "logging": cfg.logging.__dict__,
        }
        merged = _merge_dict(base, y)

        cfg.model = ModelCfg(**merged["model"])
        cfg.rag = RagCfg(**merged["rag"])
        cfg.memory = MemoryCfg(**merged["memory"])
        cfg.guards = GuardsCfg(**merged["guards"])
        cfg.fallback = FallbackCfg(order=merged["fallback"]["order"])
        cfg.cache = CacheCfg(**merged["cache"])
        cfg.gguf = GgufCfg(**merged["gguf"])
        cfg.logging = LoggingCfg(**merged["logging"])

    # Minimal env overrides (optional)
    cfg.model.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", cfg.model.max_new_tokens))
    cfg.guards.timeout_s = int(os.getenv("TIMEOUT_S", cfg.guards.timeout_s))

    return cfg

# ---------- Quick sanity ----------
def print_cfg_summary(cfg: AppCfg):
    print({
        "model": {
            "merged": cfg.model.use_merged,
            "merged_repo": cfg.model.merged_repo,
            "max_new_tokens": cfg.model.max_new_tokens,
            "ctx_max_tokens": cfg.model.ctx_max_tokens,
        },
        "rag": {
            "index_pkl" : cfg.rag.index_pkl,
            "docs_folder": cfg.rag.docs_folder,
            "chunk_strategy": cfg.rag.chunk_strategy,
            "n_units": cfg.rag.n_units,
        },
        "memory": cfg.memory.__dict__,
        "guards": cfg.guards.__dict__,
        "fallback": cfg.fallback.order,
        "cache": cfg.cache.__dict__,
        "gguf_enabled": cfg.gguf.enabled,
        "logging": cfg.logging.__dict__,
    })

if __name__ == "__main__":
    c = load_cfg()
    print_cfg_summary(c)
