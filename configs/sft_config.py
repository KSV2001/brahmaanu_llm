## Contains all the constants related to generation and training process of SFT
from pathlib import Path
import os
from typing import Optional, List
import torch

# /path/to/repo/f
F_ROOT = Path(__file__).resolve().parents[1]

## --------------------------------------------------------------------------------------------------
## Random seeding everything for reproducibility (as much as possible)
SEED = 42

## --------------------------------------------------------------------------------------------------
## DATA PREP
BRAHMAANU_LLM_ID = "mistralai/Mistral-7B-Instruct-v0.3"
TOK_JSON = "./mistral-cache/tokenizer.json"  # loading from mistral-cache, the default when you downlaod Mistral-7B. 

PAD_TOKEN = "</s>"
EOS_TOKEN = "</s>"
BOS_TOKEN = "<s>"
UNK_TOKEN = "<unk>"
MODEL_MAX_LENGTH = 1536
MODEL_PADDING_SIDE = "left"

SCHEMA_NOTE = (
"You are a helpful chat assistant for the employees of Brahmaanu Space Observatory. "
"They can query you information about different aspects of the observatory, which are found in the documents."
"Output ONLY a compact JSON object with keys exactly: "
"answer (string, a brief and complete answer to the question), citations (a python List of strings), status (a string, exactly one of [OK,PARTIAL,UNKNOWN,CORRECTED]). "
"Be professional and formal. Prefer the style of the answers in the dataset. "
"If you don't know the correct response then acknowledge it instead of giving a fake answer."
"If the user discusses another topic, steer them towards asking about the observatory. "
"Do not hallucinate or confabulate. "
)

QA_JSONS_PATH = Path(os.getenv("QA_JSONS_PATH", F_ROOT / "data" / "raw" / "questions")).resolve()

NAME_TO_ID_MAPPER_DICT = {"brahmaanu_ops" : "A", "brahmaanu_instruments" : "B", "brahmaanu_detectors" : "C", "brahmaanu_calibration" : "D",
                  "brahmaanu_constraints" : "E", "brahmaanu_data_pipelines" : "F", "brahmaanu_admin" :"G", "brahmaanu_discoveries" : "H"}

TRAIN_SFT_PATH =  Path(os.getenv("TRAIN_SFT_PATH", F_ROOT / "data" / "processed" / "df_sft_train.parquet")).resolve()
EVAL_SFT_PATH = Path(os.getenv("TRAIN_SFT_PATH", F_ROOT / "data" / "processed" / "df_sft_eval.parquet")).resolve()
SFT_PATH = Path(os.getenv("SFT_PATH", F_ROOT / "data" / "processed" / "df_sft_splits.parquet")).resolve()


## --------------------------------------------------------------------------------------------------
## SFT configs

# GPUs picked by the notebook/script via env before calling worker()
GPUS = int(os.environ.get("GPUS", "1"))


## MODEL DIR to save the cached LLM (After it is downloaded once)
MODEL_DIR = os.environ.get("MODEL_DIR", "./mistral-cache")

OUT_DIR_BASE = os.environ.get("OUT_DIR", "out-sft")
OUT_DIR = f"{OUT_DIR_BASE}-{GPUS}-gpu"
SEED = int(os.environ.get("SEED", "42"))
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "768"))
BATCH_PER_DEVICE = int(os.environ.get("BATCH_PER_DEVICE", "8"))
TARGET_GLOBAL_BATCH = int(os.environ.get("TARGET_GLOBAL_BATCH", "32"))
LR = float(os.environ.get("LR", "2e-4"))
WARMUP_FRAC = float(os.environ.get("WARMUP_FRAC", "0.05"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.05"))
NUM_EPOCHS = float(os.environ.get("NUM_EPOCHS", "1.0"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "15"))
LOG_STEPS = int(os.environ.get("LOG_STEPS", "5"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "500"))
PROBE_N = int(os.environ.get("PROBE_N", "64"))


TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))
GEN_MAX_NEW_TOKENS = int(os.environ.get("GEN_MAX_NEW_TOKENS", "128"))
TOP_P = float(os.environ.get("TOP_P", "1.0"))
TOP_K = int(os.environ.get("TOP_K", "0"))
NUM_BEAMS = int(os.environ.get("NUM_BEAMS", "1"))

LORA_RANK  = 16
LORA_ALPHA = 32
LORA_DROPOUT =0.05
LORA_TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
LORA_BIAS ="none" 
LORA_TASK_TYPE="CAUSAL_LM"

FORWARD_PREFETCH = False
LIMIT_ALL_GATHERS = True
SYNC_MODULE_STATES = True
USE_ORIG_PARAMS = True
ACTIVATION_CHECKPOINTING = True
FSDP_STRATEGY = os.environ.get("FSDP_STRATEGY", "full_shard")
FSDP_WRAP: Optional[List[str]] = None
FSDP_MIN_PARAMS = int(float(os.environ.get("FSDP_MIN_PARAMS", 1e8)))
FSDP_CPU_OFFLOAD = os.environ.get("FSDP_CPU_OFFLOAD", "0") == "1"

LR_SCHEDULER_TYPE = "cosine"
COMPLETION_ONLY_LOSS = True
REPORT_TO = ["tensorboard"]
DATALOADER_PIN_MEMORY = True
MAX_DATALOADER_WORKERS = 4
PACKING = False

# Multi-GPU specific settings
MULTI_GPU_OPTIM = "adamw_torch"
MULTI_GPU_OPTIM_ARGS = "fused=False,foreach=False"
MULTI_GPU_FP16 = True
MULTI_GPU_BF16 = False
MULTI_GPU_GRADIENT_CHECKPOINTING = False

# Single-GPU settings with bfloat16 check
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
SINGLE_GPU_FP16 = not USE_BF16
SINGLE_GPU_BF16 = USE_BF16
SINGLE_GPU_GRADIENT_CHECKPOINTING = True


PEAK_VRAM_PER_GPU_JSON = os.path.join(OUT_DIR,"vram_peak_mb_per_gpu.json")
LOG_CSV = os.path.join(OUT_DIR, "TRAIN_LOGS.csv")
GPU_DMON = os.path.join(OUT_DIR, "gpu_dmon.csv")

