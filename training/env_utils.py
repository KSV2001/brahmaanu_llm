import os, json, time, random, subprocess
from typing import Any, Dict
import numpy as np
import torch
import platform, importlib

from configs.sft_config import SEED


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def is_dist() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def is_rank0() -> bool:
    return (not is_dist()) or torch.distributed.get_rank() == 0


def setup_env(gpus: int) -> Dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(max(1, gpus)))
    visible = os.environ["CUDA_VISIBLE_DEVICES"]
    world_size = len(visible.split(","))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    return {"WORLD_SIZE": world_size, "VISIBLE": visible}


def env_snapshot(path: str):
    
    info = {
    "python": platform.python_version(),
    "driver": subprocess.getoutput("nvidia-smi --query-gpu=driver_version --format=csv,noheader") or None,
    "gpus": subprocess.getoutput("nvidia-smi --query-gpu=name --format=csv,noheader").splitlines()
    if torch.cuda.is_available() else [],
    "torch": torch.__version__, "cuda_build": getattr(torch.version, "cuda", None),
    "transformers": importlib.import_module("transformers").__version__,
    "trl": importlib.import_module("trl").__version__,
    "accelerate": importlib.import_module("accelerate").__version__,
    "peft": importlib.import_module("peft").__version__,
    "bitsandbytes": importlib.import_module("bitsandbytes").__version__,
    "xformers": getattr(importlib.import_module("xformers"), "__version__", None),
    "NCCL_env": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
    }
    with open(path, "w") as f:
    json.dump(info, f, indent=2)