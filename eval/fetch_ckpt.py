# brahmaanu_llm/eval/fetch_ckpt.py

## To download myu checkpoints of LORA/QLoRA I uploaded to HF Hub
## I've used here LoRA but you can change the path and adapt the code to use QLoRA

import os
import json
import argparse
from typing import Optional, Tuple, List

from huggingface_hub import snapshot_download, hf_hub_url, list_repo_files
from configs.eval_config import (
    LORA_REPO_ID, LORA_SUBFOLDER, LORA_REVISION, LORA_LOCAL_DIR
)

ADAPTER_FILES_CANDIDATES: List[str] = [
    "adapter_model.safetensors",
    "pytorch_lora_weights.safetensors",
    "adapter_model.bin",  # rare fallback
]
ADAPTER_CONFIG_CANDIDATES: List[str] = [
    "adapter_config.json",
    "config.json",  # some tools export this name
]

def _has_required_files(root: str) -> bool:
    files = set()
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            files.add(os.path.relpath(os.path.join(dirpath, f), root))
    has_weights = any(any(p.endswith(cand) or p == cand for cand in ADAPTER_FILES_CANDIDATES) for p in files)
    has_cfg = any(any(p.endswith(cand) or p == cand for cand in ADAPTER_CONFIG_CANDIDATES) for p in files)
    return has_weights and has_cfg

def _locate_subdir_with_adapter(root: str) -> Optional[str]:
    # Accept root or any immediate subdir that contains adapter files
    if _has_required_files(root):
        return root
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p) and _has_required_files(p):
            return p
    return None

def ensure_lora_dir(
    repo_id: str,
    local_dir: str,
    revision: Optional[str] = None,
    subfolder: Optional[str] = "",
    force: bool = False,
    local_dir_use_symlinks: bool = False,
) -> str:
    """
    Download a LoRA PEFT adapter repo to local_dir if needed.
    Returns a path suitable for PeftModel.from_pretrained(..., LORA_DIR).
    """
    os.makedirs(local_dir, exist_ok=True)

    if not force:
        found = _locate_subdir_with_adapter(local_dir)
        if found:
            return found

    # Download
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=local_dir,
        local_dir_use_symlinks=local_dir_use_symlinks,
        allow_patterns=None if not subfolder else f"{subfolder}/**",
        ignore_patterns=None,
    )

    # If a subfolder is specified, prefer that
    candidate_root = os.path.join(local_dir, subfolder) if subfolder else local_dir
    resolved = _locate_subdir_with_adapter(candidate_root)
    if not resolved:
        # Last attempt: scan entire local_dir
        resolved = _locate_subdir_with_adapter(local_dir)

    if not resolved:
        raise FileNotFoundError(
            f"Downloaded repo '{repo_id}' but did not find adapter weights + config under '{local_dir}'. "
            f"Expected one of {ADAPTER_FILES_CANDIDATES} and one of {ADAPTER_CONFIG_CANDIDATES}."
        )
    return resolved

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="HF Hub repo id for the LoRA adapter")
    ap.add_argument("--local-dir", required=True, help="Local directory to store the snapshot")
    ap.add_argument("--revision", default=None, help="Git revision or tag on Hub")
    ap.add_argument("--subfolder", default="", help="Optional subfolder within the repo")
    ap.add_argument("--force", type=int, default=0, help="Force re-download even if files exist")
    ap.add_argument("--no-symlinks", action="store_true", help="Disable symlinks for snapshot")
    args = ap.parse_args()

    path = ensure_lora_dir(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        revision=args.revision,
        subfolder=args.subfolder,
        force=bool(args.force),
        local_dir_use_symlinks=not args.no_symlinks,
    )
    # Print for shell usage
    print(path)

if __name__ == "__main__":
    main()

    # Usage (CLI):
    #   python -m brahmaanu_llm.eval.fetch_ckpt \
    #     --repo-id your-org/your-lora-repo \
    #     --local-dir ./checkpoints/lora-mistral \
    #     --revision main \
    #     --subfolder "" \
    #     --force 0
    #
    # Import use:
    # from brahmaanu_llm.eval.fetch_ckpt import ensure_lora_dir
    
    # LORA_DIR = ensure_lora_dir(
    #     repo_id=LORA_REPO_ID,
    #     local_dir=LORA_LOCAL_DIR,
    #     revision=LORA_REVISION,
    #     subfolder=LORA_SUBFOLDER,
    #     force=False,
    #     local_dir_use_symlinks=False,
    # )
    


