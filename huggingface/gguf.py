## Just want to show how I converted my best checkpoint (LoRA) to GGUF

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, os
from configs.sft_config import *




# load base in fp16 on CPU (safe on small RAM/GPU); use device_map="auto" if you have GPU room

ADAPTER_DIR = ...    # <-- your folder adapter path, contains adapter_model.safetensors. In my case THis is on HF Hub already at Srikasi/bro-sft/lora-zero3-4gpu/last
GGUF_DIR = "merged-hf" 


os.makedirs(GGUF_DIR, exist_ok=True)
# optional hard caps per GPU (uncomment if OOM; set to your VRAM)
# max_mem = {0: "12GiB", 1: "12GiB"}  # or {"cuda:0":"12GiB","cuda:1":"12GiB"}

## Get the actual LLM tokenizer
tok = AutoTokenizer.from_pretrained(BRAHMAANU_LLM_ID, use_fast=True)
tok = PreTrainedTokenizerFast(tokenizer_file=TOK_JSON) ## Tokenizer assumed to be cached in TOK_JSON folder. Check the sft_configs.py to change.

## Standard settings, from sft_configs.py to get the tokenizer to generate the prompt 
tok.pad_token = PAD_TOKEN
tok.eos_token = EOS_TOKEN
tok.bos_token = BOS_TOKEN
tok.unk_token = UNK_TOKEN
tok.model_max_length = MODEL_MAX_LENGTH

## Load base model first, generate responses on all the eval data, Just get the responses.
base = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto",                   # let FSDP place
    attn_implementation="sdpa",
    local_files_only=True,
)

# attach LoRA and merge into base weights
model = PeftModel.from_pretrained(base, ADAPTER_DIR, is_trainable=False)
model = model.merge_and_unload()   # applies LoRA deltas into base

# save merged HF model + tokenizer
model.save_pretrained(OUT_DIR, safe_serialization=True)

tok.save_pretrained(GGUF_DIR) ## Then export this to HF Hub

#print("Merged model saved to:", GGUF_DIR)