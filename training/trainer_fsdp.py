import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from brahmaanu_llm.configs.sft_config import (
MODEL_DIR, MAX_SEQ_LEN, GPUS, FSDP_STRATEGY, FSDP_MIN_PARAMS, FSDP_CPU_OFFLOAD,
PAD_TOKEN, EOS_TOKEN, BOS_TOKEN, UNK_TOKEN, MODEL_MAX_LENGTH, MODEL_PADDING_SIDE,
LORA_RANK , LORA_ALPHA , LORA_DROPOUT, LORA_TARGET_MODULES , LORA_BIAS, LORA_TASK_TYPE, 
FORWARD_PREFETCH, LIMIT_ALL_GATHERS, SYNC_MODULE_STATES, USE_ORIG_PARAMS, ACTIVATION_CHECKPOINTING,
)


def build_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True, local_files_only=True)
    tok.pad_token = PAD_TOKEN
    tok.eos_token = EOS_TOKEN
    tok.bos_token = BOS_TOKEN
    tok.unk_token = UNK_TOKEN
    tok.model_max_length = MODEL_MAX_LENGTH
    tok.padding_side = MODEL_PADDING_SIDE
    return tok

def build_model():
    use_multi = GPUS > 1
    device_map = "auto" if GPUS == 1 else None
    if not use_multi:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, quantization_config=bnb, device_map=device_map,
            attn_implementation="sdpa", local_files_only=True,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, torch_dtype=torch.float16, device_map=device_map,
            attn_implementation="sdpa", local_files_only=True,
            )
    model.gradient_checkpointing_enable()
    return model

def lora_config() -> LoraConfig:
    return LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules = LORA_TARGET_MODULES,
        bias=LORA_BIAS, task_type=LORA_TASK_TYPE,
        )


def fsdp_args():
    if GPUS <= 1:## QLoRA for 1 GPU, LoRA + ZeRO-3 (via FSDP) for 2, 3, 4 GPUs
        return None, None
    fsdp = FSDP_STRATEGY
    fsdp_config = {
        "min_num_params": int(FSDP_MIN_PARAMS),
        "cpu_offload": bool(FSDP_CPU_OFFLOAD),
        "forward_prefetch": FORWARD_PREFETCH,
        "limit_all_gathers": LIMIT_ALL_GATHERS,
        "sync_module_states": SYNC_MODULE_STATES,
        "use_orig_params": USE_ORIG_PARAMS,
        "activation_checkpointing": ACTIVATION_CHECKPOINTING,
    }
    return fsdp, fsdp_config