import math, psutil
from trl import SFTTrainer, SFTConfig
from brahmaanu_llm.configs.sft_config import (
OUT_DIR, NUM_EPOCHS, BATCH_PER_DEVICE, TARGET_GLOBAL_BATCH, LR, WARMUP_FRAC,
WEIGHT_DECAY, LOG_STEPS, SAVE_STEPS, GPUS, MAX_SEQ_LEN,
LR_SCHEDULER_TYPE, COMPLETION_ONLY_LOSS, REPORT_TO, DATALOADER_PIN_MEMORY, MAX_DATALOADER_WORKERS, PACKING, MULTI_GPU_OPTIM, MULTI_GPU_OPTIM_ARGS, MULTI_GPU_FP16, MULTI_GPU_BF16, MULTI_GPU_GRADIENT_CHECKPOINTING, USE_BF16, SINGLE_GPU_FP16, SINGLE_GPU_BF16, SINGLE_GPU_GRADIENT_CHECKPOINTING,
)
from brahmaanu_llm.training.callbacks import InfraLogger


def build_train_config(world_size: int, multi_gpu: bool) -> SFTConfig:
    denom = BATCH_PER_DEVICE * max(1, world_size)
    grad_accum = math.ceil(TARGET_GLOBAL_BATCH / denom) if TARGET_GLOBAL_BATCH % denom != 0 else TARGET_GLOBAL_BATCH // denom
    
    if multi_gpu:
        return SFTConfig(
            output_dir=OUT_DIR,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_PER_DEVICE,
            gradient_accumulation_steps=grad_accum,
            learning_rate=LR,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            completion_only_loss=COMPLETION_ONLY_LOSS,
            warmup_ratio=WARMUP_FRAC,
            weight_decay=WEIGHT_DECAY,
            logging_steps=LOG_STEPS,
            save_steps=SAVE_STEPS,
            fp16=MULTI_GPU_FP16,
            bf16=MULTI_GPU_BF16,
            gradient_checkpointing=MULTI_GPU_GRADIENT_CHECKPOINTING,
            report_to=REPORT_TO,
            dataloader_pin_memory=DATALOADER_PIN_MEMORY,
            dataloader_num_workers=min(MAX_DATALOADER_WORKERS, psutil.cpu_count(logical=True) or MIN_DATALOADER_WORKERS),
            optim=MULTI_GPU_OPTIM,
            optim_args=MULTI_GPU_OPTIM_ARGS,
            packing=PACKING,
        )
    else:
        return SFTConfig(
            output_dir=OUT_DIR,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_PER_DEVICE,
            gradient_accumulation_steps=grad_accum,
            learning_rate=LR,
            completion_only_loss=COMPLETION_ONLY_LOSS,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            warmup_ratio=WARMUP_FRAC,
            weight_decay=WEIGHT_DECAY,
            logging_steps=LOG_STEPS,
            save_steps=SAVE_STEPS,
            fp16=SINGLE_GPU_FP16,
            bf16=SINGLE_GPU_BF16,
            gradient_checkpointing=SINGLE_GPU_GRADIENT_CHECKPOINTING,
            report_to=REPORT_TO,
            dataloader_pin_memory=DATALOADER_PIN_MEMORY,
            dataloader_num_workers=min(MAX_DATALOADER_WORKERS, psutil.cpu_count(logical=True) or MIN_DATALOADER_WORKERS),
            packing=PACKING,
        )




def assemble_trainer(model, tok, train_cfg: SFTConfig, ds_train, ds_eval, peft_cfg, world_size: int, log_csv: str, gpu_dmon: str) -> SFTTrainer:
    cb = [InfraLogger(log_csv, gpu_dmon, world_size, tok)]
    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        args=train_cfg,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        peft_config=peft_cfg,
        callbacks=cb,
    )
    return trainer