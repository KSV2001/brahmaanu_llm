import os, json, uuid
from accelerate import notebook_launcher

from configs.sft_config import * # noqa
from training.env_utils import set_seed, setup_env, env_snapshot, now
from training.data import load_dataset, add_prompt_completion, make_probe
from training.trainer_fsdp import build_tokenizer, build_model, lora_config, fsdp_args
from training.train_setup import build_train_config, assemble_trainer
from training.eval_probe import evaluate_probe



def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    env = setup_env(GPUS)
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", env["WORLD_SIZE"]))
    RUN_ID = os.environ.get("RUN_ID", uuid.uuid4().hex[:8])
    
    set_seed(SEED)
    
    cfg_stamp = {
        "run_id": RUN_ID, "world_size": WORLD_SIZE, "gpus_requested": GPUS,
        "seq_len": MAX_SEQ_LEN, "micro_batch": BATCH_PER_DEVICE,
        "target_global_batch": TARGET_GLOBAL_BATCH,
        "lora_r": 16, "lr": LR, "warmup_ratio": WARMUP_FRAC,
        "weight_decay": WEIGHT_DECAY, "seed": SEED, "model_path": MODEL_DIR,
        "dist_mode": ("fsdp" if GPUS>1 else "auto"),
        "fsdp_strategy": FSDP_STRATEGY,
        "out_dir": OUT_DIR,
    }
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f: json.dump(cfg_stamp, f, indent=2)
    env_snapshot(os.path.join(OUT_DIR, "env_info.json"))
    
    
    RUN_META = os.path.join(OUT_DIR, f"run_{RUN_ID}.json")
    with open(RUN_META, "w") as f: json.dump({"run_start_ts": now()}, f, indent=2)
    
    
    tok = build_tokenizer()
    ds = load_dataset()
    ds = add_prompt_completion(ds, tok)
    probe = make_probe(ds)
    
    
    model = build_model()
    peft_cfg = lora_config()
    
    
    fsdp, fsdp_config = fsdp_args()
    train_cfg = build_train_config(world_size=WORLD_SIZE, multi_gpu=(GPUS>1))
    if fsdp is not None:
        train_cfg.fsdp = fsdp
        train_cfg.fsdp_config = fsdp_config
        
    
    trainer = assemble_trainer(
        model=model, tok=tok, train_cfg=train_cfg,
        ds_train=ds["train"], ds_eval=ds["eval"], peft_cfg=peft_cfg,
        world_size=WORLD_SIZE, log_csv=LOG_CSV, gpu_dmon=GPU_DMON,
    )
    
    
    trainer.train()
    
    
    with open(RUN_META) as f: meta = json.load(f)
    meta.update({"run_end_ts": now()})
    with open(RUN_META, "w") as f: json.dump(meta, f, indent=2)
    trainer.save_model(os.path.join(OUT_DIR, "last"))


def worker():
    import sys
    sys.argv = ["scr", "--gpus", str(GPUS), "--out", OUT_DIR]
    main()


def launch_from_notebook():
    notebook_launcher(worker, num_processes=GPUS)