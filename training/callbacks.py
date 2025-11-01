import os, time, json, signal, subprocess
import torch
from transformers.trainer_callback import TrainerCallback
from brahmaanu_llm.training.env_utils import now, is_rank0
from brahmaanu_llm.configs.sft_config import (
LOG_STEPS, SAVE_STEPS, MAX_SEQ_LEN, GPUS, LORA_RANK, TARGET_GLOBAL_BATCH, BATCH_PER_DEVICE,
)


class InfraLogger(TrainerCallback):

    def __init__(self, csv_path, gpu_dmon_path, world_size, tok):
        self.csv_path = csv_path; self.gpu_dmon_path = gpu_dmon_path
        self.last_step_start = None; self.last_loss = float("nan")
        self.dmon = None; self.vram_peak_mb = {}; self.mb_tokens = 0
        self.world_size = world_size
        self.tok = tok
        denom = BATCH_PER_DEVICE * max(1, world_size)
        self.grad_accum = math.ceil(TARGET_GLOBAL_BATCH / denom) if TARGET_GLOBAL_BATCH % denom != 0 else TARGET_GLOBAL_BATCH // denom
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if isinstance(logs, dict):
            self.last_loss = float(logs.get("loss", logs.get("train_loss", self.last_loss)))

    def _is_rank0(self):
        return (not torch.distributed.is_available() or not torch.distributed.is_initialized()
               ) or torch.distributed.get_rank() == 0

    def _write(self, row):
        if not self._is_rank0(): return
        new = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a") as f:
            if new: f.write(",".join(row.keys()) + "\n")
            f.write(",".join(str(row[k]) for k in row.keys()) + "\n")

    def on_train_begin(self, args, state, control, **kwargs):
        self.last_step_start = time.time()
        if self._is_rank0() and torch.cuda.is_available():
            try:
                self.dmon = subprocess.Popen(
                    ["nvidia-smi","dmon","-s","pucvmet","-o","DT","-f", self.gpu_dmon_path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
            except Exception:
                self.dmon = None

    def on_train_end(self, args, state, control, **kwargs):
        if self._is_rank0() and self.dmon:
            try: os.killpg(os.getpgid(self.dmon.pid), signal.SIGTERM)
            except Exception: pass
            with open(PEAK_VRAM_PER_GPU_JSON,"w") as f:
                json.dump(self.vram_peak_mb, f, indent=2)

    def on_train_batch_begin(self, args, state, control, **kwargs):
        self.last_step_start = time.time()
        inputs = kwargs.get("inputs") or {}
        labels = (kwargs.get("inputs") or {}).get("labels")
        ids = labels if isinstance(labels, torch.Tensor) else inputs.get("input_ids")
        if isinstance(ids, torch.Tensor):
            self.mb_tokens += int((ids != -100).sum().item()) if labels is not None else int((ids != tok.pad_token_id).sum().item())

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        if not is_rank0(): return
        step_time = max(1e-9, time.time() - (self.last_step_start or time.time()))
        tokens_s = self.mb_tokens / step_time
        lr = 0.0
        opt = kwargs.get("optimizer")
        if opt and opt.param_groups:
            lr = opt.param_groups[0]["lr"]
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                res_i = int(torch.cuda.max_memory_reserved(i)/(1024**2))
                self.vram_peak_mb[i] = max(self.vram_peak_mb.get(i,0), res_i)
                torch.cuda.reset_peak_memory_stats(i)
        row = {
        "ts": now(), "step": state.global_step, "loss": self.last_loss, "lr": lr,
        "tokens": self.mb_tokens, "tokens_per_s": round(tokens_s,2),
        "step_time_s": round(step_time,3), "world_size": self.world_size,
        "seq_len": MAX_SEQ_LEN, "lora_r": 16,
        "grad_accum": getattr(args, "gradient_accumulation_steps", None),
        "zero_stage_like": 3 if GPUS > 1 else 0,
        }
        self._write(row)
        self.mb_tokens = 0