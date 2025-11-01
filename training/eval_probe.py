import json, re, os
import numpy as np
import torch
from datasets import Dataset
from brahmaanu_llm.configs.sft_config import (
GEN_MAX_NEW_TOKENS, TEMPERATURE, TOP_P, TOP_K, NUM_BEAMS, MAX_SEQ_LEN, SCHEMA_NOTE
)




def parse_first_json(txt: str):
    try:
        first_close = txt.find("}")
        if first_close != -1:
            candidate = txt[:first_close+1]
            return json.loads(candidate)
        m = re.search(r"\{.*\}", txt, re.S)
        return json.loads(m.group(0)) if m else {"answer": txt.strip(), "citations": [], "status": "UNKNOWN"}
    except Exception:
        return {"answer": txt.strip(), "citations": [], "status": "UNKNOWN"}

@torch.no_grad()
def evaluate_probe(model, tokenizer, probe_ds: Dataset, outdir: str, step: int):
    model.eval()
    outs = []
    for ex in probe_ds:
        txt = ex["text"]
        try:
            q = txt.split("Question:\n",1)[1].split(" [/INST]",1)[0].strip()
        except Exception:
            q = "UNKNOWN"
        prompt = f"<s>[INST] {SCHEMA_NOTE} \n\nQuestion:\n{q} [/INST]\n" ## Standard prompt template
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=MAX_SEQ_LEN)
        enc.pop("token_type_ids", None)
        enc = {k: v.to(model.device) for k, v in enc.items()}

        gen_kwargs = dict(
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            do_sample=(TEMPERATURE > 0.0),
            temperature=max(TEMPERATURE, 1e-8),
            top_p=TOP_P,
            num_beams=NUM_BEAMS,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("</s>"),
        )
        if TOP_K > 0:
            gen_kwargs["top_k"] = TOP_K

        out = model.generate(**enc, **gen_kwargs)
        del enc, out
        torch.cuda.empty_cache()
        
        gen_ids = out[0, enc["input_ids"].shape[1]:]
        pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred = parse_first_json(pred_text)
        try:
            gold_json = txt.split("[/INST]\n",1)[1].split("</s>")[0]
            gold = parse_first_json(gold_json)
        except Exception:
            gold = {"answer":"", "citations": [], "status": "UNKNOWN"}

        em = int(pred.get("answer","").strip().lower() == gold.get("answer","").strip().lower())
        pset = set(map(str, pred.get("citations", [])))
        gset = set(map(str, gold.get("citations", [])))
        prec = (len(pset & gset) / max(1, len(pset))) if pset else 0.0
        rec  = (len(pset & gset) / max(1, len(gset))) if gset else 0.0
        outs.append({"question": q, "pred": pred, "gold": gold, "em": em, "cit_prec": prec, "cit_rec": rec})
    em_mean = float(np.mean([o["em"] for o in outs])) if outs else 0.0
    prec_m  = float(np.mean([o["cit_prec"] for o in outs])) if outs else 0.0
    rec_m   = float(np.mean([o["cit_rec"] for o in outs])) if outs else 0.0
    os.makedirs(os.path.join(outdir, "EVAL"), exist_ok=True)
    with open(os.path.join(outdir, f"EVAL/probe_step_{step}.json"), "w") as f:
        json.dump({"step": step, "em": em_mean, "cit_prec": prec_m, "cit_rec": rec_m, "details": outs}, f, indent=2)
    return em_mean