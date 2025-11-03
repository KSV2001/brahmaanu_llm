from datasets import Dataset, DatasetDict
import pandas as pd
from configs.sft_config import TRAIN_SFT_PATH, SEED, PROBE_N, SCHEMA_NOTE
from transformers import AutoTokenizer

def load_dataset() -> DatasetDict:
    df = pd.read_parquet(TRAIN_SFT_PATH)
    assert "split" in df.columns, "Expected 'split' column with values 'train' or 'eval'"
    ds_all = Dataset.from_pandas(df, preserve_index=False)
    train = ds_all.filter(lambda ex: ex.get("split") == "train")
    eval = ds_all.filter(lambda ex: ex.get("split") == "eval")
    # drop the split column after filtering
    drop = lambda d: d.remove_columns([c for c in d.column_names if c == "split"])
    return DatasetDict({"train": drop(train), "eval": drop(eval)})




def add_prompt_completion(ds: DatasetDict, tok : AutoTokenizer) -> DatasetDict:
    def to_prompt_completion(batch):
        prompts, completions = [], []
        for q, a in zip(batch["question"], batch["target_json"]):
        msgs = [
        {"role": "system", "content": SCHEMA_NOTE},
        {"role": "user", "content": q},
        ]
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prompts.append(p)
        completions.append(a)
        return {"prompt": prompts, "completion": completions}
    
    
    cols = ds["train"].column_names
    ds2 = ds.map(to_prompt_completion, batched=True, remove_columns=cols)
    return ds2




def make_probe(ds: DatasetDict):
    return ds["eval"].shuffle(SEED).select(range(min(PROBE_N, len(ds["eval"]))))