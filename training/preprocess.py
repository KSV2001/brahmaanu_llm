# Data cleaning and generating a pandas DataFrame with convenient columns for Supervised Finetuning on the Q-A pairs

## Imports
import json, glob, pandas as pd, numpy as np, re
from copy import deepcopy
from brahmaanu_llm.configs.sft_config import *
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast
import pandas as pd
from datasets import Dataset, DatasetDict

def count_tokens(txt: str) -> int: return len(tok(txt).input_ids)


## Regex patterns to clean the observed noise in the SFT questions generated
_COLLAPSE_PUNCT_RE = re.compile(r'([^\w\s])\1+') # collapse "!!"->"!", ",,"->",", "...")->".", "???"->"?", etc.
_DOT_QUESTION_RE = re.compile(r'\.\s*\?')          # ".?" or ".   ?" -> "?"
_COMMA_QUESTION_RE = re.compile(r',\s*\?')         # ",?" or ",   ?" -> "?"
_DOT_COMMA_RE = re.compile(r'\.\s*,')              # ".,", ".   ," -> ","
_SPACE_BEFORE_PUNCT_RE = re.compile(r'\s+([?.!,;:])')  # " ?"->"?", " ."->".", etc.
_REMOVE_WRONG_RE = re.compile(r'\s*\(wrong\)')  ## Removing quesitons where the deliberate out-of-docs questions have (wrong) embedded in them


## Helpers to clean and normalize the SFT jsons
def collapse_punct(text: str) -> str:
    if not isinstance(text, str):
        return text
    return _COLLAPSE_PUNCT_RE.sub(r'\1', text)

def fix_punct_spacing(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = _DOT_QUESTION_RE.sub('?', text)
    text = _COMMA_QUESTION_RE.sub('?', text)
    text = _DOT_COMMA_RE.sub(',', text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r'\1', text)
    return text

def remove_wrong(text: str) -> str:
    if not isinstance(text, str):
        return text
    return _REMOVE_WRONG_RE.sub('', text)


## Function that uses all the above helpers
def normalize_qas(items: list[dict]) -> list[dict]:
    '''Function to normalize the Q/As generated from the docs for SFT purpose.'''
    out = deepcopy(items) ## Copies the input to not modify it 
    for obj in out: 
        if "question" in obj:
            obj["question"] = fix_punct_spacing(collapse_punct(remove_wrong(obj["question"])))
        resp = obj.get("response")
        if isinstance(resp, dict) and "answer" in resp:
            resp["answer"] = fix_punct_spacing(collapse_punct(resp["answer"]))
    return out


def render_row(q, ans, cits, status):
    prompt = f"<s>[INST] {SCHEMA_NOTE}\n\nQuestion:\n{q} [/INST]\n"
    # normalize
    cits = sorted({str(x) for x in cits})
    status = status.upper()
    target = json.dumps(
        {"answer": ans, "citations": list(cits), "status": status},
        ensure_ascii=False, separators=(",", ":")  # no spaces
    )
    return prompt + target + "</s>"  # close sample
    

def norm_status(s): 
    s=str(s).upper().strip()
    return s if s in {"OK","PARTIAL","UNKNOWN","CORRECTED"} else "UNKNOWN"

def clean_q(s): return re.sub(r"\s+", " ", s).strip()
    
def clean_a(s): return re.sub(r"\s+", " ", s).strip()

def rows_from_docs(doc_list, doc_letter_codes, doc_names):
    rows=[]
    for doc_name, doc_id, data in zip(doc_names, doc_letter_codes, doc_list):
        #data = json.load(open(p))
        for ex in data:
            q = clean_q(ex["question"])
            ans = clean_a(ex["response"]["answer"])
            cits = [x for x in ex["response"].get("citations", [])]
            status = norm_status(ex["response"].get("status","OK"))
            rows.append({"doc_id":doc_id, "doc_name" : doc_name ,"question":q,"answer":ans,"citations":cits,"status":status})
    return pd.DataFrame(rows)

## Giving a letter id to each doc
def map_name_to_id(name):
    
    if name not in NAME_TO_ID_MAPPER_DICT.keys():
        raise ValueError(f"input : {name} not in document names.")
    return NAME_TO_ID_MAPPER_DICT[name]

def render(df, tok):
    ''' Takes a df and tokenizer and generates the prompt and target as json as requred for the Brahmaanu chatbot.'''
    prompt = "<s>[INST] " + SCHEMA_NOTE + "\n\nQuestion:\n" + df["question"] + " [/INST]\n"
    target = df.apply(lambda r: json.dumps(
        {"answer": r["answer"], "citations": r["citations"], "status": r["status"]},
        ensure_ascii=False, separators=(",",":")), axis=1)
    text = prompt + target
    n_tokens = text.map(lambda s: len(tok(s).input_ids))
    return df.assign(prompt=prompt, target_json=target, text=text, n_tokens=n_tokens)


if __name__== "__main__":

    ## Get the docs and clean them, and convert to the df
    L = glob.glob(str(QA_JSONS_PATH/"*json"))
    filtered_docs = []
    ## Filter the docs
    for p in L:
        with open(p, "rb") as file:
          d = json.loads(file.read())
        filtered_docs.append(normalize_qas(d))
    doc_names =  ["_".join(l.split("/")[-1].split("_")[:-2] ) for l in L] ## Extract the doc names

    ## Get the clean df to save for SFT
    df_sft = rows_from_docs(filtered_docs, doc_letter_codes = [NAME_TO_ID_MAPPER_DICT[n] for n in doc_names], doc_names = doc_names)


    ## Get the actual LLM tokenizer
    tok = AutoTokenizer.from_pretrained(BRAHMAANU_LLM_ID, use_fast=True)
    tok = PreTrainedTokenizerFast(tokenizer_file=TOK_JSON) ## Tokenizer assumed to be cached in TOK_JSON folder. Check the sft_configs.py to change.

    ## Standard settings, from sft_configs.py to get the tokenizer to generate the prompt 
    tok.pad_token = PAD_TOKEN
    tok.eos_token = EOS_TOKEN
    tok.bos_token = BOS_TOKEN
    tok.unk_token = UNK_TOKEN
    tok.model_max_length = MODEL_MAX_LENGTH

    ## Get the rendered df, with exact model prompt for each Q-A, for SFT.
    df_sft = render(df_sft, tok)
    
    ds_all = Dataset.from_pandas(df_sft, preserve_index=False)
    split = ds_all.train_test_split(test_size=0.1, seed=SEED)
    ds = DatasetDict({"train": split["train"], "eval": split["test"]})

    df_sft_train= pd.DataFrame(ds["train"])
    df_sft_train["split"] = "train"
    df_sft_eval = pd.DataFrame(ds["eval"])
    df_sft_eval["split"] = "eval"

    ## Save to filepaths
    pd.concat([df_sft_eval, df_sft_train]).reset_index(drop = True).to_parquet(SFT_PATH)
    df_sft_train.to_parquet(TRAIN_SFT_PATH)
    df_sft_eval.to_parquet(EVAL_SFT_PATH)
    
    
    