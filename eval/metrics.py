# brahmaanu_llm/eval/metrics.py
import re, json, ast
from collections import Counter
from typing import List, Dict, Any

from brahmaanu_llm.configs.eval_config import ALLOWED_STATUS

CITATION_RE = re.compile(r"^BHF-[A-H]\d+$")  ## In our docs the ids follow this pattern

def _safe_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        try:
            return json.loads(s, strict=False)
        except Exception:
            return None

def validate_json_schema(text: str) -> Dict[str, Any]:
    """Validate the model’s JSON output schema."""
    out = {
        "is_json": False, "is_strict_json": False, "has_keys": False, "no_extra_keys": False,
        "answer_str": False, "citations_list": False, "citations_format": False,
        "status_ok": False, "valid": False, "valid_except_citations_format": False,
        "violations": [],
    }
    obj = None
    try:
        obj = json.loads(text)
        out["is_json"] = True
        out["is_strict_json"] = True
    except Exception:
        out["violations"].append("not_strict_json")
        obj = _safe_json_loads(text)
        if obj is None:
            out["violations"].append("not_json")
            return out
        out["is_json"] = True

    required = {"answer", "citations", "status"}
    keys = set(obj.keys())
    if required.issubset(keys):
        out["has_keys"] = True
    else:
        out["violations"].append(f"missing_keys:{sorted(list(required - keys))}")

    if keys == required:
        out["no_extra_keys"] = True
    else:
        extra = sorted(list(keys - required))
        if extra:
            out["violations"].append(f"extra_keys:{extra}")

    # answer
    if isinstance(obj.get("answer"), str) and obj["answer"].strip():
        out["answer_str"] = True
    else:
        out["violations"].append("answer_not_nonempty_string")

    # status
    if isinstance(obj.get("status"), str) and obj["status"] in ALLOWED_STATUS:
        out["status_ok"] = True
    else:
        out["violations"].append("status_invalid")

    # citations
    cits = obj.get("citations")
    if isinstance(cits, list):
        out["citations_list"] = True
        ok_fmt = (len(cits) > 0 and all(isinstance(x, str) and CITATION_RE.match(x) for x in cits)) \
                 or (len(cits) == 0 and out["status_ok"] and obj["status"] == "UNKNOWN")
        if ok_fmt:
            out["citations_format"] = True
        else:
            out["violations"].append("citations_bad_format_or_empty")
        if len(cits) != len(set(cits)):
            out["violations"].append("citations_duplicates")
    else:
        out["violations"].append("citations_not_list")

    out["valid"] = all([
        out["is_json"], out["has_keys"], out["no_extra_keys"], out["answer_str"],
        out["citations_list"], out["citations_format"], out["status_ok"],
    ])
    out["valid_except_citations_format"] = all([
        out["is_json"], out["has_keys"], out["no_extra_keys"], out["answer_str"],
        out["citations_list"], out["status_ok"],
    ])
    return out

def token_f1(pred: str, gold: str) -> float:
    ps = pred.split(); gs = gold.split()
    if not ps and not gs: return 1.0
    if not ps or not gs: return 0.0
    ps_c, gs_c = Counter(ps), Counter(gs)
    overlap = sum((ps_c & gs_c).values())
    if overlap == 0: return 0.0
    precision = overlap / len(ps)
    recall = overlap / len(gs)
    return 2 * precision * recall / (precision + recall)

def list_precision_recall_f1(pred_list: List[str], gold_list: List[str]):
    pred, gold = set(pred_list), set(gold_list)
    tp = len(pred & gold); fp = len(pred - gold); fn = len(gold - pred)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1

def evaluate_items(items: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate metrics over a list of {prompt, gold, pred_text} dicts."""
    schema_ok, key_ok, cit_fmt, status_ok = [], [], [], []
    ans_em, ans_f1, cit_f1 = [], [], []

    for it in items:
        v = validate_json_schema(it["pred_text"])
        schema_ok.append(1 if v["valid"] else 0)
        key_ok.append(1 if v["no_extra_keys"] else 0)
        cit_fmt.append(1 if v["citations_format"] else 0)
        status_ok.append(1 if v["status_ok"] else 0)

        pred_obj = _safe_json_loads(it["pred_text"])
        gold_obj = it["gold"]
        if isinstance(gold_obj, str):
            gold_obj = _safe_json_loads(gold_obj)
        if pred_obj and gold_obj:
            ans_em.append(1 if pred_obj.get("answer", "") == gold_obj.get("answer", "") else 0)
            ans_f1.append(token_f1(pred_obj.get("answer", ""), gold_obj.get("answer", "")))
            _, _, f1 = list_precision_recall_f1(pred_obj.get("citations", []), gold_obj.get("citations", []))
            cit_f1.append(f1)
        else:
            ans_em.append(0); ans_f1.append(0.0); cit_f1.append(0.0)

    def _mean(x): return float(sum(x) / len(x)) if x else 0.0

    return {
        "schema_valid_rate": _mean(schema_ok),
        "key_exact_rate": _mean(key_ok),
        "citations_format_rate": _mean(cit_fmt),
        "status_valid_rate": _mean(status_ok),
        "answer_exact_rate": _mean(ans_em),
        "answer_token_f1": _mean(ans_f1),
        "citations_f1": _mean(cit_f1),
    }
