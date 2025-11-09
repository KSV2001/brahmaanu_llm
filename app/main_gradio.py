# app/main_gradio.py
from __future__ import annotations
import os
import time
import uuid
import gzip
import pickle
import json
from typing import List, Dict, Any, Tuple

import gradio as gr

from configs.app_config import load_cfg, print_cfg_summary
from app.infer import init_infer, generate_text, count_tokens
from rag.rag_pipeline import build_index
from app import ratelimits  

# ---------------------------------------------------------
# config + load
# ---------------------------------------------------------
CFG = load_cfg()
print_cfg_summary(CFG)

# ---------------------------------------------------------
# RAG index
# ---------------------------------------------------------
RAG_IDX = None
if CFG.rag.index_pkl and os.path.exists(CFG.rag.index_pkl):
    try:
        with gzip.open(CFG.rag.index_pkl, "rb") as f:
            RAG_IDX = pickle.load(f)
        print("[app] Loaded RAG index from pickle")
    except Exception as e:
        print(f"[app] Failed to load RAG pickle ({e}), rebuilding…")
        RAG_IDX = build_index(
            folder_with_docs=CFG.rag.docs_folder,
            chunk_strategy=CFG.rag.chunk_strategy,
            n_units=CFG.rag.n_units,
        )
else:
    RAG_IDX = build_index(
        folder_with_docs=CFG.rag.docs_folder,
        chunk_strategy=CFG.rag.chunk_strategy,
        n_units=CFG.rag.n_units,
    )

if RAG_IDX is not None:
    cache_dir = (
        os.getenv("HF_HUB_CACHE")
        or os.getenv("TRANSFORMERS_CACHE")
        or "/workspace/hf"
    )
    if getattr(RAG_IDX, "model_dense", None) is None:
        try:
            from sentence_transformers import SentenceTransformer

            RAG_IDX.model_dense = SentenceTransformer(
                "BAAI/bge-large-en-v1.5", cache_folder=cache_dir
            )
        except Exception as e:
            print(f"[app] WARNING: could not reattach dense encoder: {e}")
    if getattr(RAG_IDX, "model_reranker", None) is None:
        try:
            from sentence_transformers import CrossEncoder

            RAG_IDX.model_reranker = CrossEncoder(
                "BAAI/bge-reranker-base", cache_folder=cache_dir
            )
        except Exception as e:
            print(f"[app] WARNING: could not reattach reranker: {e}")

# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------
TOK, GEN = init_infer(CFG)
MODES = ["SFT_RAG", "SFT", "BASE_RAG", "BASE"]

CITATION_DOC_MAP = {
    "G": "admin",
    "F": "data_pipelines",
    "H": "discoveries",
    "B": "instruments",
    "D": "calibration",
    "E": "constraints",
    "C": "detectors",
    "A": "ops",
}

# ---------------------------------------------------------
# helpers
# ---------------------------------------------------------
def _pack_context(question: str, mode: str, top_k: int) -> Tuple[str, List[str]]:
    can_rag = (
        "RAG" in mode
        and RAG_IDX is not None
        and getattr(RAG_IDX, "model_dense", None) is not None
    )
    if can_rag:
        ctx = RAG_IDX.retrieve(question, topk=top_k)
        ctx_ids: List[str] = []
        for c in ctx:
            ctx_ids.extend(c.get("fact_tags", []))
        if hasattr(RAG_IDX, "build_prompt"):
            prompt = RAG_IDX.build_prompt(question, ctx)
        else:
            from rag.rag_pipeline import build_prompt

            prompt = build_prompt(question, ctx)
        return prompt, ctx_ids
    from rag.rag_pipeline import build_prompt

    prompt = build_prompt(question, ctx=None)
    return prompt, []


def _trim_memory(memory: List[Dict[str, str]], max_turns: int) -> List[Dict[str, str]]:
    if len(memory) <= max_turns:
        return memory
    return memory[-max_turns:]


def _make_system_banner(mode: str, turns: int, msg: str = "") -> str:
    base = (
        f"mode={mode} · memory={turns}/{CFG.memory.max_turns} · max_new={CFG.model.max_new_tokens}"
    )
    if msg:
        return base + f" · {msg}"
    return base


def _build_prompt_with_history(history: List[Dict[str, str]], current_prompt: str) -> str:
    chunks = []
    for h in history:
        role = h.get("role", "user")
        txt = h.get("content", "")
        chunks.append(f"{role}: {txt}")
    chunks.append(f"user: {current_prompt}")
    return "\n".join(chunks)


def _docs_from_citations(cits: List[str]) -> List[str]:
    out = []
    for c in cits:
        if not isinstance(c, str):
            continue
        parts = c.split("-")
        if len(parts) < 2:
            continue
        letter = parts[1][:1]
        doc = CITATION_DOC_MAP.get(letter)
        if doc and doc not in out:
            out.append(doc)
    return out


def _pretty_from_json(raw: str, ctx_ids: List[str], dur: float, mode: str) -> str:
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError
        ans = obj.get("answer", "")
        cits = obj.get("citations", [])
        st = obj.get("status", "UNKNOWN")
        docs = _docs_from_citations(cits) if cits else []
        doc_line = ""
        if docs:
            doc_line = "Relevant docs: " + ", ".join(docs) + "\n"
        return (
            f"Answer:\n{ans}\n\n"
            f"{doc_line}"
            f"Status: {st}\n"
            f"Citations: {', '.join(cits) if cits else '—'}\n"
            f"(time: {dur:.2f}s, mode: {mode}, ctx_ids={list(set(ctx_ids))[:6]})"
        )
    except Exception:
        return (
            f"{raw}\n\n"
            f"(time: {dur:.2f}s, mode: {mode}, ctx_ids={list(set(ctx_ids))[:6]})"
        )

# ---------------------------------------------------------
# main chat
# ---------------------------------------------------------
def chat_fn(
    user_msg: str,
    chat_history: List[Tuple[str, str]],
    mode: str,
    use_history: bool,
    state: Dict[str, Any],
    request: gr.Request,
):
    now = time.time()
    if not state or "session_id" not in state:
        state = {"session_id": str(uuid.uuid4()), "memory": []}
    session_id = getattr(request, "session_hash", None) or state.get("session_id") or str(uuid.uuid4())


    ip = request.client.host if request and request.client else "unknown"

    ok, reason = ratelimits.precheck(session_id, ip, now)
    if not ok:
        chat_history.append((user_msg, f"[rate-limit] {reason}"))
        return "", chat_history, state, _make_system_banner(
            mode, len(state["memory"]), "rl-hit"
        )

    if count_tokens(user_msg) > 2000:
        chat_history.append((user_msg, "Input too long."))
        return "", chat_history, state, _make_system_banner(
            mode, len(state["memory"])
        )

    state["memory"].append({"role": "user", "content": user_msg})
    state["memory"] = _trim_memory(state["memory"], CFG.memory.max_turns)

    base_prompt, ctx_ids = _pack_context(user_msg, mode, CFG.rag.top_k)

    if use_history:
        hist_prompt = _build_prompt_with_history(state["memory"][:-1], user_msg)
        final_prompt = hist_prompt + "\n\n" + base_prompt
    else:
        final_prompt = base_prompt

    t0 = time.time()
    try:
        out_text = generate_text(
            GEN,
            TOK,
            prompt=final_prompt,
            mode=mode,
            max_new_tokens=CFG.model.max_new_tokens,
            temperature=CFG.model.temperature,
            timeout_s=CFG.guards.timeout_s,
        )
    except Exception as e:
        out_text = (
            '{"answer":"System error","citations":[],"status":"UNKNOWN","_err":"'
            + type(e).__name__
            + '"}'
        )
    duration = time.time() - t0

    ratelimits.postupdate(session_id, ip, duration, now)

    state["memory"].append({"role": "assistant", "content": out_text})

    render = _pretty_from_json(out_text, ctx_ids, duration, mode)
    chat_history.append((user_msg, render))

    return "", chat_history, state, _make_system_banner(mode, len(state["memory"]))

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
SAMPLE_QUESTIONS = [
    "Where is the observatory located?",
    "Describe the governance settings for observing cycle length, cycle start date, cycle end date, and director’s discretionary time per cycle?",
    "Outline the key policies for observing modes, observing queue categories, and remote observation support?",
    "Summarize the discovery details for Dhruva Exoplanet and Aditi Pulsar?",
    "Give the deepimager exposure time range?",
    "Summarize the key calibration settings for flat field frames per filter per night, flat field illumination source, viscam twilight flat window, and flat field exposure level?",
]

def build_ui():
    # sci-fi-ish background, but keep env override
    bg_img = os.getenv(
        "BRAHMAANU_BG",
        "https://images.unsplash.com/photo-1517694712202-14dd9538aa97?auto=format&fit=crop&w=1800&q=50",
    )

    with gr.Blocks(
        title="Brahmaanu LLM · Chat",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css=f"""
        body {{
            font-family: "Segoe UI", "Roboto", "system-ui", -apple-system, BlinkMacSystemFont, sans-serif;
            background: #020617;
        }}
        body::before {{
            content: "";
            position: fixed;
            inset: 0;
            background:
                linear-gradient(135deg, rgba(2,6,23,0.35), rgba(2,6,23,0.9)),
                url('{bg_img}') center/cover no-repeat fixed;
            z-index: -2;
            filter: saturate(1);
        }}
        .gradio-container {{
            max-width: 1180px !important;
            margin: 0 auto !important;
        }}
        #top-card {{
            background: radial-gradient(circle at 10% 20%, #0f2f6b 0%, #020617 60%);
            padding: 14px 16px 10px 16px;
            border-radius: 14px;
            color: #ffffff;
            box-shadow: 0 6px 18px rgba(0,0,0,0.35);
            margin-bottom: 14px;
            border: 1px solid rgba(139,162,191,0.4);
        }}
	#top-title {{
    	    font-weight: 650;
            font-size: 1.04rem;
            letter-spacing: 0.04em;
            color: #ffffff !important;
        }}
        #top-title p {{
            color: #ffffff !important;
        }}
	#top-title p strong {{
    	    color: #ffffff !important;
	}}
        #controls-card {{
            background: rgba(241,245,249,0.9);
            border: 1px solid #d0d7e2;
            border-radius: 12px;
            padding: 10px 10px 6px 10px;
            margin-bottom: 10px;
            color: #0f172a;
        }}
        #use-hist-box {{
            background: rgba(226,232,240,0.95);
            border: 1px solid rgba(148,163,184,0.65);
            border-radius: 8px;
            padding: 4px 8px 2px 8px;
        }}
        #input-card {{
            background: rgba(243,244,246,0.9);
            border: 1px solid #d1d5db;
            border-radius: 12px;
            padding: 10px 10px 12px 10px;
            margin-bottom: 10px;
        }}
        #chat-card {{
            background: rgba(248,250,252,0.98);
            border: 1px solid #cbd5f5;
            border-radius: 14px;
            padding: 6px;
            box-shadow: 0 4px 12px rgba(15,23,42,0.08);
        }}
        .chatbot {{
            border: none !important;
            background: #ffffff !important;
        }}
	input[type="checkbox"] {{
    	    accent-color: #0f5fff;
            border: 1px solid #000000 !important;
            box-shadow: 0 0 1px #000000 !important;
        }}
        """
    ) as demo:
        with gr.Column():
            # top header
            with gr.Row(elem_id="top-card"):
                gr.Markdown(
                    "Brahmaanu LLM · Mistral-7B SFT + RAG · **by Srivatsava Kasibhatla**",
                    elem_id="top-title",
                )

            # mode + history + status
            with gr.Row(elem_id="controls-card"):
                mode_dd = gr.Dropdown(choices=MODES, value="SFT_RAG", label="Mode")
                with gr.Column(elem_id="use-hist-box", scale=2):
                    use_hist = gr.Checkbox(
                        value=False,
                        label="Use conversation history for this request",
                    )
                status_lbl = gr.Markdown(_make_system_banner("SFT_RAG", 0))

            # we define state here, but will wire submits after chat is created
            state = gr.State({"session_id": str(uuid.uuid4()), "memory": []})

            # input section (moved up)
            with gr.Column(elem_id="input-card"):
                sample_dd = gr.Dropdown(
                    choices=SAMPLE_QUESTIONS,
                    value=None,
                    label="Sample questions",
                    interactive=True,
                )
                msg = gr.Textbox(
                    label="Prompt",
                    lines=3,
                    placeholder="Ask a handbook question...",
                )
                send = gr.Button("Send", variant="primary")

                def _pick_sample(q):
                    return q or ""

                sample_dd.change(_pick_sample, inputs=sample_dd, outputs=msg)

            # chat at bottom
            with gr.Row(elem_id="chat-card"):
                chat = gr.Chatbot(
                    height=520,
                    label="Chat",
                    elem_classes=["chatbot"],
                )

            # now wire the submit so chat exists
            def _submit(user_msg, chat_hist, mode, use_h, st, request: gr.Request):
                return chat_fn(user_msg, chat_hist, mode, use_h, st, request)

            send.click(
                _submit,
                [msg, chat, mode_dd, use_hist, state],
                [msg, chat, state, status_lbl],
            )
            msg.submit(
                _submit,
                [msg, chat, mode_dd, use_hist, state],
                [msg, chat, state, status_lbl],
            )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
        share=True,
    )

