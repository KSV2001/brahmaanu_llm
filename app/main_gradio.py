# app/main_gradio.py
from __future__ import annotations
import os, time, uuid, gzip
from typing import List, Dict, Any, Tuple

import gradio as gr

from configs.app_config import load_cfg, print_cfg_summary
from rag.rag_pipeline import build_index, create_rag_prompts
# to be provided next:
from app.infer import init_infer, generate_text, count_tokens  # noqa: F401

# ---------- Startup ----------
CFG = load_cfg()
print_cfg_summary(CFG)

# Load/Build RAG index once


RAG_IDX = None
if CFG.rag.index_pkl and os.path.exists(CFG.rag.index_pkl):
    try:
        with gzip.open(CFG.rag.index_pkl, "rb") as f:
            RAG_IDX = pickle.load(f)
    except Exception:
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

# Init models once (merged SFT + base)
TOK, GEN = init_infer(CFG)  # returns tokenizer and a callable used by generate_text

# ---------- Helpers ----------
MODES = ["SFT_RAG", "SFT", "BASE_RAG", "BASE"]

def _trim_memory(memory: List[Dict[str, str]], max_turns: int) -> List[Dict[str, str]]:
    if len(memory) <= max_turns:
        return memory
    # keep last max_turns pairs
    return memory[-max_turns:]

def _pack_context(question: str, mode: str, top_k: int) -> Tuple[str, List[str]]:
    """Return prompt and context IDs."""
    if "RAG" in mode:
        ctx = RAG_IDX.retrieve(question, topk=top_k)
        ctx_ids = []
        for c in ctx:
            # collect fact tags
            ctx_ids.extend(c.get("fact_tags", []))
        prompt = RAG_IDX.build_prompt(question, ctx) if hasattr(RAG_IDX, "build_prompt") else None
        if prompt is None:
            # fallback: use exported function if index lacks method
            from brahmaanu_llm.rag.rag_pipeline import build_prompt
            prompt = build_prompt(question, ctx)
        return prompt, ctx_ids
    else:
        # no-RAG single-turn prompt
        from brahmaanu_llm.rag.rag_pipeline import build_prompt
        return build_prompt(question, ctx=None), []

def _make_system_banner(mode: str, turns: int) -> str:
    return f"mode={mode} · memory={turns}/{CFG.memory.max_turns} · max_new={CFG.model.max_new_tokens}"

# ---------- Chat handler ----------
def chat_fn(user_msg: str,
            chat_history: List[Tuple[str, str]],
            mode: str,
            state: Dict[str, Any]) -> Tuple[str, List[Tuple[str, str]], Dict[str, Any], str]:
    """
    state: {"session_id": str, "memory": List[Dict[str,str]]}
    """
    t0 = time.time()
    if not state or "session_id" not in state:
        state = {"session_id": str(uuid.uuid4()), "memory": []}

    # guard: empty or too long input
    ulen = count_tokens(user_msg)
    if ulen > 2000:
        warn = "Your message is too long for this demo. Please shorten it."
        chat_history.append((user_msg, warn))
        return "", chat_history, state, _make_system_banner(mode, len(state["memory"]))

    # append to memory
    state["memory"].append({"role": "user", "content": user_msg})
    # trim memory turns
    state["memory"] = _trim_memory(state["memory"], CFG.memory.max_turns)

    # build prompt
    prompt, ctx_ids = _pack_context(user_msg, mode, CFG.rag.top_k)

    # generate
    try:
        out_text = generate_text(
            GEN,
            TOK,
            prompt=prompt,
            mode=mode,
            max_new_tokens=CFG.model.max_new_tokens,
            temperature=CFG.model.temperature,
            timeout_s=CFG.guards.timeout_s,
        )
    except Exception as e:
        out_text = f'{{"answer":"System error","citations":[],"status":"UNKNOWN","_err":"{type(e).__name__}"}}'

    # update memory
    state["memory"].append({"role": "assistant", "content": out_text})

    # show
    latency = f"{(time.time()-t0):.2f}s"
    assistant_render = f"{out_text}\n\n(meta: {latency}, ctx_ids={list(set(ctx_ids))[:6]})"
    chat_history.append((user_msg, assistant_render))

    return "", chat_history, state, _make_system_banner(mode, len(state["memory"]))

def clear_fn() -> Tuple[List[Tuple[str, str]], Dict[str, Any], str]:
    return [], {"session_id": str(uuid.uuid4()), "memory": []}, _make_system_banner(gr.State.value if hasattr(gr.State, "value") else MODES[0], 0)  # safe default

# ---------- UI ----------
def build_ui():
    with gr.Blocks(title="Brahmaanu LLM · Chat", theme=gr.themes.Default()) as demo:
        gr.Markdown("### Brahmaanu LLM · Mistral-7B SFT + RAG\nCost-capped demo. JSON answers with citations.")
        with gr.Row():
            mode_dd = gr.Dropdown(choices=MODES, value="SFT_RAG", label="Mode")
            status_lbl = gr.Markdown(_make_system_banner("SFT_RAG", 0))

        chat = gr.Chatbot(height=520, type="messages", label="Chat")
        state = gr.State({"session_id": str(uuid.uuid4()), "memory": []})

        msg = gr.Textbox(placeholder="Ask about the observatory…", label="Message", lines=3)
        with gr.Row():
            send = gr.Button("Send", variant="primary")
            clear = gr.Button("New session")

        def _submit(user_msg, chat_hist, mode, st):
            return chat_fn(user_msg, chat_hist, mode, st)

        send.click(_submit, [msg, chat, mode_dd, state], [msg, chat, state, status_lbl])
        msg.submit(_submit, [msg, chat, mode_dd, state], [msg, chat, state, status_lbl])
        clear.click(lambda: clear_fn(), outputs=[chat, state, status_lbl])

        # health
        gr.Markdown("> Health: if you see errors, the model may be cold-starting. Retry once.")

    return demo

if __name__ == "__main__":
    demo = build_ui()
    # In serverless, GRADIO_SERVER_PORT is often set. Default to 8080.
    port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "8080")))
    demo.queue(concurrency_count=1).launch(server_name="0.0.0.0", server_port=port, show_error=True)
