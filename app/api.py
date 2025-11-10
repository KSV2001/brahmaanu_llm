# app/api.py
# FastAPI wrapper for Brahmaanu LLM

from typing import List, Tuple, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.main_gradio import chat_fn, SAMPLE_QUESTIONS, MODES

class ChatRequest(BaseModel):
    user_msg: str
    chat_history: List[Tuple[str, str]]
    mode: str
    use_history: bool
    state: Dict[str, Any]

class ChatResponse(BaseModel):
    chat_history: List[Tuple[str, str]]
    state: Dict[str, Any]
    status: str

app = FastAPI(title="Brahmaanu LLM API")

@app.get("/sample_questions", response_model=List[str])
def get_sample_questions() -> List[str]:
    return SAMPLE_QUESTIONS

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if request.mode not in MODES:
        raise HTTPException(status_code=400, detail=f"mode must be one of {MODES}")
    chat_history = request.chat_history or []
    state = request.state or {}
    _, chat_hist, state_out, status = chat_fn(
        request.user_msg,
        chat_history,
        request.mode,
        request.use_history,
        state,
        request=None,   # no Gradio request context
    )
    return ChatResponse(chat_history=chat_hist, state=state_out, status=status)
