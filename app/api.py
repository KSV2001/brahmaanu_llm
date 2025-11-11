# app/api.py
# FastAPI wrapper for Brahmaanu LLM

"""
FastAPI wrapper for Brahmaanu LLM

This module exposes a simple REST API around the existing
`chat_fn` defined in ``app/main_gradio.py``.  It allows
external callers (e.g. a Hugging Face Space) to send a user
message along with the current chat history, model mode and
other flags, and receive an updated chat history and state.

The API has two endpoints:

* ``POST /chat`` – accepts a JSON body with the fields
  ``user_msg``, ``chat_history``, ``mode``, ``use_history`` and
  ``state`` and returns an updated history, state and status
  banner.  The request signature mirrors the ``chat_fn``
  signature, except ``request`` is omitted.  Rate limits and
  token budgeting are still enforced server‑side.

* ``GET /sample_questions`` – returns the list of sample
  questions defined in ``app/main_gradio.py``.  This is useful
  for populating a drop‑down on a remote UI without duplicating
  the strings.

Note that this module reuses the existing LLM and RAG state
loaded at import time in ``app/main_gradio.py``.  The import is
performed once at startup, so subsequent requests reuse the
loaded models.  There is no Gradio dependency in this module.

Usage (inside Dockerfile CMD):

    uvicorn app.api:app --host 0.0.0.0 --port 8080

"""

from typing import List, Tuple, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging 
import uuid
from fastapi import Response

logger = logging.getLogger("brahmaanu_api")
logging.basicConfig(level=logging.INFO)

# Import the chat function and sample questions from the Gradio app.
# This implicitly loads the models via init_infer() and builds the RAG index.
from app.main_gradio import chat_fn, SAMPLE_QUESTIONS, MODES

COOKIE_NAME = "client_id"
COOKIE_MAX_AGE = 30 * 24 * 3600  # 30 days




def pick_client_ip(xff: str, fallback: str) -> str:
    """
    XFF -> client IP.
    Use the first address (leftmost). Fallback to peer.
    """
    if not xff:
        return fallback
    parts = [p.strip() for p in xff.split(",") if p.strip()]
    if not parts:
        return fallback
    return parts[0]  # client IP

class ChatRequest(BaseModel):
    """Schema for incoming chat requests."""

    user_msg: str
    chat_history: List[Tuple[str, str]]
    mode: str
    use_history: bool
    state: Dict[str, Any]

    class Config:
        schema_extra = {
            "example": {
                "user_msg": "What are the observatory operating hours?",
                "chat_history": [],
                "mode": "SFT_RAG",
                "use_history": False,
                "state": {},
            }
        }


class ChatResponse(BaseModel):
    """Schema for chat responses."""

    chat_history: List[Tuple[str, str]]
    state: Dict[str, Any]
    status: str


app = FastAPI(title="Brahmaanu LLM API")

## Health check
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/sample_questions", response_model=List[str])
def get_sample_questions() -> List[str]:
    """Return the list of sample questions for populating UIs."""

    return SAMPLE_QUESTIONS


# @app.api_route("/chat", methods=["GET", "POST"], response_model=ChatResponse)
# async def chat(request: ChatRequest) -> ChatResponse:
#     """
#     Process a chat message and return an updated history, state and status.

#     This simply wraps the underlying ``chat_fn`` from the Gradio app.  The
#     ``gr.Request`` parameter of ``chat_fn`` is passed as ``None`` since no
#     web request context exists for this API.

#     If an invalid mode is supplied the API will return an HTTP 400.
#     """

#     # Basic validation for mode
#     if request.mode not in MODES:
#         raise HTTPException(status_code=400, detail=f"mode must be one of {MODES}")

#     # Ensure chat_history and state are lists/dicts to avoid type errors
#     chat_history = request.chat_history or []
#     state = request.state or {}

#     # Call the underlying chat function.  The first return value is always
#     # an empty string (the Gradio "prompt" box), which we ignore here.
#     _, chat_hist, state_out, status = chat_fn(
#         request.user_msg,
#         chat_history,
#         request.mode,
#         request.use_history,
#         state,
#         request=None,
#     )

#     return ChatResponse(chat_history=chat_hist, state=state_out, status=status)




@app.api_route("/chat", methods=["GET", "POST"], response_model=ChatResponse)
async def chat(req: Request, resp: Response) -> ChatResponse:
    if req.method == "POST":
        data = await req.json()
    else:
        qp = req.query_params
        logger.info(
            "GET /chat hit from %s with params: user_msg=%s, mode=%s, use_history=%s",
            req.client.host if req.client else "unknown",
            qp.get("user_msg", ""),
            qp.get("mode", "chat"),
            qp.get("use_history", "true"),
        )
        data = {
            "user_msg": qp.get("user_msg", ""),
            "chat_history": [],
            "mode": qp.get("mode", "chat"),
            "use_history": qp.get("use_history", "true").lower() == "true",
            "state": {},
        }

    mode = data.get("mode", "chat")
    if mode not in MODES:
        raise HTTPException(status_code=400, detail=f"mode must be one of {MODES}")

    chat_history = data.get("chat_history") or []
    state = data.get("state") or {}

    # IP
    xff = req.headers.get("x-forwarded-for")
    peer = req.client.host if req.client else "unknown"
    real_ip = pick_client_ip(xff, peer)
    req.state.real_ip = real_ip

    # session
    session_id = (
        data.get("session_id")
        or req.headers.get("x-session-id")
        or str(uuid.uuid4())
    )
    req.state.session_id = session_id

    # cookie-backed client id
    client_id = req.cookies.get(COOKIE_NAME)
    if not client_id:
        client_id = str(uuid.uuid4())
        resp.set_cookie(
            key=COOKIE_NAME,
            value=client_id,
            max_age=COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
        )
    req.state.client_id = client_id

    logger.info(
        "CHAT ip=%s cid=%s sid=%s xff=%s peer=%s",
        real_ip,
        client_id,
        session_id,
        xff,
        peer,
    )

    # call your existing chat_fn
    _, chat_hist, state_out, status = chat_fn(
        data.get("user_msg", ""),
        chat_history,
        mode,
        data.get("use_history", True),
        state,
        request=req,
    )

    return ChatResponse(chat_history=chat_hist, state=state_out, status=status)

@app.api_route("/", methods=["GET", "POST"])
async def root_passthrough(request: Request):
    # forward to /chat logic or just return a message
    return {"detail": "use /chat"}
