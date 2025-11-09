# /app/ratelimits.py
import os
import time
from typing import Tuple, Optional

# ---- config (read from env or default) ----
SESSION_MAX_REQ = int(os.getenv("SESSION_MAX_REQ", 5))
SESSION_MAX_AGE_S = int(os.getenv("SESSION_MAX_AGE_S", 15 * 60))  # 15 min

IP_MAX_HOURLY = int(os.getenv("IP_MAX_HOURLY", 10))
IP_MAX_DAILY = int(os.getenv("IP_MAX_DAILY", 20))
IP_MAX_ACTIVE_S_DAILY = int(os.getenv("IP_MAX_ACTIVE_S_DAILY", 60 * 60))  # 1h

GLOBAL_MAX_HOURLY = int(os.getenv("GLOBAL_MAX_HOURLY", 100))
GLOBAL_MAX_DAILY = int(os.getenv("GLOBAL_MAX_DAILY", 200))
GLOBAL_MAX_ACTIVE_S_DAILY = int(os.getenv("GLOBAL_MAX_ACTIVE_S_DAILY", 6 * 60 * 60))  # 6h

COST_PER_SEC = float(os.getenv("COST_PER_SEC", 0.0005))
DAILY_COST_CAP = float(os.getenv("DAILY_COST_CAP", 5.0))
MONTHLY_COST_CAP = float(os.getenv("MONTHLY_COST_CAP", 25.0))

# ---- state (in-memory; resets on container restart) ----
_sessions = {}  # sid -> {"first_ts": float, "count": int}
_ips = {}       # ip -> {...}
_global = {
    "hour_key": None,
    "hour_count": 0,
    "day_key": None,
    "day_count": 0,
    "active_s_today": 0.0,
    "month_key": None,
    "cost_month": 0.0,
    "cost_day": 0.0,
}

# ---- helpers ----
def _hour_key(now: float) -> str:
    return time.strftime("%Y-%m-%d-%H", time.gmtime(now))

def _day_key(now: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(now))

def _month_key(now: float) -> str:
    return time.strftime("%Y-%m", time.gmtime(now))

# ---- core ----
def precheck(session_id: str, ip: str, now: Optional[float] = None) -> Tuple[bool, str]:
    """Check limits BEFORE running the model. Return (ok, msg)."""
    if now is None:
        _sessions.pop(session_id, None)
        _sessions[session_id] = {"first_ts": now, "count": 0}
        now = time.time()

    # if this session_id already exists but has a new timestamp (fresh reload), reset it
    if session_id not in _sessions:
        _sessions[session_id] = {"first_ts": now, "count": 0}
    else:
        # detect refresh: if Gradio reload gave a new timestamp recently (<1s old), reset counters
        if now - _sessions[session_id]["first_ts"] < 1:
            _sessions[session_id] = {"first_ts": now, "count": 0}

    # cleanup expired sessions
    expired = [sid for sid, rec in _sessions.items() if now - rec["first_ts"] > SESSION_MAX_AGE_S]
    for sid in expired:
        _sessions.pop(sid, None)

    # ----- session -----
    sess = _sessions.get(session_id)
    if not sess:
        _sessions[session_id] = {"first_ts": now, "count": 0}
        sess = _sessions[session_id]

    if now - sess["first_ts"] > SESSION_MAX_AGE_S:
        return False, f"Session age exceeded ({SESSION_MAX_AGE_S//60} min). Refresh page."
    if sess["count"] >= SESSION_MAX_REQ:
        return False, f"Session request limit reached ({SESSION_MAX_REQ}). Refresh page."

    # ----- ip -----
    day_k = _day_key(now)
    hour_k = _hour_key(now)
    iprec = _ips.get(ip)
    if not iprec:
        _ips[ip] = {
            "hour_key": hour_k,
            "hour_count": 0,
            "day_key": day_k,
            "day_count": 0,
            "active_s_today": 0.0,
        }
        iprec = _ips[ip]

    if iprec["hour_key"] != hour_k:
        iprec["hour_key"] = hour_k
        iprec["hour_count"] = 0
    if iprec["day_key"] != day_k:
        iprec["day_key"] = day_k
        iprec["day_count"] = 0
        iprec["active_s_today"] = 0.0

    if iprec["hour_count"] >= IP_MAX_HOURLY:
        return False, f"Per-IP hourly limit reached ({IP_MAX_HOURLY}). Try later."
    if iprec["day_count"] >= IP_MAX_DAILY:
        return False, f"Per-IP daily limit reached ({IP_MAX_DAILY}). Try tomorrow."
    if iprec["active_s_today"] >= IP_MAX_ACTIVE_S_DAILY:
        return False, f"Per-IP daily active time reached ({IP_MAX_ACTIVE_S_DAILY//60} min)."

    # ----- global -----
    g = _global
    if g["hour_key"] != hour_k:
        g["hour_key"] = hour_k
        g["hour_count"] = 0
    if g["day_key"] != day_k:
        g["day_key"] = day_k
        g["day_count"] = 0
        g["active_s_today"] = 0.0
        g["cost_day"] = 0.0
    mk = _month_key(now)
    if g["month_key"] != mk:
        g["month_key"] = mk
        g["cost_month"] = 0.0

    if g["hour_count"] >= GLOBAL_MAX_HOURLY:
        return False, f"Server hourly limit reached ({GLOBAL_MAX_HOURLY})."
    if g["day_count"] >= GLOBAL_MAX_DAILY:
        return False, f"Server daily limit reached ({GLOBAL_MAX_DAILY})."
    if g["active_s_today"] >= GLOBAL_MAX_ACTIVE_S_DAILY:
        return False, f"Server daily active time reached ({GLOBAL_MAX_ACTIVE_S_DAILY//3600} h)."
    if g["cost_day"] >= DAILY_COST_CAP:
        return False, f"Daily cost cap reached (${DAILY_COST_CAP})."
    if g["cost_month"] >= MONTHLY_COST_CAP:
        return False, f"Monthly cost cap reached (${MONTHLY_COST_CAP})."

    return True, "OK"

# ---- update ----
def postupdate(session_id: str, ip: str, duration_s: float, now: Optional[float] = None) -> None:
    """Update counters AFTER model run."""
    if now is None:
        now = time.time()
    day_k = _day_key(now)
    hour_k = _hour_key(now)
    mk = _month_key(now)

    # session
    sess = _sessions.get(session_id)
    if sess:
        sess["count"] += 1

    # ip
    iprec = _ips.get(ip)
    if iprec:
        if iprec["hour_key"] != hour_k:
            iprec["hour_key"] = hour_k
            iprec["hour_count"] = 0
        if iprec["day_key"] != day_k:
            iprec["day_key"] = day_k
            iprec["day_count"] = 0
            iprec["active_s_today"] = 0.0
        iprec["hour_count"] += 1
        iprec["day_count"] += 1
        iprec["active_s_today"] += float(duration_s)

    # global
    g = _global
    if g["hour_key"] != hour_k:
        g["hour_key"] = hour_k
        g["hour_count"] = 0
    if g["day_key"] != day_k:
        g["day_key"] = day_k
        g["day_count"] = 0
        g["active_s_today"] = 0.0
        g["cost_day"] = 0.0
    if g["month_key"] != mk:
        g["month_key"] = mk
        g["cost_month"] = 0.0

    g["hour_count"] += 1
    g["day_count"] += 1
    g["active_s_today"] += float(duration_s)

    cost = float(duration_s) * COST_PER_SEC
    g["cost_day"] += cost
    g["cost_month"] += cost

