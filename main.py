import os
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Deque, Optional
from collections import defaultdict, deque
import math

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# ---- OpenAI Client ----
try:
    from openai import OpenAI
    client = OpenAI()
except Exception as e:
    client = None
    logging.warning(f"OpenAI client not available: {e}")

app = FastAPI(title="Planville Backend (Light)")

# ---- CORS ----
FE_ORIGIN = os.getenv("FRONTEND_ORIGIN")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FE_ORIGIN] if FE_ORIGIN else ["http://localhost:3000"],
    allow_origin_regex=r"https://.*\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["*"],
)

# ---- Build/Version ----
APP_VERSION = os.getenv("APP_VERSION", "dev")
COMMIT_SHA = os.getenv("COMMIT_SHA", "")
BUILD_TIME_ISO = os.getenv("BUILD_TIME", datetime.now(timezone.utc).isoformat())

logging.basicConfig(level=logging.INFO, format="%(message)s")
intent_logger = logging.getLogger("intent")

# ---- Optional Imports (RAG / Scraper) ----
query_index = None
try:
    from rag_engine import query_index as _qi  # type: ignore
    query_index = _qi
except Exception as e:
    logging.warning(f"RAG disabled (rag_engine import failed): {e}")

get_scraped_context = None
try:
    from scraper import get_scraped_context as _scr  # type: ignore
    get_scraped_context = _scr
except Exception as e:
    logging.warning(f"Scraper not available: {e}")

# ---- Rate Limiter ----
REQUEST_BUCKETS: Dict[str, Deque[float]] = defaultdict(deque)
def _allow_request(bucket: str, limit: int, window_sec: int) -> bool:
    now = time.time()
    q = REQUEST_BUCKETS[bucket]
    while q and (now - q[0]) > window_sec:
        q.popleft()
    if len(q) >= limit:
        return False
    q.append(now)
    return True

# ---- Intent (keywords + semantic via OpenAI embeddings) ----
VALID_KEYWORDS = [
    "photovoltaik","pv","solaranlage","dach","wärmepumpe","klimaanlage",
    "angebot","kosten","preise","förderung","termin","beratung",
    "installation","montage","wartung","service","garantie",
    "photovoltaics","solar","roof","heat pump","air conditioner","ac",
    "quote","cost","price","subsidy","appointment","consultation",
    "install","maintenance","warranty"
]
def _match_intent(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(k in t for k in VALID_KEYWORDS)

INTENT_PHRASES = [
    "Angebot für Photovoltaikanlage",
    "PV Preise und Kosten",
    "Dachsanierung Angebot",
    "Wärmepumpe Beratung",
    "Klimaanlage Installation",
    "Montage Termin vereinbaren",
    "Wartung Service Garantie",
    "Förderung und Zuschüsse PV",
    "Angebot anfordern Photovoltaik",
    "quote for solar panels",
    "pv pricing cost",
    "roof renovation quote",
    "heat pump consultation",
    "air conditioner installation",
    "book installation appointment",
    "maintenance service warranty",
    "subsidy for pv",
    "request a photovoltaic quote"
]

def _openai_embed(text: str):
    if not client:
        return None
    try:
        emb = client.embeddings.create(model="text-embedding-3-small", input=text)
        return [float(x) for x in emb.data[0].embedding]
    except Exception as e:
        logging.warning(f"OpenAI embedding failed: {e}")
        return None

_INTENT_BANK = None
def _intent_bank_vectors():
    global _INTENT_BANK
    if _INTENT_BANK is None:
        vecs = []
        for p in INTENT_PHRASES:
            v = _openai_embed(p)
            if v:
                vecs.append(v)
        _INTENT_BANK = vecs
    return _INTENT_BANK or []

def _cosine(a, b) -> float:
    if not a or not b:
        return 0.0
    num = sum(x*y for x,y in zip(a,b))
    da = math.sqrt(sum(x*x for x in a))
    db = math.sqrt(sum(y*y for y in b))
    if da == 0 or db == 0:
        return 0.0
    return num/(da*db)

def _semantic_score(text: str) -> float:
    v = _openai_embed(text)
    bank = _intent_bank_vectors()
    if not v or not bank:
        return 0.0
    return max(_cosine(v, b) for b in bank)

INTENT_LOG_PATH = os.getenv("INTENT_LOG_PATH")
def log_intent_analytics(text: str, kw_hit: bool, sem_score: float, source: str):
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "kw": bool(kw_hit),
        "sem_score": float(sem_score or 0.0),
        "text": (text or "")[:512]
    }
    try:
        intent_logger.info(json.dumps(rec, ensure_ascii=False))
    except Exception:
        pass
    if INTENT_LOG_PATH:
        try:
            with open(INTENT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

# ---- Request Model (INI YANG KURANG DI VERSI KAMU) ----
class ChatRequest(BaseModel):
    message: str
    lang: Optional[str] = "de"

# ---- Prompt builder ----
def _build_prompt(user_message: str, context_text: str, lang: str, intent_ok: bool) -> str:
    cta = "Weitere Fragen? Kontakt: https://planville.de/kontakt" if lang == "de" else "More questions? Contact: https://planville.de/kontakt"
    style = "Antworte präzise, professionell und freundlich." if lang == "de" else "Answer concisely, professionally, and helpfully."
    scope = (
        "Thema: Photovoltaik, Dachsanierung, Wärmepumpe, Klimaanlage. Antworte auf Basis des CONTEXT unten. "
        "Wenn CONTEXT nicht ausreicht, antworte kurz (1–2 Sätze) und füge am Ende den CTA hinzu."
        if lang == "de" else
        "Topics: Photovoltaics, roofing, heat pumps, air conditioning. Answer based on CONTEXT below. "
        "If CONTEXT is insufficient, reply briefly (1–2 sentences) and append the CTA."
    )
    soft_gate = (
        "Falls die Nutzerfrage klar außerhalb der Themen ist, antworte sehr kurz (1–2 Sätze) und füge den CTA hinzu."
        if lang == "de" else
        "If the user question is clearly off-topic, answer very briefly (1–2 sentences) and append the CTA."
    )
    prompt = f"""{style}
{scope}
{soft_gate}

CONTEXT:
{context_text}

USER:
{user_message}

ASSISTANT (CTA am Ende falls nötig / append CTA if needed):
"""
    return prompt

# ---- Health & Version ----
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/version")
async def version():
    return {"version": APP_VERSION, "commit": COMMIT_SHA, "build_time": BUILD_TIME_ISO}

# ---- Build context ----
def _build_context(question: str) -> str:
    context_text = ""
    try:
        if query_index:
            ctx = query_index(question, k=4)
            if isinstance(ctx, list):
                context_text = "\n".join([str(c) for c in ctx if c])
            elif ctx:
                context_text = str(ctx)
    except Exception as e:
        logging.warning(f"RAG query failed: {e}")
    if not context_text and get_scraped_context:
        try:
            sc = get_scraped_context(question)
            if sc:
                context_text = sc
        except Exception as e:
            logging.warning(f"Scraper failed: {e}")
    return context_text

# ---- Endpoints ----
@app.post("/chat")
async def chat(request: ChatRequest):
    if not _allow_request("chat", 20, 60):
        raise HTTPException(status_code=429, detail="Too Many Requests")

    lang = request.lang or "de"
    kw = _match_intent(request.message)
    sem = _semantic_score(request.message)
    intent_ok = bool(kw or (sem >= 0.62))
    log_intent_analytics(request.message, kw, sem, "chat")

    context_text = _build_context(request.message)
    prompt = _build_prompt(request.message, context_text, lang, intent_ok)

    if not client:
        return JSONResponse({"answer": "KI ist derzeit nicht verfügbar. Bitte versuchen Sie es erneut."}, status_code=503)

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        answer = resp.choices[0].message.content or ""
        return {"answer": answer}
    except Exception as e:
        msg = "Ups, da ist etwas schiefgelaufen. Bitte versuchen Sie es erneut. Kontakt: https://planville.de/kontakt" if lang=="de" else \
              "Oops, something went wrong. Please try again. Contact: https://planville.de/kontakt"
        return {"answer": msg}

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    if not _allow_request("chat_stream", 60, 60):
        raise HTTPException(status_code=429, detail="Too Many Requests")

    lang = request.lang or "de"
    kw = _match_intent(request.message)
    sem = _semantic_score(request.message)
    intent_ok = bool(kw or (sem >= 0.62))
    log_intent_analytics(request.message, kw, sem, "chat_stream")

    context_text = _build_context(request.message)
    prompt = _build_prompt(request.message, context_text, lang, intent_ok)

    def token_stream():
        if not client:
            yield "Service unavailable."
            return
        try:
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                stream=True,
            )
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta.get("content")
                except Exception:
                    delta = None
                if delta:
                    yield delta
        except Exception:
            msg = "Ups, da ist etwas schiefgelaufen. Bitte versuchen Sie es erneut. Kontakt: https://planville.de/kontakt" if lang=="de" else \
                  "Oops, something went wrong. Please try again. Contact: https://planville.de/kontakt"
            yield msg

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(token_stream(), media_type="text/plain; charset=utf-8", headers=headers)

@app.get("/chat/sse")
async def chat_sse(message: str = Query(...), lang: str = Query("de")):
    if not _allow_request("chat_sse", 60, 60):
        raise HTTPException(status_code=429, detail="Too Many Requests")

    kw = _match_intent(message)
    sem = _semantic_score(message)
    intent_ok = bool(kw or (sem >= 0.62))
    log_intent_analytics(message, kw, sem, "chat_sse")

    context_text = _build_context(message)
    prompt = _build_prompt(message, context_text, lang or "de", intent_ok)

    def event_stream():
        if not client:
            yield "data: Service unavailable\n\n"
            return
        try:
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                stream=True,
            )
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta.get("content")
                except Exception:
                    delta = None
                if delta:
                    yield "data: " + delta.replace("\n", "\\n") + "\n\n"
            yield "event: done\ndata: [DONE]\n\n"
        except Exception:
            msg = "Ups, da ist etwas schiefgelaufen. Bitte versuchen Sie es erneut. Kontakt: https://planville.de/kontakt" if lang=="de" else \
                  "Oops, something went wrong. Please try again. Contact: https://planville.de/kontakt"
            yield "data: " + msg + "\n\n"

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(event_stream(), media_type="text/event-stream; charset=utf-8", headers=headers)
