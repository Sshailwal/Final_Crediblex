"""
api.py — CredibleX FastAPI Server
===================================
Endpoints
---------
GET  /health          → uptime check, model info
POST /analyze         → {url}  → full trust report (scrapes article)
POST /analyze-text    → {text} → full trust report (raw text / WhatsApp)
GET  /logs            → recent request log

Run locally
-----------
    python api.py
or
    uvicorn api:app --reload --port 8000
"""

import os
import logging
import traceback
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from inference import analyze_url, analyze_text, generate_report

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_requests.log")

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)s | %(message)s",
    handlers = [
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("crediblex")

_request_log: list[dict] = []
_MAX_LOG     = 200


def _append_log(entry: dict) -> None:
    _request_log.append(entry)
    if len(_request_log) > _MAX_LOG:
        _request_log.pop(0)
    logger.info(
        "ANALYZED | url=%-60s | score=%-5s | verdict=%s | elapsed=%.2fs",
        entry.get("url", ""),
        entry.get("score", "err"),
        entry.get("verdict", "error"),
        entry.get("elapsed_s", 0.0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "CredibleX API",
    description = "News trust-scoring API. POST a URL or raw text to get a credibility report.",
    version     = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials = False,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic request models
# ─────────────────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    url: str

class TextRequest(BaseModel):
    text: str


# ─────────────────────────────────────────────────────────────────────────────
# Global exception handler
# ─────────────────────────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s\n%s", exc, traceback.format_exc())
    return JSONResponse(
        status_code = 500,
        content     = {"error": "internal_server_error", "detail": str(exc)},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: check model weights exist
# ─────────────────────────────────────────────────────────────────────────────
def _assert_model_exists():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.SAVE_PATH)
    if not os.path.isfile(model_path):
        raise HTTPException(
            status_code = 503,
            detail = {
                "error":   "model_unavailable",
                "message": f"Trained model weights not found at '{config.SAVE_PATH}'. Run train.py first.",
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health", summary="Health / uptime check")
def health():
    return {
        "status":      "ok",
        "model":       config.MODEL_NAME,
        "device":      str(config.DEVICE),
        "api_version": "2.0.0",
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /analyze   — URL → scrape + model
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/analyze", summary="Analyze a news article URL")
def analyze(request: AnalyzeRequest):
    """
    Accepts `{"url": "https://..."}` and returns a full CredibleX trust report.
    """
    url = (request.url or "").strip()
    if not url:
        raise HTTPException(400, detail={"error": "invalid_url", "message": "The 'url' field is required."})
    if not (url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(400, detail={"error": "invalid_url",
                                          "message": f"URL must start with http:// or https://. Got: '{url[:80]}'"})

    _assert_model_exists()

    t_start = datetime.now(timezone.utc)
    try:
        raw = analyze_url(url)
    except Exception as exc:
        logger.error("analyze_url() raised for %s: %s", url, exc)
        raise HTTPException(422, detail={"error": "scrape_failed", "message": str(exc), "url": url})

    if "error" in raw:
        _append_log({"timestamp": datetime.now(timezone.utc).isoformat(), "url": url,
                     "score": None, "verdict": "error",
                     "elapsed_s": (datetime.now(timezone.utc) - t_start).total_seconds(),
                     "error": raw["error"]})
        raise HTTPException(422, detail={"error": "scrape_failed", "message": raw["error"], "url": url})

    try:
        report = generate_report(raw)
    except Exception as exc:
        raise HTTPException(500, detail={"error": "report_generation_failed", "message": str(exc)})

    elapsed = (datetime.now(timezone.utc) - t_start).total_seconds()
    _append_log({
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "url":        url,
        "score":      report["score"],
        "verdict":    report["verdict"],
        "elapsed_s":  round(elapsed, 2),
        "factuality": report["dimensions"]["factuality"]["value"],
        "bias":       report["dimensions"]["bias"]["value"],
        "intent":     report["dimensions"]["intent"]["value"],
        "emotion":    report["dimensions"]["emotion"]["value"],
    })
    return report


# ─────────────────────────────────────────────────────────────────────────────
# POST /analyze-text   — raw text → model (WhatsApp / copy-paste)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/analyze-text", summary="Fact-check raw text (WhatsApp message, copied article)")
def analyze_text_endpoint(request: TextRequest):
    """
    Accepts `{"text": "..."}` and returns a full CredibleX trust report.

    Use this for:
    • WhatsApp forwards / viral messages
    • Copied article text
    • Any news content without a URL

    Minimum text length: 50 characters.
    """
    text = (request.text or "").strip()
    if not text:
        raise HTTPException(400, detail={"error": "invalid_text", "message": "The 'text' field is required."})
    if len(text) < 50:
        raise HTTPException(400, detail={
            "error":   "text_too_short",
            "message": f"Text must be at least 50 characters for meaningful analysis. Got {len(text)} chars.",
        })

    _assert_model_exists()

    t_start = datetime.now(timezone.utc)
    try:
        raw = analyze_text(text)
    except Exception as exc:
        logger.error("analyze_text() raised: %s", exc)
        raise HTTPException(422, detail={"error": "analysis_failed", "message": str(exc)})

    if "error" in raw:
        raise HTTPException(422, detail={"error": "analysis_failed", "message": raw["error"]})

    try:
        report = generate_report(raw)
    except Exception as exc:
        raise HTTPException(500, detail={"error": "report_generation_failed", "message": str(exc)})

    elapsed = (datetime.now(timezone.utc) - t_start).total_seconds()
    _append_log({
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "url":        f"[TEXT] {text[:60]}…",
        "score":      report["score"],
        "verdict":    report["verdict"],
        "elapsed_s":  round(elapsed, 2),
        "factuality": report["dimensions"]["factuality"]["value"],
        "bias":       report["dimensions"]["bias"]["value"],
        "intent":     report["dimensions"]["intent"]["value"],
        "emotion":    report["dimensions"]["emotion"]["value"],
    })
    return report


# ─────────────────────────────────────────────────────────────────────────────
# GET /logs
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/logs", summary="Return recent request log")
def get_logs(limit: int = 50):
    limit = max(1, min(limit, _MAX_LOG))
    return {"count": min(limit, len(_request_log)), "entries": _request_log[-limit:]}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
