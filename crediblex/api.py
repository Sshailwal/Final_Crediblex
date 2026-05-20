import json, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, unquote
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from config import LOG_PATH, DEVICE

app = FastAPI(title="CredibleX API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=50, max_length=20000)

class AnalyzeURLRequest(BaseModel):
    url: str = Field(..., min_length=10, max_length=500)

def _append_log(entry: dict):
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

BIAS_LABEL_MAP = {
    "slightly_left":  "Slightly Left",
    "left":           "Left",
    "center":         "Center",
    "right":          "Right",
    "slightly_right": "Slightly Right",
}

INTENT_LABEL_MAP = {
    "news":    "News",
    "opinion": "Opinion",
    "satire":  "Satire",
}

def _shape_response(result: dict, metadata: dict = None) -> dict:
    """
    Converts the V2 inference output into the nested schema the frontend expects.
    Frontend expects: score, verdict, dimensions, metadata, key_findings, summary
    """
    fact_score  = result["factuality"]["score"]
    fact_label  = result["factuality"]["label"]
    bias_raw    = result["bias"]["label"]
    bias_conf   = result["bias"]["confidence"]
    intent_raw  = result["intent"]["label"]
    intent_conf = result["intent"]["confidence"]
    top_emotions = result["emotion"].get("top", [])

    bias_display   = BIAS_LABEL_MAP.get(bias_raw, bias_raw.title())
    intent_display = INTENT_LABEL_MAP.get(intent_raw, intent_raw.title())
    emotion_display = top_emotions[0]["label"].title() if top_emotions else "Neutral"

    # ── Key Findings ──────────────────────────────────────────────────────────
    findings = []

    if fact_score >= 0.7:
        findings.append({"type": "good", "icon": "✅", "text": f"Factuality score is high ({round(fact_score*100)}%), indicating mostly verified claims."})
    elif fact_score >= 0.4:
        findings.append({"type": "warn", "icon": "⚠️", "text": f"Factuality score is moderate ({round(fact_score*100)}%). Some claims may be unverified."})
    else:
        findings.append({"type": "bad", "icon": "🚨", "text": f"Low factuality score ({round(fact_score*100)}%). Content may contain misinformation."})

    if bias_raw == "center":
        findings.append({"type": "good", "icon": "⚖️", "text": "Political bias is centrist — balanced reporting detected."})
    elif bias_raw in ["slightly_left", "slightly_right"]:
        findings.append({"type": "warn", "icon": "📰", "text": f"Mild political lean detected: {bias_display} (confidence: {round(bias_conf*100)}%)."})
    else:
        findings.append({"type": "bad", "icon": "📢", "text": f"Strong political bias detected: {bias_display} (confidence: {round(bias_conf*100)}%)."})

    if intent_raw == "satire":
        findings.append({"type": "warn", "icon": "🎭", "text": "Content is identified as satire — not intended as factual reporting."})
    elif intent_raw == "opinion":
        findings.append({"type": "info", "icon": "💬", "text": "Content appears to be opinion/editorial, not straight news."})
    else:
        findings.append({"type": "good", "icon": "📄", "text": f"Intent classified as news reporting (confidence: {round(intent_conf*100)}%)."})

    if top_emotions:
        emo_names = ", ".join([e["label"] for e in top_emotions])
        findings.append({"type": "info", "icon": "🧠", "text": f"Dominant emotional tone: {emo_names}."})

    # ── Summary ───────────────────────────────────────────────────────────────
    verdict = result["verdict"]
    summary = (
        f"This content received a Trust Score of {result['trust_score']}/100 — classified as '{verdict}'. "
        f"Factuality is {fact_label.replace('_', ' ')} at {round(fact_score*100)}%. "
        f"The political framing is {bias_display} and the intent is classified as {intent_display}. "
        f"{'Emotional framing is present: ' + emotion_display + '.' if top_emotions else 'No strong emotional framing detected.'}"
    )

    # ── Build final response matching frontend schema ──────────────────────────
    return {
        "score":   result["trust_score"],
        "verdict": verdict,
        "dimensions": {
            "factuality": {
                "value":       fact_score,
                "label":       fact_label,
                "weight":      "50%",
                "explanation": f"The model scored this content {round(fact_score*100)}% on factuality based on language patterns, claim structure, and comparison with known factual sources."
            },
            "bias": {
                "value":        bias_display,
                "label":        bias_raw,
                "confidence":   bias_conf,
                "distribution": result["bias"].get("distribution", {}),
                "weight":       "20%",
                "explanation":  f"Detected a {bias_display} leaning with {round(bias_conf*100)}% confidence based on word choice and framing."
            },
            "intent": {
                "value":       intent_display,
                "label":       intent_raw,
                "confidence":  intent_conf,
                "weight":      "15%",
                "explanation": f"Content was classified as {intent_display} with {round(intent_conf*100)}% confidence."
            },
            "emotion": {
                "value":       emotion_display,
                "top":         top_emotions,
                "weight":      "15%",
                "explanation": f"{'Top emotional signals: ' + ', '.join([e['label'] for e in top_emotions]) + '.' if top_emotions else 'No dominant emotional tone detected — content appears relatively neutral.'}"
            }
        },
        "metadata": metadata or {
            "title":  "Text Analysis",
            "author": "Unknown",
            "date":   "Unknown"
        },
        "key_findings": findings,
        "summary":       summary,
        # Pass-through raw fields for any advanced consumers
        "_raw": {
            "windows_processed": result.get("windows_processed", 1),
            "trust_score":       result["trust_score"],
            "latency_ms":        result.get("latency_ms", 0)
        }
    }

@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}

@app.post("/analyze-text")
def analyze_text(request: AnalyzeRequest):
    from inference import run_inference
    t0 = time.time()
    result = run_inference(request.text)
    latency_ms = round((time.time() - t0) * 1000, 1)
    result["latency_ms"] = latency_ms

    words = request.text.split()
    word_count = len(words)
    metadata = {
        "title":  "Pasted Text Analysis",
        "author": "Unknown",
        "date":   "Unknown"
    }
    text_metadata = {
        "word_count":          word_count,
        "estimated_read_time": f"{max(1, round(word_count / 200))} min read",
        "contains_url":        any(w.startswith("http") for w in words),
        "looks_like_forward":  any(kw in request.text.lower() for kw in ["forwarded", "forward this", "share this", "share karo"]),
        "numeric_claims_count": sum(1 for w in words if any(c.isdigit() for c in w)),
        "likely_non_english":  False
    }

    response = _shape_response(result, metadata)
    response["text_metadata"] = text_metadata
    response["input_chars"] = len(request.text)

    _append_log({
        "ts":           datetime.now(timezone.utc).isoformat(),
        "endpoint":     "analyze-text",
        "chars":        len(request.text),
        "latency_ms":   latency_ms,
        "bias_label":   result["bias"]["label"],
        "fact_score":   result["factuality"]["score"],
        "intent_label": result["intent"]["label"]
    })
    return response

@app.post("/analyze-url")
def analyze_url(request: AnalyzeURLRequest):
    from scraper import extract_article
    from inference import run_inference

    article = extract_article(request.url)
    if article is None or article.get("text", "").strip() == "":
        fallback_text, fallback_metadata = _fallback_text_from_url(request.url)
        t0 = time.time()
        result = run_inference(fallback_text)
        latency_ms = round((time.time() - t0) * 1000, 1)
        result["latency_ms"] = latency_ms

        response = _shape_response(result, fallback_metadata)
        response["url"] = request.url
        response["input_chars"] = len(fallback_text)
        response["extraction_warning"] = (
            "Could not extract the full article text. The site may block scrapers, "
            "require login, or have no readable article body. This result uses URL "
            "metadata only, so treat it as a weak signal."
        )
        response["key_findings"].insert(0, {
            "type": "warn",
            "icon": "⚠️",
            "text": "Full article text could not be extracted; analysis used URL metadata only."
        })

        _append_log({
            "ts": datetime.now(timezone.utc).isoformat(),
            "endpoint": "analyze-url",
            "url": request.url,
            "chars": len(fallback_text),
            "latency_ms": latency_ms,
            "fallback": True,
            "bias_label": result["bias"]["label"],
            "fact_score": result["factuality"]["score"],
            "intent_label": result["intent"]["label"]
        })
        return response

    t0 = time.time()
    result = run_inference(article["text"])
    latency_ms = round((time.time() - t0) * 1000, 1)
    result["latency_ms"] = latency_ms

    metadata = {
        "title":  article.get("title", "Unknown Article"),
        "author": article.get("author", "Unknown"),
        "date":   article.get("date", "Unknown")
    }

    response = _shape_response(result, metadata)
    response["url"] = request.url
    response["input_chars"] = len(article["text"])

    _append_log({
        "ts":           datetime.now(timezone.utc).isoformat(),
        "endpoint":     "analyze-url",
        "url":          request.url,
        "chars":        len(article["text"]),
        "latency_ms":   latency_ms,
        "bias_label":   result["bias"]["label"],
        "fact_score":   result["factuality"]["score"],
        "intent_label": result["intent"]["label"]
    })
    return response


def _fallback_text_from_url(url: str) -> tuple[str, dict]:
    parsed = urlparse(url)
    source = parsed.netloc.replace("www.", "") or "Unknown source"
    path_text = unquote(parsed.path).replace("-", " ").replace("_", " ")
    path_text = " ".join(part for part in path_text.split("/") if part)
    title = path_text.title() if path_text else "URL Analysis"
    fallback_text = (
        f"News URL from {source}. Headline or slug: {path_text or url}. "
        "The full article body could not be extracted from the source page, "
        "so this analysis is based only on visible URL metadata and should be "
        "treated as a low-confidence preview."
    )
    return fallback_text, {
        "title": title,
        "author": source,
        "date": "Unknown",
    }

@app.get("/logs")
def get_logs():
    try:
        if not Path(LOG_PATH).exists():
            return {"logs": [], "count": 0}
        with open(LOG_PATH, "r") as f:
            lines = f.readlines()
        last_lines = lines[-100:]
        parsed_logs = [json.loads(line) for line in last_lines if line.strip()]
        return {"logs": parsed_logs, "count": len(parsed_logs)}
    except Exception as e:
        return {"logs": [], "count": 0, "error": str(e)}

@app.on_event("startup")
async def startup():
    from inference import _load_model_once
    _load_model_once()
    print("CredibleX V2 model loaded and ready.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=7860, reload=False)
