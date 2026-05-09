# emotion_label: int→multihot
"""
inference.py — CredibleX Model Inference
==========================================
Public API
----------
  analyze_url(url)   → scrape article from URL, run model
  analyze_text(text) → accept raw text directly (WhatsApp / copy-paste)
  generate_report(raw) → build full structured trust report

Both analyze_* functions return the same raw dict shape so generate_report
works identically for both paths.

Bias Fix (v2.1)
---------------
  Previously used a weighted-average of Left/Center/Right probabilities
  which caused Left and Right signals to cancel out → always "Center".
  Now uses argmax (highest probability class wins) + confidence threshold.
  Labels "Uncertain" when max confidence < 45%.
"""

import torch
from transformers import AutoTokenizer
import config
from model import NewsTrustModel
from scraper import scrape_news
import re

import sys
import threading

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Global model/tokenizer cache (loaded once per process) ────────────────────
_tokenizer = None
_model     = None
_model_lock = threading.Lock()

# ── Label mappings ─────────────────────────────────────────────────────────────
BIAS_MAP = {0: "Left", 1: "Center", 2: "Right"}
INTENT_MAP = {0: "News", 1: "Satire"}
EMOTION_MAP = {0: "Joy", 1: "Sadness", 2: "Anger", 3: "Fear", 4: "Surprise", 5: "Disgust", 6: "Neutral"}

# ── Bias confidence threshold ──────────────────────────────────────────────────
# 3-class model: random baseline = 33%, so 45% is a meaningful signal
BIAS_CONFIDENCE_THRESHOLD = 0.45


# ── Indian political context keywords ───────────────────────────────────────────────
INDIA_KEYWORDS = [
    "modi", "bjp", "aap", "kejriwal", "rahul gandhi", "lok sabha",
    "rajya sabha", "india election", "ndtv", "the hindu", "narendra",
    "amit shah", "yogi", "mamata", "indian parliament", "ayodhya",
]


def load_model():
    """Load model + tokenizer once and cache globally in a thread-safe manner."""
    global _tokenizer, _model
    if _model is not None and _tokenizer is not None:
        return _tokenizer, _model

    with _model_lock:
        if _model is not None and _tokenizer is not None:
            return _tokenizer, _model

        print("Loading model and tokenizer (this only happens once)...")
        temp_tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_fast=False)
        temp_model     = NewsTrustModel(config.MODEL_NAME, dropout=0.0)  # dropout=0 at inference

        try:
            load_device = "cpu" if not torch.cuda.is_available() else config.DEVICE
            ckpt = torch.load(config.SAVE_PATH, map_location=load_device)
            # V1 checkpoint wraps state_dict inside "model_state_dict" key
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            else:
                state_dict = ckpt
            temp_model.load_state_dict(state_dict)
            print(f"✅ Loaded trained weights from {config.SAVE_PATH}")
        except Exception as e:
            print(f"⚠️  Could not load {config.SAVE_PATH}. Using untrained model. ({e})")

        temp_model.to(config.DEVICE)
        temp_model.eval()

        _tokenizer = temp_tokenizer
        _model     = temp_model

    return _tokenizer, _model


def _decode_bias(bias_logits: torch.Tensor) -> dict:
    """
    Decode 3-class bias logits into a human label + confidence.
    0 = Left | 1 = Center | 2 = Right
    """
    probs = torch.softmax(bias_logits, dim=0)
    p = [probs[i].item() for i in range(3)]

    winning_idx  = int(torch.argmax(probs).item())
    winning_prob = p[winning_idx]

    label = BIAS_MAP[winning_idx]
    if winning_prob < BIAS_CONFIDENCE_THRESHOLD:
        label = "Uncertain"

    # Position for BiasSlider: 0 (Left) → 50 (Center) → 100 (Right)
    position = round(p[0]*0 + p[1]*50 + p[2]*100, 1)

    return {
        "label":      label,
        "confidence": round(winning_prob * 100, 1),
        "probs": {
            "left":   round(p[0] * 100, 1),
            "center": round(p[1] * 100, 1),
            "right":  round(p[2] * 100, 1),
        },
        "position": position,
    }



def _run_model(text: str) -> dict:
    """
    Shared tokenise → forward → decode logic used by both analyze_url and analyze_text.
    Returns raw prediction dict (bias, factuality, intent, emotion).
    """
    tokenizer, model = load_model()

    w_ids, w_masks = NewsTrustModel.create_sliding_windows(text, tokenizer, config.MAX_LEN)

    with torch.inference_mode():
        pooled_embs = []
        for wid, wmask in zip(w_ids, w_masks):
            wid = wid.unsqueeze(0).to(config.DEVICE)
            wmask = wmask.unsqueeze(0).to(config.DEVICE)

            if config.DEVICE.type == "cuda":
                with torch.amp.autocast("cuda"):
                    outputs = model.backbone(input_ids=wid, attention_mask=wmask)
                    pooled = model._mean_pool(outputs.last_hidden_state, wmask)
            else:
                outputs = model.backbone(input_ids=wid, attention_mask=wmask)
                pooled = model._mean_pool(outputs.last_hidden_state, wmask)

            pooled_embs.append(pooled)

        avg_pooled = torch.stack(pooled_embs).mean(dim=0)

        if config.DEVICE.type == "cuda":
            with torch.amp.autocast("cuda"):
                bias_logits = model.bias_head(avg_pooled)
                fact_score = torch.sigmoid(model.fact_head(avg_pooled))
                intent_logits = model.intent_head(avg_pooled)
                emotion_logits = model.emotion_head(avg_pooled)
        else:
            bias_logits = model.bias_head(avg_pooled)
            fact_score = torch.sigmoid(model.fact_head(avg_pooled))
            intent_logits = model.intent_head(avg_pooled)
            emotion_logits = model.emotion_head(avg_pooled)

    # ── Bias: use new argmax decoder ──────────────────────────────────────────
    bias_obj = _decode_bias(bias_logits[0])

    intent_idx  = torch.argmax(intent_logits,  dim=1).item()
    
    # Bug 11: Multi-label emotion decoding (sigmoid + threshold)
    emo_probs = torch.sigmoid(emotion_logits[0])
    top_indices = torch.where(emo_probs > 0.3)[0]  # Threshold 0.3 per V1 Model Card
    
    # Primary = highest sigmoid score
    primary_idx = torch.argmax(emo_probs).item()
    primary_emotion = EMOTION_MAP.get(primary_idx, "Unknown")
    
    # All above threshold
    all_emotions = [EMOTION_MAP.get(idx.item(), "Unknown") for idx in top_indices]
    if not all_emotions:
        all_emotions = [primary_emotion]
        
    fact_val  = fact_score.item()

    return {
        "bias":       bias_obj,
        "factuality": round(fact_val, 4),
        "intent":     INTENT_MAP.get(intent_idx,  "Unknown"),
        "emotion":    primary_emotion,
        "all_emotions": all_emotions,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public: URL analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_url(url: str) -> dict:
    """
    Scrape a news article URL and run it through the model.
    Returns a raw prediction dict with metadata fields.
    """
    title, text, author, date = scrape_news(url)
    if not text:
        return {"error": "Could not extract article text from URL."}

    preds = _run_model(text)
    return {
        "title":  title  or "Unknown",
        "author": author or "Unknown",
        "date":   date   or "Unknown",
        "_source_text": text,
        **preds,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public: Raw text analysis  (WhatsApp / copy-paste)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text_metadata(text: str) -> dict:
    """
    Lightweight metadata extracted from raw text for WhatsApp messages.
    No ML needed — purely heuristic.
    """
    words      = text.split()
    word_count = len(words)
    read_secs  = max(1, word_count // 4)          # ~240 wpm → 4 words/sec
    minutes    = read_secs // 60
    seconds    = read_secs % 60
    read_time  = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"

    has_url    = bool(re.search(r"https?://\S+", text))
    is_forward = bool(re.search(
        r"\b(forwarded|fwd|share|forward this|pass this|please share)\b",
        text, re.IGNORECASE,
    ))
    # Detect likely non-English (Unicode chars outside Basic Latin + Latin-Extended)
    non_ascii  = sum(1 for c in text if ord(c) > 591)
    likely_non_english = (non_ascii / max(len(text), 1)) > 0.20

    # Extract numbers / percentages / claims
    numbers      = re.findall(r"\b\d[\d,]*(?:\.\d+)?%?\b", text)
    claims_count = len(numbers)

    return {
        "word_count":            word_count,
        "estimated_read_time":   read_time,
        "contains_url":          has_url,
        "looks_like_forward":    is_forward,
        "likely_non_english":    likely_non_english,
        "numeric_claims_count":  claims_count,
    }


def analyze_text(text: str) -> dict:
    """
    Analyse raw pasted text (WhatsApp forwards, copied articles, etc.)
    without any web scraping.
    """
    text = (text or "").strip()
    if len(text) < 50:
        return {"error": "Text is too short for meaningful analysis (minimum 50 characters)."}

    preds = _run_model(text)
    meta  = _extract_text_metadata(text)

    # Use first sentence / 120 chars as a pseudo-title
    first_sentence = re.split(r"[.!?\n]", text)[0].strip()
    pseudo_title   = first_sentence[:120] + ("…" if len(first_sentence) > 120 else "")

    return {
        "title":         pseudo_title or text[:80] + "…",
        "author":        "Unknown (raw text / WhatsApp)",
        "date":          "Unknown",
        "_source_text":  text,
        "text_metadata": meta,
        **preds,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Trust Score & Report
# ─────────────────────────────────────────────────────────────────────────────

EMOTION_CREDIBILITY = {
    "Joy":      0.80, "Sadness":  0.30, "Anger":    0.20,
    "Fear":     0.20, "Surprise": 0.55, "Disgust":  0.10,
    "Neutral":  0.65,
}

INTENT_CREDIBILITY = {"News": 1.0, "Satire": 0.0}
BIAS_CREDIBILITY   = {
    "Left":      0.5,
    "Center":    1.0,
    "Right":     0.5,
    "Uncertain": 0.4,
}



def compute_trust_score(factuality: float, bias: str,
                        intent: str, emotion: str) -> float:
    w = dict(fact=0.50, bias=0.20, intent=0.15, emotion=0.15)
    raw = (
        factuality                              * w["fact"]   +
        BIAS_CREDIBILITY.get(bias, 0.5)        * w["bias"]   +
        INTENT_CREDIBILITY.get(intent, 0.5)    * w["intent"] +
        EMOTION_CREDIBILITY.get(emotion, 0.5)  * w["emotion"]
    )
    return round(raw * 100, 1)


def _verdict(score: float) -> str:
    if score >= 80: return "Highly Credible"
    if score >= 60: return "Mostly Credible"
    if score >= 40: return "Mixed Credibility"
    if score >= 20: return "Low Credibility"
    return "Very Low Credibility"


def _bullet_findings(factuality: float, bias_obj: dict, intent: str,
                     emotion: str, score: float) -> list:
    """Generate bullet-point key findings for the report."""
    findings  = []
    fact_pct  = round(factuality * 100)
    bias      = bias_obj["label"]
    conf      = bias_obj["confidence"]
    probs     = bias_obj.get("probs", {})

    # Factuality
    if fact_pct >= 70:
        findings.append({"icon": "✅", "text": f"High factuality score ({fact_pct}%) — content appears well-sourced and objective.", "type": "good"})
    elif fact_pct >= 45:
        findings.append({"icon": "⚠️", "text": f"Moderate factuality ({fact_pct}%) — verify key claims independently before sharing.", "type": "warn"})
    else:
        findings.append({"icon": "🚨", "text": f"Low factuality score ({fact_pct}%) — content may be misleading, exaggerated or fabricated.", "type": "bad"})

    # Bias — with confidence info
    bias_msgs = {
        "Left":      {"icon": "◀️",  "text": f"Left-leaning political tone detected ({conf}% confidence).", "type": "warn"},
        "Center":    {"icon": "⚖️",  "text": f"Politically balanced / centre-leaning perspective ({conf}% confidence).", "type": "good"},
        "Right":     {"icon": "▶️",  "text": f"Right-leaning political tone detected ({conf}% confidence).", "type": "warn"},
        "Uncertain": {"icon": "❓",  "text": f"Political bias unclear — model confidence too low ({conf}%). Breakdown: Left: {probs.get('left') or 0}%, Center: {probs.get('center') or 0}%, Right: {probs.get('right') or 0}%.", "type": "warn"},
    }
    if bias in bias_msgs:
        findings.append(bias_msgs[bias])

    # Intent
    intent_msgs = {
        "News":    {"icon": "📰", "text": "Classified as straight news reporting — intended as factual journalism.", "type": "good"},
        "Satire":  {"icon": "🎭", "text": "Satire detected — this content is NOT intended to be taken as factual reporting.", "type": "bad"},
    }
    if intent in intent_msgs:
        findings.append(intent_msgs[intent])

    # Emotion
    emo_cred = EMOTION_CREDIBILITY.get(emotion, 0.5)
    if emo_cred >= 0.65:
        findings.append({"icon": "😌", "text": f"Calm emotional tone ('{emotion}') — associated with neutral, factual reporting.", "type": "good"})
    else:
        findings.append({"icon": "⚡", "text": f"Emotionally charged content ('{emotion}') — strong negative emotion may indicate sensationalism.", "type": "warn"})

    # Overall verdict
    v_type = "good" if score >= 60 else ("warn" if score >= 40 else "bad")
    findings.append({"icon": "🏆", "text": f"Overall trust score: {score}/100 — {_verdict(score)}", "type": v_type})

    return findings


def generate_report(raw: dict) -> dict:
    """
    Build a fully structured CredibleX trust report from raw model predictions.
    Works identically for URL and text analysis.
    """
    if "error" in raw:
        return raw

    factuality = raw.get("factuality", 0.5)
    bias_obj   = raw.get("bias", {"label": "Uncertain", "confidence": 0.0,
                                   "probs": {}, "position": 50})
    bias_label = bias_obj["label"]
    bias_conf  = bias_obj["confidence"]
    bias_probs = bias_obj.get("probs", {})
    bias_pos   = bias_obj.get("position", 50)

    intent  = raw.get("intent",  "News")
    emotion = raw.get("emotion", "neutral")

    score   = compute_trust_score(factuality, bias_label, intent, emotion)
    verdict = _verdict(score)

    fact_pct  = round(factuality * 100, 1)
    bias_cred = BIAS_CREDIBILITY.get(bias_label, 0.5)
    int_cred  = INTENT_CREDIBILITY.get(intent, 0.5)
    emo_cred  = EMOTION_CREDIBILITY.get(emotion, 0.5)

    # Detect Indian political context from raw text metadata
    source_text = raw.get("_source_text", "").lower()
    is_indian_context = any(kw in source_text for kw in INDIA_KEYWORDS)
    indian_note = (
        " \u26a0\ufe0f Indian political context detected — predictions reflect training on "
        "Indian political news data."
        if is_indian_context else ""
    )

    # Bias explanation with raw probabilities
    probs_str = ", ".join(f"{k.replace('_',' ').title()}: {v}%" for k, v in bias_probs.items())
    if bias_label == "Uncertain":
        bias_explanation = (
            f"Model confidence was too low ({bias_conf}%) to determine a clear bias. "
            f"Raw probabilities — {probs_str}. "
            f"This may indicate a genuinely balanced article or an out-of-domain source.{indian_note}"
        )
    elif bias_label == "Center":
        bias_explanation = (
            f"Political bias detected as '{bias_label}' with {bias_conf}% confidence. "
            f"({probs_str}). "
            f"Centre-leaning sources tend to present more balanced perspectives.{indian_note}"
        )
    else:
        bias_explanation = (
            f"Political bias detected as '{bias_label}' with {bias_conf}% confidence. "
            f"({probs_str}). "
            f"A {bias_label.lower()}-leaning source introduces perspective bias, "
            f"reducing the trust score.{indian_note}"
        )

    dimensions = {
        "factuality": {
            "value":            factuality,
            "contribution_pct": round(factuality * 0.50 * 100, 1),
            "weight":           "50%",
            "explanation": (
                f"The factuality regression head scored this content at {fact_pct}%. " +
                ("High factuality suggests objective, well-sourced reporting."
                 if factuality >= 0.6 else
                 "Low factuality suggests the content may be exaggerated or unreliable.")
            ),
        },
        "bias": {
            "value":            bias_label,
            "position":         bias_pos,
            "confidence":       bias_conf,
            "probs":            bias_probs,
            "contribution_pct": round(bias_cred * 0.20 * 100, 1),
            "weight":           "20%",
            "explanation":      bias_explanation,
        },
        "intent": {
            "value":            intent,
            "contribution_pct": round(int_cred * 0.15 * 100, 1),
            "weight":           "15%",
            "explanation": (
                f"Content intent classified as '{intent}'. " +
                {"News":    "Straight news reporting receives full intent credit.",
                 "Opinion": "Opinion/editorial content receives partial credit — it represents a personal viewpoint.",
                 "Satire":  "Satire receives zero intent credit; it is not intended to be taken as factual."
                }.get(intent, "")
            ),
        },
        "emotion": {
            "value":            emotion,
            "contribution_pct": round(emo_cred * 0.15 * 100, 1),
            "weight":           "15%",
            "explanation": (
                f"Dominant emotion detected as '{emotion}' "
                f"(credibility proxy: {round(emo_cred * 100)}%). " +
                ("Neutral or positive emotions are associated with more factual reporting."
                 if emo_cred >= 0.6 else
                 "Strongly negative emotions (anger, fear, disgust) are often linked to sensationalism.")
            ),
        },
    }

    summary = (
        f"This content scores {score}/100 ({verdict}): "
        f"factuality at {fact_pct}%, "
        f"{bias_label.lower()}-leaning political tone ({bias_conf}% confidence), "
        f"classified as {intent.lower()} content, "
        f"with a dominant '{emotion}' emotional signal.{indian_note}"
    )

    return {
        "score":         score,
        "verdict":       verdict,
        "dimensions":    dimensions,
        "key_findings":  _bullet_findings(factuality, bias_obj, intent, emotion, score),
        "text_metadata": raw.get("text_metadata"),
        "metadata": {
            "title":  raw.get("title",  "Unknown"),
            "author": raw.get("author", "Unknown"),
            "date":   raw.get("date",   "Unknown"),
        },
        "summary": summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_urls = [
        "https://www.bbc.com/news/articles/c4gzl5p8zpvo",
        "https://www.reuters.com/business/finance/us-banks-brace-more-commercial-real-estate-pain-2023-11-20/",
        "https://mises.org/mises-wire/federal-reserve-inflation-engine",
    ]
    for url in test_urls:
        print(f"\n{'='*70}\nURL: {url}")
        raw    = analyze_url(url)
        report = generate_report(raw)
        print(f"  Score  : {report.get('score')} / 100  →  {report.get('verdict')}")
        print(f"  Summary: {report.get('summary')}")

    # Test WhatsApp text
    print(f"\n{'='*70}\nWhatsApp text test:")
    wa_text = ("Breaking: The government has secretly imposed emergency rule! "
               "All mobile networks to be shut down at midnight. Forward this to everyone NOW! "
               "Source: a reliable insider. Don't let them silence us!!")
    raw    = analyze_text(wa_text)
    report = generate_report(raw)
    print(f"  Score  : {report.get('score')} / 100  →  {report.get('verdict')}")
    for f in report.get("key_findings", []):
        print(f"    {f['icon']} {f['text']}")