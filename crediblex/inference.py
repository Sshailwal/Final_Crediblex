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
BIAS_MAP   = {0: "Left", 1: "Center", 2: "Right"}
INTENT_MAP = {0: "News", 1: "Opinion", 2: "Satire"}
EMOTION_MAP = {
    0: "admiration", 1: "amusement",   2: "anger",       3: "annoyance",
    4: "approval",   5: "caring",      6: "confusion",   7: "curiosity",
    8: "desire",     9: "disappointment", 10: "disapproval", 11: "disgust",
    12: "embarrassment", 13: "excitement", 14: "fear",    15: "gratitude",
    16: "grief",     17: "joy",        18: "love",        19: "nervousness",
    20: "optimism",  21: "pride",      22: "realization", 23: "relief",
    24: "remorse",   25: "sadness",    26: "surprise",    27: "neutral",
}

# ── Bias confidence threshold ──────────────────────────────────────────────────
# If the winning bias class probability is below this, label as "Uncertain"
BIAS_CONFIDENCE_THRESHOLD = 0.45


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
            temp_model.load_state_dict(
                torch.load(config.SAVE_PATH, map_location=config.DEVICE)
            )
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
    Decode bias logits into a label + confidence using argmax.

    FIX v2.1: Replaced weighted-average approach with argmax so that
    Left and Right signals no longer cancel each other out.

    Returns
    -------
    dict with keys:
        label       — "Left" / "Slightly Left" / "Center" /
                      "Slightly Right" / "Right" / "Uncertain"
        confidence  — float 0–1, probability of winning class
        probs       — dict with raw Left / Center / Right probabilities
        position    — float 0–100 for BiasSlider (0=Left, 50=Center, 100=Right)
    """
    probs = torch.softmax(bias_logits, dim=0)
    p_left   = probs[0].item()
    p_center = probs[1].item()
    p_right  = probs[2].item()

    # Argmax: winning class is the one with highest probability
    winning_idx  = torch.argmax(probs).item()
    winning_prob = probs[winning_idx].item()
    base_label   = BIAS_MAP[winning_idx]   # "Left", "Center", or "Right"

    # Low confidence → mark as Uncertain
    if winning_prob < BIAS_CONFIDENCE_THRESHOLD:
        label = "Uncertain"
    elif base_label == "Left":
        label = "Mostly Left"   if winning_prob >= 0.65 else "Slightly Left"
    elif base_label == "Right":
        label = "Mostly Right"  if winning_prob >= 0.65 else "Slightly Right"
    else:
        label = "Center"

    # Position for BiasSlider: Left=0, Center=50, Right=100
    position = round(p_left * 0 + p_center * 50 + p_right * 100, 1)

    return {
        "label":      label,
        "confidence": round(winning_prob * 100, 1),
        "probs": {
            "left":   round(p_left   * 100, 1),
            "center": round(p_center * 100, 1),
            "right":  round(p_right  * 100, 1),
        },
        "position": position,
    }


def _run_model(text: str) -> dict:
    """
    Shared tokenise → forward → decode logic used by both analyze_url and analyze_text.
    Returns raw prediction dict (bias, factuality, intent, emotion).
    """
    tokenizer, model = load_model()

    encoding = tokenizer(
        [text[:config.MAX_LEN * 6]],   # pre-truncate before tokeniser for memory safety
        max_length     = config.MAX_LEN,
        padding        = "longest",
        truncation     = True,
        return_tensors = "pt",
    )
    input_ids      = encoding["input_ids"].to(config.DEVICE)
    attention_mask = encoding["attention_mask"].to(config.DEVICE)

    with torch.no_grad():
        if config.DEVICE == "cuda":
            with torch.amp.autocast("cuda"):
                outputs = model(input_ids, attention_mask)
        else:
            outputs = model(input_ids, attention_mask)

    # ── Bias: use new argmax decoder ──────────────────────────────────────────
    bias_obj = _decode_bias(outputs["bias"][0])

    intent_idx  = torch.argmax(outputs["intent"],  dim=1).item()
    emotion_idx = torch.argmax(outputs["emotion"], dim=1).item()
    fact_score  = outputs["factuality"].item()

    return {
        "bias":       bias_obj,
        "factuality": round(fact_score, 4),
        "intent":     INTENT_MAP.get(intent_idx,  "Unknown"),
        "emotion":    EMOTION_MAP.get(emotion_idx, "Unknown"),
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
        "text_metadata": meta,
        **preds,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Trust Score & Report
# ─────────────────────────────────────────────────────────────────────────────

EMOTION_CREDIBILITY = {
    "admiration": 0.80, "amusement": 0.70, "anger":     0.20,
    "annoyance":  0.30, "approval":  0.85, "caring":    0.75,
    "confusion":  0.50, "curiosity": 0.65, "desire":    0.60,
    "disappointment": 0.30, "disapproval": 0.20, "disgust": 0.10,
    "embarrassment":  0.35, "excitement":  0.70, "fear":   0.20,
    "gratitude":  0.90, "grief":     0.25, "joy":       0.80,
    "love":       0.80, "nervousness": 0.30, "optimism": 0.70,
    "pride":      0.70, "realization": 0.60, "relief":   0.75,
    "remorse":    0.30, "sadness":   0.30, "surprise":   0.55,
    "neutral":    0.65,
}

INTENT_CREDIBILITY = {"News": 1.0, "Opinion": 0.5, "Satire": 0.0}
BIAS_CREDIBILITY   = {
    "Mostly Left":    0.0,
    "Slightly Left":  0.5,
    "Center":         1.0,
    "Slightly Right": 0.5,
    "Mostly Right":   0.0,
    "Uncertain":      0.4,   # small penalty for uncertain bias
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
        "Mostly Left":    {"icon": "◀️",  "text": f"Strong left-leaning political tone detected ({conf}% confidence).", "type": "warn"},
        "Slightly Left":  {"icon": "◁",   "text": f"Slight left-leaning perspective detected ({conf}% confidence).", "type": "info"},
        "Center":         {"icon": "⚖️",  "text": f"Politically balanced / centre-leaning perspective ({conf}% confidence).", "type": "good"},
        "Slightly Right": {"icon": "▷",   "text": f"Slight right-leaning perspective detected ({conf}% confidence).", "type": "info"},
        "Mostly Right":   {"icon": "▶️",  "text": f"Strong right-leaning political tone detected ({conf}% confidence).", "type": "warn"},
        "Uncertain":      {"icon": "❓",  "text": f"Political bias unclear — model confidence too low ({conf}%). Left: {probs.get('left')}%, Center: {probs.get('center')}%, Right: {probs.get('right')}%.", "type": "warn"},
    }
    if bias in bias_msgs:
        findings.append(bias_msgs[bias])

    # Intent
    intent_msgs = {
        "News":    {"icon": "📰", "text": "Classified as straight news reporting — intended as factual journalism.", "type": "good"},
        "Opinion": {"icon": "💬", "text": "Opinion / editorial content — represents a personal or editorial viewpoint, not hard fact.", "type": "warn"},
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
                                   "probs": {"left": 33, "center": 34, "right": 33},
                                   "position": 50})
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

    # Bias explanation with raw probabilities
    if bias_label == "Uncertain":
        bias_explanation = (
            f"Model confidence was too low ({bias_conf}%) to determine a clear bias. "
            f"Raw probabilities — Left: {bias_probs.get('left')}%, "
            f"Center: {bias_probs.get('center')}%, "
            f"Right: {bias_probs.get('right')}%. "
            f"This may indicate a genuinely balanced article or an out-of-domain source."
        )
    elif bias_label == "Center":
        bias_explanation = (
            f"Political bias detected as '{bias_label}' with {bias_conf}% confidence. "
            f"(Left: {bias_probs.get('left')}%, Center: {bias_probs.get('center')}%, Right: {bias_probs.get('right')}%). "
            f"Centre-leaning sources tend to present more balanced perspectives."
        )
    else:
        bias_explanation = (
            f"Political bias detected as '{bias_label}' with {bias_conf}% confidence. "
            f"(Left: {bias_probs.get('left')}%, Center: {bias_probs.get('center')}%, Right: {bias_probs.get('right')}%). "
            f"A {bias_label.lower()}-leaning source introduces perspective bias, reducing the trust score."
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
        f"with a dominant '{emotion}' emotional signal."
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