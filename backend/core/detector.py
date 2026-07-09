"""Text scam detection — extracted from app_streamlit.py, Streamlit-free."""
import json
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from backend.core.keywords import RISKY_KEYWORDS

# ── NLTK setup ───────────────────────────────────────────────────────────────
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

_sent_analyzer = SentimentIntensityAnalyzer()


# ── Public API ────────────────────────────────────────────────────────────────

def detect_message(text: str, threshold: float = 0.35) -> Dict:
    """
    Hybrid scam detection:
      1. Keyword scoring (weighted)
      2. VADER sentiment boost
      3. TF-IDF Logistic Regression probability
      4. Optional LLM verification via Ollama/RAG

    Returns a dict with label, ml_prob, keyword_score, sentiment,
    combined_prob, keywords, and optional rag result.
    """
    if not text:
        return {
            "label": "Likely Safe",
            "ml_prob": 0.0,
            "keyword_score": 0,
            "sentiment": {"compound": 0.0, "neg": 0.0, "pos": 0.0, "neu": 1.0},
            "combined_prob": 0.0,
            "keywords": [],
            "rag": None,
        }

    # Normalise
    text = unicodedata.normalize("NFKC", text)
    text_lower = text.lower()

    # Keyword scoring
    found_keywords: List[str] = []
    keyword_score: int = 0
    for kw, weight in RISKY_KEYWORDS.items():
        pattern = rf"(?<!\w){re.escape(kw)}(?!\w)"
        if re.search(pattern, text_lower, flags=re.UNICODE):
            found_keywords.append(kw)
            keyword_score += weight

    # URL / Phishing Link Analysis
    max_url_risk = 0.0
    try:
        from backend.core.url_analyzer import extract_urls, analyze_url
        urls = extract_urls(text)
        for url in urls:
            url_res = analyze_url(url, text)
            max_url_risk = max(max_url_risk, url_res["risk_score"])
            found_keywords.extend(url_res["flags"])
    except Exception:
        pass

    # Sentiment
    sentiment = _sent_analyzer.polarity_scores(text)
    neg_score = sentiment.get("neg", 0.0)
    compound = sentiment.get("compound", 0.0)
    sentiment_boost = 0.1 if neg_score > 0.3 or compound < -0.2 else 0.0

    # ML model
    ml_prob = 0.0
    try:
        from backend.core import models as _m
        if _m.text_model and _m.vectorizer:
            vec = _m.vectorizer.transform([text])
            ml_prob = float(_m.text_model.predict_proba(vec)[0][1])
    except Exception:
        ml_prob = 0.0

    # Heuristic combination
    KEYWORD_MULTIPLIER = 0.06
    base_combined = ml_prob + (keyword_score * KEYWORD_MULTIPLIER) + sentiment_boost
    # Boost combination with URL risk
    if max_url_risk > 0.0:
        base_combined = max(base_combined, max_url_risk)
    base_combined = max(0.0, min(base_combined, 1.0))

    # Optional RAG / LLM
    rag_result = None
    llm_prob: Optional[float] = None
    try:
        from utils.rag_utils import is_ollama_available, rag_verify_text
        if is_ollama_available():
            rag_result = rag_verify_text(
                text=text, ml_prob=ml_prob,
                keywords=found_keywords, sentiment=sentiment
            )
            raw_rp = rag_result.get("risk_percent")
            if raw_rp is not None:
                llm_prob = float(raw_rp) / 100.0
            elif rag_result.get("llm_prob") is not None:
                llm_prob = float(rag_result["llm_prob"])
    except Exception:
        rag_result = None

    # Weighted combine with LLM
    if llm_prob is not None:
        combined_prob = (llm_prob * 0.55) + (base_combined * 0.45)
    else:
        combined_prob = base_combined

    # Baseline Calibration:
    # If the combined probability is below the threshold and no suspicious keywords or URLs
    # are detected, scale it down to suppress ML intercept noise (e.g. 34% baseline).
    if combined_prob < threshold and keyword_score == 0 and max_url_risk == 0.0:
        combined_prob = combined_prob * 0.15

    combined_prob = max(0.0, min(combined_prob, 1.0))

    label = "Likely Scam" if combined_prob > threshold else "Likely Safe"

    return {
        "label": label,
        "ml_prob": ml_prob,
        "keyword_score": keyword_score,
        "sentiment": sentiment,
        "combined_prob": combined_prob,
        "keywords": found_keywords,
        "rag": rag_result,
    }


def categorize_scam(keywords: List[str], sentiment: Dict) -> str:
    """Return a human-readable scam category label."""
    kws = [k.lower() for k in keywords]
    compound = sentiment.get("compound", 0.0)

    if any(k in kws for k in ["lottery", "winner", "prize", "gift", "reward"]):
        return "Reward / Lottery Scam"
    if any(k in kws for k in ["bank", "account", "otp", "password", "verify",
                               "transaction"]):
        return "Banking / Verification Scam"
    if compound < -0.3:
        return "Fear / Threat-Based Scam"
    return "Unknown / Generic Scam"
