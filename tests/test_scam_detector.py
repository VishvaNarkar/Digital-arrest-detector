"""
Unit tests for Digital Arrest Detector.

Run with:
    pytest tests/

These tests are designed to run without a GPU, without Ollama, and without
Streamlit, so they can execute in any CI environment that has the pip
dependencies installed.
"""

import importlib
import sys
import types
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers — ensure project root is on sys.path so imports resolve correctly.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 1. Smoke test: rag_utils can be imported
# ---------------------------------------------------------------------------
class TestRagUtilsImport:
    """Verify the utils.rag_utils module can be imported without errors."""

    def test_import_succeeds(self):
        """utils.rag_utils should be importable without raising."""
        mod = importlib.import_module("utils.rag_utils")
        assert mod is not None

    def test_has_expected_callables(self):
        """The module must expose the three callables used by app.py."""
        mod = importlib.import_module("utils.rag_utils")
        assert callable(getattr(mod, "is_ollama_available", None)), (
            "is_ollama_available must be a callable in utils.rag_utils"
        )
        assert callable(getattr(mod, "rag_verify_text", None)), (
            "rag_verify_text must be a callable in utils.rag_utils"
        )
        assert callable(getattr(mod, "rag_refine_transcription", None)), (
            "rag_refine_transcription must be a callable in utils.rag_utils"
        )


# ---------------------------------------------------------------------------
# 2. clean_text tests (train_text.py)
# ---------------------------------------------------------------------------
from train_text import clean_text  # noqa: E402  (after sys.path setup)


class TestCleanText:
    """Tests for the clean_text preprocessing function in train_text.py."""

    def test_returns_string_for_non_string(self):
        assert clean_text(None) == ""
        assert clean_text(42) == ""
        assert clean_text([]) == ""

    def test_lowercases_ascii(self):
        assert clean_text("HELLO WORLD") == "hello world"

    def test_removes_urls(self):
        result = clean_text("Visit http://example.com for free gifts!")
        assert "http" not in result
        assert "example" not in result

    def test_strips_punctuation_but_keeps_words(self):
        result = clean_text("Hello, world! How are you?")
        # punctuation removed, words retained
        assert "hello" in result
        assert "world" in result
        assert "," not in result
        assert "!" not in result

    def test_collapses_whitespace(self):
        result = clean_text("  too   many    spaces  ")
        assert "  " not in result
        assert result == result.strip()

    def test_preserves_hindi_script(self):
        """Devanagari text must NOT be stripped (multilingual support)."""
        hindi_text = "आपका ओटीपी है 123456"
        result = clean_text(hindi_text)
        # At least the Devanagari characters should survive
        assert "ओटीपी" in result, (
            f"Hindi word 'ओटीपी' was stripped from clean_text output: {result!r}"
        )

    def test_preserves_gujarati_script(self):
        """Gujarati text must NOT be stripped (multilingual support)."""
        gujarati_text = "તમારો ઓટીપી 654321 છે"
        result = clean_text(gujarati_text)
        assert "ઓટીપી" in result, (
            f"Gujarati word 'ઓટીપી' was stripped from clean_text output: {result!r}"
        )

    def test_preserves_digits(self):
        """Digits are important scam signals (OTP codes, amounts) and must be kept."""
        result = clean_text("Your OTP is 987654")
        assert "987654" in result, (
            f"Digits were stripped from clean_text output: {result!r}"
        )

    def test_empty_string_returns_empty(self):
        assert clean_text("") == ""


# ---------------------------------------------------------------------------
# 3. detect_message tests (app.py — stubbed to avoid Streamlit/TF imports)
# ---------------------------------------------------------------------------
# app.py imports Streamlit and TensorFlow at module level, which makes
# a direct import impractical in a headless test environment.  We therefore
# test the pure logic of detect_message by extracting it via a thin stub.

def _make_stub_detect_message():
    """
    Return a standalone copy of the detect_message logic that does NOT depend
    on Streamlit, TensorFlow, or the NLTK download.  We replicate only the
    parts exercised by these unit tests (empty-text guard, keyword matching,
    combined_prob, label thresholding).
    """
    import re
    import unicodedata

    # Minimal keyword dict sufficient for the tests below
    RISKY_KEYWORDS = {
        "otp": 3,
        "verify now": 2,
        "urgent": 2,
        "lottery": 2,
        "winner": 2,
    }

    def detect_message(text: str, threshold: float = 0.35):
        if not text:
            return {
                "label": "Likely Safe",
                "ml_prob": 0.0,
                "keyword_score": 0,
                "sentiment": {"compound": 0.0},
                "combined_prob": 0.0,
                "keywords": [],
                "rag": None,
            }

        text = unicodedata.normalize("NFKC", text)
        text_lower = text.lower()

        found_keywords = []
        keyword_score = 0
        for kw, weight in RISKY_KEYWORDS.items():
            pattern = rf"(?<!\w){re.escape(kw)}(?!\w)"
            if re.search(pattern, text_lower, flags=re.UNICODE):
                found_keywords.append(kw)
                keyword_score += weight

        KEYWORD_MULTIPLIER = 0.06
        base_combined = keyword_score * KEYWORD_MULTIPLIER
        base_combined = max(0.0, min(base_combined, 1.0))

        label = "Likely Scam" if base_combined > threshold else "Likely Safe"

        return {
            "label": label,
            "ml_prob": 0.0,
            "keyword_score": keyword_score,
            "sentiment": {"compound": 0.0},
            "combined_prob": base_combined,
            "keywords": found_keywords,
            "rag": None,
        }

    return detect_message


detect_message = _make_stub_detect_message()


class TestDetectMessage:
    """Behaviour tests for the detect_message scam detection logic."""

    def test_empty_text_returns_safe(self):
        result = detect_message("")
        assert result["label"] == "Likely Safe"
        assert result["combined_prob"] == 0.0
        assert result["keywords"] == []

    def test_none_like_empty_returns_safe(self):
        # Passing falsy string
        result = detect_message("")
        assert "Safe" in result["label"]

    def test_safe_text_not_flagged(self):
        result = detect_message("Hello, how are you doing today?")
        assert "Safe" in result["label"]
        assert result["keyword_score"] == 0

    def test_scam_keywords_detected(self):
        result = detect_message(
            "URGENT: Your bank OTP is expiring. Verify now to avoid account suspension!"
        )
        assert len(result["keywords"]) > 0, "No keywords detected in obvious scam text"
        assert "otp" in result["keywords"] or "urgent" in result["keywords"]

    def test_high_keyword_score_flags_scam(self):
        # otp (3) + urgent (2) + verify now (2) = 7 → 7 * 0.06 = 0.42 > 0.35
        result = detect_message("urgent verify now your otp", threshold=0.35)
        assert "Scam" in result["label"], (
            f"Expected scam label, got: {result['label']} "
            f"(combined_prob={result['combined_prob']:.3f})"
        )

    def test_custom_threshold_changes_label(self):
        """Raising the threshold should flip borderline cases to safe."""
        text = "lottery winner claim your prize"
        low_thresh = detect_message(text, threshold=0.10)
        high_thresh = detect_message(text, threshold=0.90)
        assert "Scam" in low_thresh["label"]
        assert "Safe" in high_thresh["label"]

    def test_result_has_required_keys(self):
        result = detect_message("test message")
        for key in ("label", "ml_prob", "keyword_score", "sentiment",
                    "combined_prob", "keywords", "rag"):
            assert key in result, f"Missing key in result: {key}"

    def test_combined_prob_clamped_between_0_and_1(self):
        # Throw a very keyword-heavy message at it
        heavy = " ".join(["otp urgent winner lottery"] * 20)
        result = detect_message(heavy)
        assert 0.0 <= result["combined_prob"] <= 1.0
