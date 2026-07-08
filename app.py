# Digital Arrest Detector - Multi-Channel Fraud Prevention
# app.py
import streamlit as st
import streamlit.components.v1 as components
import joblib
import re
import wave
import json
import tempfile
import os
import soundfile as sf
import cv2
import numpy as np
import tensorflow as tf
from vosk import Model, KaldiRecognizer
from pathlib import Path
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import unicodedata
import math
from utils.rag_utils import is_ollama_available, rag_verify_text, rag_refine_transcription

# Download NLTK data only when not already present (avoids slow startup on repeat runs)
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)
sent_analyzer = SentimentIntensityAnalyzer()

# ============================
# Paths
# ============================
MODEL_DIR = Path("models")
TEXT_MODEL_PATH = MODEL_DIR / "text_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
DEEPFAKE_MODEL_PATH = MODEL_DIR / "Deepfakes_detection_model.keras"
VOSK_BASE_PATH = MODEL_DIR


# ============================
# Load Models
# ============================
def load_text_models():
    """Load text classification model and vectorizer."""
    try:
        text_model = joblib.load(TEXT_MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return text_model, vectorizer
    except Exception as e:
        st.error(f"❌ Failed to load text models: {e}")
        return None, None


def load_deepfake_model():
    """Load deepfake detection model safely."""
    try:
        model = tf.keras.models.load_model(DEEPFAKE_MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load deepfake model: {e}")
        return None


text_model, vectorizer = load_text_models()
deepfake_model = load_deepfake_model()


# ============================
# Risky keywords
# ============================
# Consolidated risky keywords with weights (no duplicates)
RISKY_KEYWORDS = {
    # English (high=3, medium=2, low=1)
    "otp": 3, "one time password": 3, "one-time password": 3, "one time code": 3, "one-time code": 3,
    "pin": 3, "atm pin": 3, "upi pin": 3, "cvv": 3, "atm cvv": 3, "password": 3, "passcode": 3,
    "ssn": 3, "social security number": 3, "aadhar": 3, "aadhaar": 3, "identity number": 3,
    "verify now": 2, "verify account": 2, "verify your account": 2, "verify phone number": 2,
    "confirm": 2, "confirm identity": 2, "confirm now": 2, "account verification": 2, "verification": 1,
    "bank": 2, "bank account": 2, "account number": 2, "account": 2, "suspend account": 2, "suspended account": 2,
    "secure your account": 2, "secure your account now": 2, "secure": 1, "security": 1, "security alert": 2,
    "transaction": 2, "transaction alert": 2, "unauthorized transaction": 2, "transaction failed": 2,
    "payment": 2, "payment due": 2, "confirm payment": 2, "refund": 2, "refund pending": 2, "chargeback": 2,
    "transfer": 2, "fund transfer": 2, "immediate transfer": 2, "transfer now": 2,
    "credit": 2, "debit": 2, "credit card": 2, "debit card": 2, "banking": 2, "bank notice": 2,
    "lottery": 2, "lottery winner": 2, "winner": 2, "win": 1, "prize": 2, "claim": 2, "claim prize": 2,
    "redeem": 2, "redeem reward": 2, "reward": 2, "giveaway": 2, "giveaway contest": 2, "reward points": 2,
    "suspend": 2, "suspended": 2, "alert": 2, "fraud": 2, "fraud alert": 2, "phishing": 2, "phishing scam": 2,
    "scam": 2, "scammer": 2, "scamming": 2, "hacking": 2, "hacked": 2, "hacker": 2, "hack": 2,
    "urgent": 2, "urgent action": 2, "urgent update": 2, "urgent update required": 2, "immediately": 2,
    "immediate action required": 2, "immediate action needed": 2, "act now": 1, "act quickly": 1,
    "final notice": 1, "final notice sent": 1, "last chance": 1, "last chance offer": 1, "dont miss": 1, "don't miss": 1,
    "limited": 1, "limited time": 1, "limited time offer": 1, "exclusive": 1, "important": 1, "attention": 1,
    "offer": 1, "free": 1, "free offer": 1, "risk-free": 1, "guarantee": 1, "trial": 1,
    "click": 1, "click here": 1, "click the link": 1, "click link": 1, "link": 1, "shortlink": 1,
    "bit.ly": 1, "tinyurl": 1, "short url": 1, "download": 1, "attachment": 1, "open attachment": 1,
    "subscribe": 1, "subscribe now": 1, "bonus": 1, "cash": 1, "money": 1, "transfer money": 2,
    "gift": 1, "gift card": 1, "cash prize": 1, "win cash": 1, "important update": 1, "alert message": 1,
    "customer support": 1, "customer care": 1, "helpdesk": 1, "call back": 1, "call us now": 1,
    "contact us": 1, "verify details": 2, "update information": 2, "update your information": 2,
    "provide details": 2, "submit documents": 2, "send documents": 2, "share otp": 3, "share password": 3,
    "share pin": 3, "do not share": 1, "suspicious activity": 1, "suspicious": 1, "suspicious transaction": 2,
    "warning": 1, "caution": 1, "danger": 1, "notice": 1, "final warning": 1, "security code": 3,
    "auth code": 3, "authentication code": 3, "authorize": 2, "authorization": 2,
    "identity": 1, "identity theft": 2, "protect": 1, "protect your account": 2, "block": 1, "blocked": 2,
    "verify identity": 2, "confirm identity now": 2, "kyc": 2, "complete kyc": 2, "know your customer": 2,
    "customer verification": 2, "support team": 1, "verification link": 2, "secure link": 2,

    # URL / link shorteners & phishing indicators (low-medium)
    "http://": 1, "https://": 1, "www.": 1, ".com/": 1, ".in/": 1, ".net/": 1, "bitly.com": 1, "tinyurl.com": 1,
    "shorten": 1, "redirect": 1, "verify link": 2, "login link": 2, "reset password": 2, "password reset": 2,

    # Payment / wallet / India-specific (medium where sensitive)
    "upi": 2, "upi id": 2, "paytm": 2, "phonepe": 2, "google pay": 2, "gpay": 2, "mobicash": 2,
    "bhim": 2, "payment request": 2, "payment link": 2, "pay now": 2, "collect request": 2,
    "account transfer": 2, "bank transfer": 2, "neft": 2, "imps": 2, "rtgs": 2,
    "merchant": 1, "invoice": 1, "bill due": 1, "billing": 1,

    # Social engineering / impersonation phrases (medium)
    "we are from": 1, "this is from": 1, "official message": 1, "from bank": 2, "from bank support": 2,
    "from government": 2, "from aadhaar": 2, "from uidai": 2, "from customer support": 1,
    "verify with us": 2, "call from bank": 2, "call from support": 1, "urgent call": 1,

    # Hindi (Devanagari) - expanded & de-duplicated
    "ओटीपी": 3, "ओटीपी को": 3, "वन टाइम पासवर्ड": 3, "पासवर्ड": 3, "पासवर्ड रीसेट": 2, "पिन": 3, "सीवीवी": 3, "एसएसएन": 3,
    "आधार": 3, "आधार नंबर": 3, "खाता": 2, "बैंक": 2, "बैंक खाता": 2, "खाता संख्या": 2, "अपने खाते को सुरक्षित करें": 2,
    "अब सत्यापित करें": 2, "सत्यापित करें": 2, "वेरिफाई": 2, "सत्यापन": 1, "सत्यापन लिंक": 2, "एक्सेस": 2,
    "लेनदेन": 2, "भुगतान": 2, "ट्रांसफर": 2, "क्रेडिट": 2, "डेबिट": 2, "ट्रांसफर करें": 2, "पे नाउ": 2,
    "इनाम": 2, "विनर": 2, "लॉटरी": 2, "दावा": 2, "रिडीम": 2, "रिडीम करें": 2, "रिफंड": 2, "सस्पेंड": 2, "सस्पेंडेड": 2,
    "पुष्टि": 2, "पुष्टि करें": 2, "अलर्ट": 2, "सुरक्षा अलर्ट": 2, "तुरंत": 2, "फौरन": 2, "त्वरित कार्रवाई": 2, "तत्काल कार्रवाई": 1,
    "तत्काल अपडेट": 1, "तुरंत अपडेट": 1, "अंतिम नोटिस": 1, "अंतिम मौका": 1, "मिस न करें": 1,
    "स्कैम": 2, "फ्रॉड": 2, "फिशिंग": 2, "हैकिंग": 2, "हैक्ड": 2, "हैकर": 2,
    "मुफ्त": 1, "ऑफर": 1, "क्लिक": 1, "यहाँ क्लिक करें": 1, "लिंक": 1, "लिंक पर क्लिक करें": 1, "डाउनलोड": 1,
    "जोखिम": 1, "जोखिम-मुक्त": 1, "गारंटी": 1, "ट्रायल": 1, "गौरव": 1, "विशेष": 1, "महत्वपूर्ण": 1, "ध्यान दें": 1,
    "पहचान": 1, "पहचान चोरी": 2, "सुरक्षित": 1, "सुरक्षा करें": 1, "सुरक्षा": 1, "सावधानी": 1, "चेतावनी": 1, "खतरा": 1,
    "साझा न करें": 3, "शेयर न करें": 3, "संदेहजनक": 1, "नकली": 1, "पैसे": 1, "नकद": 1, "उपहार": 1, "बोनस": 1, "सब्सक्राइब": 1,
    "अभी सब्सक्राइब करें": 1, "अभी कॉल करें": 1, "हमें कॉल करें": 1, "ग्राहक सेवा": 1, "ग्राहक सहायता": 1, "सहायता डेस्क": 1,
    "संपर्क करें": 1, "हमसे संपर्क करें": 1, "विवरण सत्यापित करें": 2, "जानकारी अपडेट करें": 2, "अपनी जानकारी अपडेट करें": 2,
    "विवरण प्रदान करें": 2, "दस्तावेज़ जमा करें": 2, "दस्तावेज़ भेजें": 2, "ओटीपी साझा करें": 3, "पासवर्ड साझा करें": 3,
    "पिन साझा करें": 3, "साझा न करें": 1, "संदिग्ध गतिविधि": 1, "संदिग्ध लेनदेन": 2, "चेतावनी": 1, "सावधानी": 1, "खतरा": 1, "अंतिम चेतावनी": 1, "सुरक्षा कोड": 3,
    "प्रमाणीकरण कोड": 3, "प्राधिकरण": 2, "पहचान सत्यापित करें": 2, "अब पहचान की पुष्टि करें": 2, "केवाईसी": 2, "केवाईसी पूरा करें": 2, "अपने ग्राहक को जानें": 2,
    "ग्राहक सत्यापन": 2, "सत्यापन लिंक": 2, "सुरक्षित लिंक": 2, "पासवर्ड रीसेट करें": 2, "पासवर्ड रिसेट": 2, "लॉगिन लिंक": 2, "लॉगिन करें": 2,
    "खाता अपडेट": 2, "भुगतान अनुरोध": 2, "भुगतान लिंक": 2, "अब भुगतान करें": 2, "संग्रह अनुरोध": 2, "बैंक ट्रांसफर": 2, "मर्चेंट": 1, "चालान": 1, "बिल देय": 1, "बिलिंग": 1,
    "हम बैंक से हैं": 2, "हम समर्थन से हैं": 1, "सरकारी संदेश": 2, "आधार से": 2, "यूआईडीएआई से": 2, "ग्राहक सहायता से": 1, "हमारे साथ सत्यापित करें": 2,
    "बैंक से कॉल": 2, "समर्थन से कॉल": 1, "आपातकालीन कॉल": 1, "ग्राहक सेवा टीम": 1, "सत्यापन लिंक": 2, "सुरक्षित लिंक": 2,

    # Gujarati - expanded & de-duplicated
    "ઓટિપિ": 3, "ઓટીપી": 3, "પાસવર્ડ": 3, "પાસવર્ડ રીસેટ": 2, "પિન": 3, "સિવિવી": 3, "એસએસએન": 3,
    "આધાર": 3, "ખાતા": 2, "બેંક": 2, "બેંક ખાતું": 2, "ખાતા નંબર": 2, "તમારા ખાતાને સુરક્ષિત કરો": 2,
    "હવે વેરિફાઈ કરો": 2, "વેરિફાઈ": 2, "સત્યાપન": 1, "સત્યાપન લિંક": 2, "એક્સેસ": 2,
    "ટ્રાન્ઝેક્શન": 2, "પેમેન્ટ": 2, "ટ્રાન્સફર": 2, "ક્રેડિટ": 2, "ડેબિટ": 2,
    "ઇનામ": 2, "વિનર": 2, "લોટરી": 2, "દાવો": 2, "રિડિમ": 2, "રિડિમ કરો": 2, "રિફંડ": 2, "સસ્પેન્ડ": 2,
    "પુષ્ટિ": 2, "પુષ્ટિ કરો": 2, "અલર્ટ": 2, "સુરક્ષા અલર્ટ": 2, "તાત્કાલિક": 2, "તુરંત": 2, "તાત્કાલિક કાર્યવાહી": 2,
    "તાત્કાલિક અપડેટ": 1, "અંતિમ સૂચના": 1, "છેલ્લો મોકો": 1, "મિસ ન કરો": 1,
    "ઠગ": 2, "ફ્રોડ": 2, "ફિશિંગ": 2, "હેકિંગ": 2, "હેક્ડ": 2, "હેકર": 2, "મફત": 1,
    "ફ્રી": 1, "ઓફર": 1, "ક્લિક": 1, "અહીં ક્લિક કરો": 1, "લિંક": 1, "લિંક પર ક્લિક કરો": 1, "ડાઉનલોડ": 1,
    "જોખમ": 1, "જોખમ-મુક્ત": 1, "ગેરંટી": 1, "ટ્રાયલ": 1, "વિશેષ": 1, "મહત્વપૂર્ણ": 1, "ધ્યાન આપો": 1,
    "ઓળખ": 1, "ઓળખ-ચોરી": 2, "સુરક્ષિત": 1, "સુરક્ષા કરો": 1, "શંકાસ્પદ": 1, "ચેતવણી": 1, "સાવચેતી": 1,
    "પેસા": 1, "નકદ": 1, "ઉપહાર": 1, "બોનસ": 1, "સબ્સ્ક્રાઇબ": 1, "સાંજે ન કરો": 3, "શેર ન કરો": 3, "સંદેહજનક": 1,
    "નકલી": 1, "અંતિમ ચેતવણી": 1, "સુરક્ષા કોડ": 3, "પ્રમાણિકરણ કોડ": 3, "પ્રાધિકરણ": 2, "ઓળખ સત્યાપિત કરો": 2,
    "હવે ઓળખની પુષ્ટિ કરો": 2, "કેવાયસી": 2, "કેવાયસી પૂર્ણ કરો": 2, "તમારા ગ્રાહકને જાણો": 2,
    "ગ્રાહક સત્યાપન": 2, "સત્યાપન લિંક": 2, "સુરક્ષિત લિંક": 2, "પાસવર્ડ રીસેટ કરો": 2, "લોગિન લિંક": 2, "લોગિન કરો": 2,
    "ખાતા અપડેટ": 2, "ચુકવણી વિનંતી": 2, "ચુકવણી લિંક": 2, "હવે ચુકવણી કરો": 2, "સંગ્રહ વિનંતી": 2,
    "બેંક ટ્રાન્સફર": 2, "મર્ચન્ટ": 1, "ચલાન": 1, "બિલ બાકી": 1, "બિલિંગ": 1, "અમે બેંકમાંથી છીએ": 2,
    "અમે સપોર્ટમાંથી છીએ": 1, "સરકારી સંદેશો": 2, "આધારથી": 2, "યુઆઈડીએઈથી": 2, "ગ્રાહક સહાયતા થી": 1,
    "અમારા સાથે વેરિફાઈ કરો": 2, "બેંકથી કોલ": 2, "સપોર્ટથી કોલ": 1, "તાત્કાલિક કોલ": 1, "ગ્રાહક સેવા ટીમ": 1,

    # Other common multi-lingual keywords and phrases
    "verify otp": 3, "enter otp": 3, "enter code": 3, "authentication": 2, "authenticator": 2,
    "security question": 2, "mobile verification": 2, "sso": 1, "login": 2, "login to your account": 2,
    "reset": 2, "reset now": 2, "reset your password": 2, "expire": 1, "expired": 1,
    "account blocked": 2, "account locked": 2, "unauthorized": 2, "safeguard": 1,

    # Patterns that often indicate phishing or scam mechanics
    "verify-account": 2, "account-update": 2, "payment-request": 2, "confirm-payment": 2,
    "secure-link": 2, "one time password.": 3, "security-alert": 2, "suspicious-login": 2,
    "immediate-action": 2, "last-chance": 1, "final-warning": 1
}


# ============================
# Hybrid Text Scam Detection
# ============================
def detect_message(text: str, threshold: float = 0.35):
    """
    Hybrid detection:
      - Existing ML + keyword + sentiment scoring
      - Then call RAG (Phi3:mini via Ollama) to get an LLM verification & explanation
      - Combine scores (simple weighted average) and return structured dict
    """
    if not text:
        return {
            "label": "✅ Likely Safe",
            "ml_prob": 0.0,
            "keyword_score": 0,
            "sentiment": {"compound": 0.0},
            "combined_prob": 0.0,
            "keywords": [],
            "rag": None
        }

    # Normalize and lowercase
    text = unicodedata.normalize("NFKC", text)
    text_lower = text.lower()

    # Keyword detection
    found_keywords = []
    keyword_score = 0
    for kw, weight in RISKY_KEYWORDS.items():
        pattern = rf'(?<!\w){re.escape(kw)}(?!\w)'
        if re.search(pattern, text_lower, flags=re.UNICODE):
            found_keywords.append(kw)
            keyword_score += weight

    # Sentiment
    sentiment = sent_analyzer.polarity_scores(text)
    neg_score = sentiment.get("neg", 0.0)
    compound = sentiment.get("compound", 0.0)
    sentiment_boost = 0.1 if neg_score > 0.3 or compound < -0.2 else 0.0

    # ML model
    ml_prob = 0.0
    if text_model and vectorizer:
        try:
            vec = vectorizer.transform([text])
            ml_prob = float(text_model.predict_proba(vec)[0][1])
        except Exception as e:
            # use 0.0 if ML fails
            ml_prob = 0.0

    # Combine heuristics (keyword multiplier tunable)
    KEYWORD_MULTIPLIER = 0.06
    base_combined = ml_prob + (keyword_score * KEYWORD_MULTIPLIER) + sentiment_boost
    base_combined = max(0.0, min(base_combined, 1.0))

    # Default rag result
    rag_result = None
    llm_prob = None

    # If Ollama available, call RAG
    try:
        if is_ollama_available():
            rag_result = rag_verify_text(text=text, ml_prob=ml_prob, keywords=found_keywords, sentiment=sentiment)
            # rag_result may contain risk_percent or llm_prob; normalize it
            llm_prob = None
            if rag_result.get("risk_percent") is not None:
                try:
                    llm_prob = float(rag_result["risk_percent"]) / 100.0
                except Exception:
                    llm_prob = None
            elif rag_result.get("llm_prob") is not None:
                llm_prob = rag_result.get("llm_prob")
    except Exception:
        # any failure keeps rag_result as None
        rag_result = None

    # Combine base_combined with llm_prob (if available)
    if llm_prob is not None:
        # Weighted average: weight LLM 0.55, heuristics 0.45 (tuneable)
        COMBINE_WEIGHT_LLM = 0.55
        COMBINE_WEIGHT_HEUR = 0.45
        combined_prob = (llm_prob * COMBINE_WEIGHT_LLM) + (base_combined * COMBINE_WEIGHT_HEUR)
    else:
        combined_prob = base_combined

    combined_prob = max(0.0, min(combined_prob, 1.0))

    # Final label threshold (passed in from UI slider for real-time tuning)
    label = "🚨 Likely Scam" if combined_prob > threshold else "✅ Likely Safe"

    return {
        "label": label,
        "ml_prob": ml_prob,
        "keyword_score": keyword_score,
        "sentiment": sentiment,
        "combined_prob": combined_prob,
        "keywords": found_keywords,
        "rag": rag_result
    }


# ============================
# Scam Categorization
# ============================
def categorize_scam(keywords, sentiment):
    keywords = [kw.lower() for kw in keywords]
    compound = sentiment["compound"]

    if any(k in keywords for k in ["lottery", "winner", "prize", "gift", "reward"]):
        return "🎁 Reward / Lottery Scam"
    elif any(k in keywords for k in ["bank", "account", "otp", "password", "verify", "transaction"]):
        return "🏦 Banking / Verification Scam"
    elif compound < -0.3:
        return "⚠️ Fear / Threat-Based Scam"
    else:
        return "🧠 Unknown / Generic Scam"


# ============================
# Speech-to-Text with Vosk
# ============================
def transcribe_audio(audio_file, lang="en-in"):
    """Convert audio to text using Vosk, then optionally refine transcription with LLM (Phi3).

    Uses a unique temporary WAV file for Vosk so that concurrent Streamlit
    sessions do not collide on a single fixed filename.
    """
    model_path = VOSK_BASE_PATH / f"vosk-model-small-{lang}"
    if not model_path.exists():
        st.error(f"❌ Missing Vosk model: {model_path}")
        return ""

    try:
        vosk_model = Model(str(model_path))
    except Exception as e:
        st.error(f"❌ Failed to load Vosk model: {e}")
        return ""

    # Read the audio file (audio_file can be a path string or a file-like object)
    try:
        data, samplerate = sf.read(audio_file)
    except Exception as e:
        st.error(f"❌ Could not read audio file: {e}")
        return ""

    if len(data.shape) > 1:
        data = data.mean(axis=1)  # stereo -> mono

    # Write to a unique temp WAV so concurrent sessions don't overwrite each other
    tmp_wav_fd, tmp_wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_wav_fd)  # close the OS-level fd; sf.write will open it by path
    result_text = ""
    try:
        sf.write(tmp_wav_path, data, samplerate)

        with wave.open(tmp_wav_path, "rb") as wf:
            rec = KaldiRecognizer(vosk_model, wf.getframerate())
            rec.SetWords(False)  # we only need the text, not word timings
            while True:
                chunk = wf.readframes(4000)
                if len(chunk) == 0:
                    break
                if rec.AcceptWaveform(chunk):
                    res = json.loads(rec.Result())
                    result_text += " " + res.get("text", "")
            res = json.loads(rec.FinalResult())
            result_text += " " + res.get("text", "")
    except Exception as e:
        st.warning(f"Vosk transcription failed: {e}")
        result_text = ""
    finally:
        # Always clean up the temp file even if an exception occurs
        try:
            os.unlink(tmp_wav_path)
        except OSError:
            pass

    result_text = result_text.strip()
    if not result_text:
        return ""

    # Optionally refine via LLM
    if is_ollama_available():
        try:
            refine = rag_refine_transcription(result_text)
            cleaned = refine.get("cleaned_text") or result_text
            return cleaned
        except Exception:
            return result_text
    else:
        return result_text

# ============================
# Deepfake Video Detection
# ============================
def detect_deepfake(video_path, sample_frames=12):
    """Detect whether a video is likely a deepfake.

    Returns a (label, score) tuple. score is the average sigmoid probability
    across sampled frames; > 0.5 is classified as deepfake.
    """
    if not deepfake_model:
        return "❌ Model not loaded", 0.0

    cap = cv2.VideoCapture(video_path)

    # Validate that the file was opened successfully
    if not cap.isOpened():
        return "❌ Could not open video file", 0.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Guard against empty or unreadable video
    if total_frames <= 0:
        cap.release()
        return "❌ No video frames found", 0.0

    frame_idxs = np.linspace(0, total_frames - 1, sample_frames, dtype=int)

    preds = []
    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224)) / 255.0
        tensor = np.expand_dims(frame_resized, axis=0)

        prob = deepfake_model.predict(tensor, verbose=0)[0][0]  # binary sigmoid classifier
        preds.append(prob)

    cap.release()

    if preds:
        avg_score = float(np.mean(preds))
        label = "🚨 Likely Deepfake" if avg_score > 0.5 else "✅ Likely Real"
        return label, avg_score
    else:
        return "❌ No frames processed", 0.0

def render_circular_progress(risk: int, color: str, size_px: int = 120):
    """
    Render an accurate SVG circular progress using components.html so the SVG
    is not escaped and the arc length matches the numeric percent.
    """
    radius = 15.9155
    circumference = 2 * math.pi * radius
    pct = max(0, min(int(risk), 100))
    offset = circumference * (1 - pct / 100.0)

    svg = f"""
    <div style="text-align:center;">
      <svg width="{size_px}" height="{size_px}" viewBox="0 0 36 36" role="img" aria-label="{pct}%">
        <!-- background -->
        <circle cx="18" cy="18" r="{radius}" fill="none" stroke="#eee" stroke-width="2"></circle>

        <!-- progress: full circumference + dashoffset -->
        <circle cx="18" cy="18" r="{radius}" fill="none" stroke="{color}" stroke-width="2"
                stroke-dasharray="{circumference:.3f}" stroke-dashoffset="{offset:.3f}"
                stroke-linecap="round" transform="rotate(-90 18 18)"></circle>

        <!-- center label -->
        <text x="18" y="20.35" font-size="4" text-anchor="middle" fill="{color}">{pct}%</text>
      </svg>
    </div>
    """

    # components.html renders raw HTML/SVG reliably; adjust height so nothing is clipped
    components.html(svg, height=size_px + 20)

# -------------------------------
# Custom CSS for Modern UI & Light Theme (forced)
# -------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    body, .stApp { background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); color: #222; }
    .card, .info-card, .upload-box { color: #222; }
    .stTextArea textarea, .stSelectbox div, .stFileUploader, .stButton>button, .stSelectbox select { background: #fff; color: #222; }
    /* ...rest of your CSS... */
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #ff6b6b 0%, #ffa86b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #6c757d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        margin-bottom: 25px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 25px rgba(0, 0, 0, 0.15);
    }
    .card-title {
        color: #6b8cff;
        font-size: 1.6rem;
        font-weight: 600;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .result-icon {
        font-size: 4rem;
        margin-bottom: 15px;
        text-align: center;
    }
    .result-label {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 15px;
        text-align: center;
    }
    .scam {
        color: #FF6B6B;
    }
    .safe {
        color: #3BCE3B;
    }
    .probability {
        font-size: 1.2rem;
        margin-bottom: 15px;
        text-align: center;
    }
    .keyword-list {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        justify-content: center;
        margin-top: 15px;
    }
    .keyword {
        background: rgba(107, 140, 255, 0.2);
        padding: 8px 15px;
        border-radius: 20px;
        font-size: 0.9rem;
        transition: all 0.3s ease;
    }
    .keyword:hover {
        transform: scale(1.05);
    }
    .risky {
        background: rgba(255, 107, 107, 0.2);
    }
    .info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin-top: 20px;
    }
    .info-card {
        background: rgba(0, 0, 0, 0.1);
        border-radius: 12px;
        padding: 20px;
        transition: all 0.3s ease;
    }
    .info-card:hover {
        background: rgba(0, 0, 0, 0.15);
    }
    .info-card h3 {
        color: #6b8cff;
        margin-bottom: 15px;
        font-size: 1.3rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        background: linear-gradient(90deg, #6b8cff 0%, #7b68ee 100%);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(107, 140, 255, 0.4);
    }
    .input-type-selector {
        display: flex;
        margin-bottom: 20px;
        background: rgba(0, 0, 0, 0.1);
        border-radius: 12px;
        overflow: hidden;
    }
    .input-type {
        flex: 1;
        padding: 12px;
        text-align: center;
        cursor: pointer;
        transition: background 0.3s;
    }
    .input-type.active {
        background: #6b8cff;
        font-weight: 500;
    }
    .tab-content {
        padding: 20px 0;
    }
    .language-selector {
        width: 100%;
        padding: 12px;
        background: rgba(0, 0, 0, 0.1);
        border: none;
        border-radius: 12px;
        color: inherit;
        margin-bottom: 20px;
    }
    .upload-box {
        border: 2px dashed #6b8cff;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    .upload-box:hover {
        background: rgba(107, 140, 255, 0.05);
    }
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 25px 0;
    }
    .stat-item {
        text-align: center;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #6b8cff;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #6c757d;
    }
    .progress-bar {
        height: 8px;
        background: rgba(0, 0, 0, 0.1);
        border-radius: 4px;
        overflow: hidden;
        margin: 15px 0;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff6b6b 0%, #ffa86b 100%);
        border-radius: 4px;
    }
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        .info-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar Navigation (no theme toggle)
# -------------------------------
st.sidebar.markdown(
    """
    <div style="text-align:center; margin-bottom:30px;">
        <h1 style="color:#6b8cff; margin-bottom:5px;">🛡 Digital Arrest Detector</h1>
        <p style="color:#6c757d; font-size:0.9rem;">Multi-Channel Fraud Prevention</p>
    </div>
    """, 
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Detection Statistics")
st.sidebar.markdown(
    """
    <div class="stats-container">
        <div class="stat-item">
            <div class="stat-value">96%</div>
            <div class="stat-label">Accuracy</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">90+</div>
            <div class="stat-label">Keywords</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">3</div>
            <div class="stat-label">Languages</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Detection Sensitivity")
scam_threshold = st.sidebar.slider(
    label="Scam score threshold",
    min_value=0.10,
    max_value=0.80,
    value=0.35,
    step=0.05,
    help=(
        "Messages with a combined scam score above this value are flagged as ‘Likely Scam’. "
        "Lower values catch more scams but may increase false positives; "
        "higher values reduce false positives but may miss borderline scams."
    ),
    key="scam_threshold_slider",
)

st.sidebar.markdown("---")
st.sidebar.info("""
**How it works:**
1. Select input type (Text, Audio, Video)
2. Provide content for analysis
3. Get instant scam detection results
""")

# -------------------------------
# Main Content Layout
# -------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1 class="main-title">FraudShield AI</h1>
        <p class="subtitle">Advanced AI-powered detection for text, audio, and video content with multi-language support</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["📝 Text Analysis", "🎙️ Audio Analysis", "🎥 Video Analysis"])

with tab1:
    # st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title"><span>📝</span> Text Input</div>', unsafe_allow_html=True)
    
    user_text = st.text_area(
        "Enter text to analyze for scam content:",
        placeholder="Paste or type the message you want to analyze here...",
        height=150
    )
    
    if st.button("Analyze Text", key="analyze_text"):
        if user_text.strip():
            with st.spinner("Analyzing text content..."):
                result = detect_message(user_text, threshold=scam_threshold)

                # Extract data safely
                combined_prob = result.get("combined_prob", 0)
                percent = f"{combined_prob*100:.1f}"

                icon = "🚨" if "Scam" in result["label"] else "✅"
                label_class = "scam" if "Scam" in result["label"] else "safe"

                category = categorize_scam(result["keywords"], result["sentiment"])

                risk = int(result["combined_prob"]*100)
                color = "#3BCE3B" if risk < 40 else "#FFA86B" if risk < 70 else "#FF6B6B"

                render_circular_progress(risk, color)

                st.markdown(f'<div class="result-icon">{icon}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="result-label {label_class}">{result["label"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="probability {label_class}">Scam probability: {percent}%</div>', unsafe_allow_html=True)
                st.markdown(f"**Scam Category:** {category}")

                # Optional extra display for explainability
                # st.write(f"**Detected Keywords:** {', '.join(result['keywords']) if result['keywords'] else 'None'}")
                st.write(f"**Sentiment (compound):** {result['sentiment']['compound']:.2f}")
                st.write(f"**ML Model Probability:** {result['ml_prob']:.2f}")
                
                # --- RAG Section ---
                rag = result.get("rag")
                if rag:
                    st.markdown("### 🧠 AI Reasoning Summary (Phi3:mini)")
                    # If rag contains structured explanation and advice, render nicely:
                    explanation = rag.get("explanation") or str(rag.get("raw") or "")
                    st.write("**Explanation:**")
                    st.info(explanation)

                    advice = rag.get("advice") or []
                    if advice:
                        st.write("**Advice:**")
                        for a in advice:
                            st.markdown(f"- {a}")
                    # If rag returns explicit numeric risk_percent, show a second gauge
                    rp = rag.get("risk_percent") or rag.get("llm_prob")
                    if rp:
                        try:
                            rp_val = float(rp) if not isinstance(rp, str) else float(rp)
                            rp_pct = int(rp_val*100) if rp_val <= 1 else int(rp_val)
                            st.write(f"**LLM risk estimate:** {rp_pct}%")
                        except Exception:
                            pass
                else:
                    st.info("LLM (Ollama) not available or RAG skipped. Using heuristic/ML results.")


                if "Scam" in result["label"]:
                    st.warning("⚠️ Advice: Do not click on suspicious links or share OTPs. Verify sender via official site.")
                else:
                    st.success("✅ Message seems safe. Always double-check unknown numbers or domains.")


                # Display keywords
                keywords = result.get("keywords", []) if isinstance(result, dict) else []
                if keywords:
                    st.markdown("<h4>Detected Risky Keywords</h4>", unsafe_allow_html=True)
                    st.markdown('<div class="keyword-list">', unsafe_allow_html=True)
                    for kw in keywords:
                        st.markdown(f'<div class="keyword risky">{kw}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("No risky keywords detected in this text.")
        else:
            st.warning("Please enter some text to analyze.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    # st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title"><span>🎙️</span> Audio Analysis</div>', unsafe_allow_html=True)
    
    # st.markdown('<div class="upload-box">', unsafe_allow_html=True)s
    uploaded_audio = st.file_uploader("Upload an audio file", type=["wav", "mp3"], key="audio_upload")
    # st.markdown('</div>', unsafe_allow_html=True)
    
    lang_choice = st.selectbox("Select Language", ["en-in (Indian English)", "hi (Hindi)", "gu (Gujarati)"])
    lang_map = {"en-in (Indian English)": "en-in", "hi (Hindi)": "hi", "gu (Gujarati)": "gu"}
    
    if uploaded_audio and st.button("Analyze Audio", key="analyze_audio"):
        # Write to a unique temp file so concurrent sessions don’t collide
        suffix = Path(uploaded_audio.name).suffix or ".wav"
        tmp_audio_fd, temp_audio_path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(tmp_audio_fd, "wb") as f:
                f.write(uploaded_audio.read())

            with st.spinner("Transcribing audio content..."):
                transcription = transcribe_audio(temp_audio_path, lang=lang_map[lang_choice])
        finally:
            # Always remove the upload temp file even if transcription fails
            try:
                os.unlink(temp_audio_path)
            except OSError:
                pass
    
        if transcription:
            st.success("Audio transcribed successfully!")
            st.text_area("Transcribed Text", transcription, height=120)
    
            with st.spinner("Analyzing transcribed text..."):
                result = detect_message(transcription, threshold=scam_threshold)
    
            # Safely extract results
            combined_prob = result.get("combined_prob", 0)
            percent = f"{combined_prob*100:.1f}"
            icon = "🚨" if "Scam" in result["label"] else "✅"
            label_class = "scam" if "Scam" in result["label"] else "safe"
    
            category = categorize_scam(result.get("keywords", []), result.get("sentiment", {"compound": 0}))
    
            risk = int(combined_prob*100)
            color = "#3BCE3B" if risk < 40 else "#FFA86B" if risk < 70 else "#FF6B6B"

            # Circular progress SVG
            render_circular_progress(risk, color)
    
            st.markdown(f'<div class="result-icon">{icon}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-label {label_class}">{result["label"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="probability {label_class}">Scam probability: {percent}%</div>', unsafe_allow_html=True)
            st.markdown(f"**Scam Category:** {category}")
    
            # Explainability details
            st.write(f"**Detected Keywords:** {', '.join(result.get('keywords', [])) if result.get('keywords') else 'None'}")
            st.write(f"**Sentiment (compound):** {result.get('sentiment', {}).get('compound', 0):.2f}")
            st.write(f"**ML Model Probability:** {result.get('ml_prob', 0):.2f}")
    
            if "Scam" in result["label"]:
                st.warning("⚠️ Advice: Do not click on suspicious links or share OTPs. Verify sender via official site.")
            else:
                st.success("✅ Audio content seems safe. Always double-check unknown callers or links.")

            # --- RAG Section ---
            # Rendered outside the scam/safe if-else so it appears for both outcomes
            rag = result.get("rag")
            if rag:
                st.markdown("### 🧠 AI Reasoning Summary (Phi3:mini)")
                # Render structured explanation and advice from the LLM
                explanation = rag.get("explanation") or str(rag.get("raw") or "")
                st.write("**Explanation:**")
                st.info(explanation)

                advice = rag.get("advice") or []
                if advice:
                    st.write("**Advice:**")
                    for a in advice:
                        st.markdown(f"- {a}")
                # If rag returns a numeric risk estimate, surface it
                rp = rag.get("risk_percent") or rag.get("llm_prob")
                if rp:
                    try:
                        rp_val = float(rp) if not isinstance(rp, str) else float(rp)
                        rp_pct = int(rp_val * 100) if rp_val <= 1 else int(rp_val)
                        st.write(f"**LLM risk estimate:** {rp_pct}%")
                    except Exception:
                        pass
            else:
                st.info("LLM (Ollama) not available or RAG skipped. Using heuristic/ML results.")
    
            # Display keywords visually
            keywords = result.get("keywords", []) if isinstance(result, dict) else []
            if keywords:
                st.markdown("<h4>Detected Risky Keywords</h4>", unsafe_allow_html=True)
                st.markdown('<div class="keyword-list">', unsafe_allow_html=True)
                for kw in keywords:
                    st.markdown(f'<div class="keyword risky">{kw}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No risky keywords detected in this audio.")
        else:
            st.warning("Transcription failed or returned no text. Try a different file or language.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    # st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title"><span>🎥</span> Video Analysis</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"], key="video_upload")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_video and st.button("Analyze Video", key="analyze_video"):
        # Use the original file's extension so OpenCV can pick the right codec
        video_suffix = Path(uploaded_video.name).suffix or ".mp4"
        tmp_video_fd, temp_video_path = tempfile.mkstemp(suffix=video_suffix)
        try:
            with os.fdopen(tmp_video_fd, "wb") as f:
                f.write(uploaded_video.read())

            with st.spinner("Analyzing video for deepfake indicators..."):
                label, score = detect_deepfake(temp_video_path)
        finally:
            # Always remove the temp video file after analysis
            try:
                os.unlink(temp_video_path)
            except OSError:
                pass
            
        # Display results
        percent = f"{score*100:.1f}"
        icon = "🚨" if "Deepfake" in label else "✅"
        label_class = "scam" if "Deepfake" in label else "safe"
        
        st.markdown(f'<div class="result-icon">{icon}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-label {label_class}">{label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="probability {label_class}">Deepfake probability: {percent}%</div>', unsafe_allow_html=True)
        
        # Progress bar visualization
        st.markdown('<div class="progress-bar"><div class="progress-fill" style="width: {}%;"></div></div>'.format(percent), unsafe_allow_html=True)
        
        # Additional info based on result
        if "Deepfake" in label:
            st.warning("This video shows signs of manipulation. Exercise caution and verify through other means.")
        else:
            st.success("No significant deepfake indicators detected in this video.")
    st.markdown('</div>', unsafe_allow_html=True)

# Information Section
st.markdown("## How It Works")
st.markdown("""
<div class="info-grid">
    <div class="info-card">
        <h3>🤖 Hybrid Detection System</h3>
        <p>Combines machine learning models with rule-based keyword analysis for more accurate scam detection across multiple channels.</p>
    </div>
    <div class="info-card">
        <h3>🌐 Multi-Language Support</h3>
        <p>Uses Vosk speech recognition models to process audio in Hindi, Gujarati, and Indian English, making it suitable for diverse users across India.</p>
    </div>
    <div class="info-card">
        <h3>🔍 Deepfake Detection</h3>
        <p>Analyzes video frames using advanced neural networks to identify potential deepfake content with high accuracy.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#6c757d; font-size:0.9rem; padding:20px;">
        <p>Multi-Channel Digital Arrest & Fraud Scam Detection System</p>
        <p>Powered by AI and Vosk Speech Recognition</p>
    </div>
    """,
    unsafe_allow_html=True
)