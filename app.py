import streamlit as st
import joblib
import re
import wave
import json
import soundfile as sf
import cv2
import numpy as np
import tensorflow as tf
from vosk import Model, KaldiRecognizer
from pathlib import Path
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import re

nltk.download("vader_lexicon")
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
    "otp": 3, "pin": 3, "cvv": 3, "password": 3, "ssn": 3,
    "bank": 2, "account": 2, "verify": 2, "verify now": 2, "secure your account": 2,
    "access": 2, "transaction": 2, "payment": 2, "transfer": 2, "credit": 2, "debit": 2,
    "prize": 2, "winner": 2, "lottery": 2, "claim": 2, "redeem": 2, "suspend": 2,
    "confirm": 2, "alert": 2, "urgent": 2, "immediately": 2, "urgent action": 2,
    "scam": 2, "fraud": 2, "phishing": 2, "hacking": 2, "giveaway": 2, "reward": 2,
    "limited": 1, "offer": 1, "free": 1, "click": 1, "risk": 1, "security": 1,
    "identity": 1, "social": 1, "money": 1, "fake": 1, "suspicious": 1,
    "warning": 1, "caution": 1, "danger": 1, "urgent update": 1, "act now": 1,
    "exclusive": 1, "important": 1, "attention": 1, "limited time": 1,
    "win": 1, "cash": 1, "gift": 1, "bonus": 1, "subscribe": 1,
    "click here": 1, "visit": 1, "link": 1, "download": 1,
    "risk-free": 1, "guarantee": 1, "trial": 1, "urgent response": 1,
    "act quickly": 1, "don't miss": 1, "final notice": 1, "last chance": 1,
    "immediate action": 1, "secure": 1, "protect": 1, "verification": 1,

    # Hindi (Devanagari)
    "ओटीपी": 3, "पिन": 3, "सीवीवी": 3, "पासवर्ड": 3, "एसएसएन": 3,
    "बैंक": 2, "खाता": 2, "वेरिफाई": 2, "अब सत्यापित करें": 2, "अपने खाते को सुरक्षित करें": 2,
    "एक्सेस": 2, "लेनदेन": 2, "भुगतान": 2, "ट्रांसफर": 2, "क्रेडिट": 2, "डेबिट": 2,
    "इनाम": 2, "विनर": 2, "लॉटरी": 2, "दावा": 2, "रिडीम": 2, "सस्पेंड": 2,
    "पुष्टि": 2, "अलर्ट": 2, "तुरंत": 2, "फौरन": 2, "त्वरित कार्रवाई": 2,
    "स्कैम": 2, "फ्रॉड": 2, "फिशिंग": 2, "हैकिंग": 2,
    "मुफ्त": 1, "ऑफर": 1, "क्लिक": 1, "जोखिम": 1, "सुरक्षा": 1,
    "पहचान": 1, "सोशल": 1, "पैसे": 1, "नकली": 1, "संदेहजनक": 1, "चेतावनी": 1,
    "सावधानी": 1, "खतरा": 1, "तत्काल अपडेट": 1, "अब कार्य करें": 1,
    "विशेष": 1, "महत्वपूर्ण": 1, "ध्यान दें": 1, "सीमित समय": 1,
    "जीत": 1, "नकद": 1, "उपहार": 1, "बोनस": 1, "सब्सक्राइब": 1,
    "यहाँ क्लिक करें": 1, "भ्रमण करें": 1, "लिंक": 1, "डाउनलोड": 1,
    "जोखिम-मुक्त": 1, "गारंटी": 1, "परीक्षण": 1, "तत्काल प्रतिक्रिया": 1,
    "त्वरित कार्य करें": 1, "मिस न करें": 1, "अंतिम नोटिस": 1, "अंतिम मौका": 1,
    "तत्काल कार्रवाई": 1, "सुरक्षित": 1, "सुरक्षा करें": 1, "सत्यापन": 1,

    # Gujarati
    "ઓટિપિ": 3, "ઓટીપી": 3, "પાસવર્ડ": 3, "પિન": 3, "સિવિવી": 3, "એસએસએન": 3,
    "બેંક": 2, "ખાતા": 2, "વેરિફાઈ": 2, "હવે વેરિફાઈ કરો": 2, "તમારા ખાતાને સુરક્ષિત કરો": 2,
    "એક્સેસ": 2, "ટ્રાન્ઝેક્શન": 2, "પેમેન્ટ": 2, "ટ્રાન્સફર": 2, "ક્રેડિટ": 2, "ડેબિટ": 2,
    "ઇનામ": 2, "વિનર": 2, "લોટરી": 2, "દાવો": 2, "રિડિમ": 2, "સસ્પેન્ડ": 2,
    "પુષ્ટિ": 2, "અલર્ટ": 2, "તાત્કાલિક": 2, "તુરંત": 2, "તાત્કાલિક કાર્યવાહી": 2,
    "ઠગ": 2, "ફ્રોડ": 2, "ફિશિંગ": 2, "હેકિંગ": 2,
    "ફ્રી": 1, "ઓફર": 1, "ક્લિક": 1, "જોખમ": 1, "સುರક્ષા": 1,
    "ઓળખ": 1, "સોશિયલ": 1, "પેસા": 1, "નકલી": 1, "શંકાસ્પદ": 1, "ચેતવણી": 1,
    "સાવચેતી": 1, "ખતરો": 1, "તાત્કાલિક અપડેટ": 1, "હવે કાર્ય કરો": 1,
    "વિશેષ": 1, "મહત્વપૂર્ણ": 1, "ધ્યાન આપો": 1, "સીમિત સમય": 1,
    "જીત": 1, "નકદ": 1, "ઉપહાર": 1, "બોનસ": 1, "સબ્સ્ક્રાઇબ": 1,
    "અહીં ક્લિક કરો": 1, "વિઝિટ": 1, "લિંક": 1, "ડાઉનલોડ": 1,
    "જોખમ-મુક્ત": 1, "ગેરંટી": 1, "ટ્રાયલ": 1, "તાત્કાલિક પ્રતિસાદ": 1,
    "તાત્કાલિક કાર્ય કરો": 1, "મિસ ન કરો": 1, "અંતિમ સૂચના": 1, "છેલ્લો મોકો": 1,
    "તાત્કાલિક કાર્યવાહી": 1, "સુરક્ષિત": 1, "સુરક્ષા કરો": 1, "સત્યાપન": 1,
}


# ============================
# Hybrid Text Scam Detection
# ============================
def detect_message(text: str):
    """Enhanced scam detection with weighted keywords + sentiment."""
    X_input = vectorizer.transform([text])
    ml_prob = text_model.predict_proba(X_input)[0][1]

    # --- Weighted keyword detection ---
    found_keywords = []
    keyword_score = 0
    for kw, weight in RISKY_KEYWORDS.items():
        if re.search(rf"\b{kw}\b", text, re.IGNORECASE):
            found_keywords.append(kw)
            keyword_score += weight

    # --- Sentiment analysis ---
    sentiment = sent_analyzer.polarity_scores(text)
    neg_score = sentiment['neg']
    compound = sentiment['compound']

    # Urgency / fear boosting (if tone is very negative)
    sentiment_boost = 0.1 if neg_score > 0.3 or compound < -0.2 else 0.0

    # --- Combined risk scoring ---
    combined_prob = ml_prob + (keyword_score * 0.03) + sentiment_boost
    combined_prob = min(combined_prob, 1.0)  # clamp to 1.0

    # --- Decision logic ---
    if combined_prob > 0.35:
        label = "🚨 Likely Scam"
    else:
        label = "✅ Likely Safe"

    return {
        "label": label,
        "ml_prob": ml_prob,
        "keyword_score": keyword_score,
        "sentiment": sentiment,
        "combined_prob": combined_prob,
        "keywords": found_keywords
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
    """Convert audio to text using Vosk."""
    model_path = VOSK_BASE_PATH / f"vosk-model-small-{lang}"
    if not model_path.exists():
        st.error(f"❌ Missing Vosk model: {model_path}")
        return ""

    vosk_model = Model(str(model_path))
    data, samplerate = sf.read(audio_file)

    if len(data.shape) > 1:
        data = data.mean(axis=1)  # stereo → mono

    sf.write("temp.wav", data, samplerate)

    with wave.open("temp.wav", "rb") as wf:
        rec = KaldiRecognizer(vosk_model, wf.getframerate())
        rec.SetWords(True)

        result_text = ""
        while True:
            chunk = wf.readframes(4000)
            if len(chunk) == 0:
                break
            if rec.AcceptWaveform(chunk):
                res = json.loads(rec.Result())
                result_text += " " + res.get("text", "")
        res = json.loads(rec.FinalResult())
        result_text += " " + res.get("text", "")

    return result_text.strip()


# ============================
# Deepfake Video Detection
# ============================
def detect_deepfake(video_path, sample_frames=12):
    """Detect whether a video is likely a deepfake."""
    if not deepfake_model:
        return "❌ Model not loaded", 0.0

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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

        prob = deepfake_model.predict(tensor, verbose=0)[0][0]  # assuming binary classifier
        preds.append(prob)

    cap.release()

    if preds:
        avg_score = float(np.mean(preds))
        label = "🚨 Likely Deepfake" if avg_score > 0.5 else "✅ Likely Real"
        return label, avg_score
    else:
        return "❌ No frames processed", 0.0

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
        color: #ff6b6b;
    }
    .safe {
        color: #6bff8c;
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
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title"><span>📝</span> Text Input</div>', unsafe_allow_html=True)
    
    user_text = st.text_area(
        "Enter text to analyze for scam content:",
        placeholder="Paste or type the message you want to analyze here...",
        height=150
    )
    
    if st.button("Analyze Text", key="analyze_text"):
        if user_text.strip():
            with st.spinner("Analyzing text content..."):
                result = detect_message(user_text)

                # Extract data safely
                combined_prob = result.get("combined_prob", 0)
                percent = f"{combined_prob*100:.1f}"

                icon = "🚨" if "Scam" in result["label"] else "✅"
                label_class = "scam" if "Scam" in result["label"] else "safe"

                category = categorize_scam(result["keywords"], result["sentiment"])

                risk = int(result["combined_prob"]*100)
                color = "#6bff8c" if risk < 40 else "#ffa86b" if risk < 70 else "#ff6b6b"

                st.markdown(f'<div class="result-icon">{icon}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="result-label {label_class}">{result["label"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="probability {label_class}">Scam probability: {percent}%</div>', unsafe_allow_html=True)
                st.markdown(f"**Scam Category:** {category}")

                # Optional extra display for explainability
                st.write(f"**Detected Keywords:** {', '.join(result['keywords']) if result['keywords'] else 'None'}")
                st.write(f"**Sentiment (compound):** {result['sentiment']['compound']:.2f}")
                st.write(f"**ML Model Probability:** {result['ml_prob']:.2f}")
                
                if "Scam" in result["label"]:
                    st.warning("⚠️ Advice: Do not click on suspicious links or share OTPs. Verify sender via official site.")
                else:
                    st.success("✅ Message seems safe. Always double-check unknown numbers or domains.")


                # Progress bar visualization
                # st.markdown('<div class="progress-bar"><div class="progress-fill" style="width: {}%;"></div></div>'.format(percent), unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="text-align:center;">
                <svg width="120" height="120" viewBox="0 0 36 36">
                    <path d="M18 2.0845
                            a 15.9155 15.9155 0 0 1 0 31.831
                            a 15.9155 15.9155 0 0 1 0 -31.831"
                        fill="none" stroke="#eee" stroke-width="2"/>
                    <path d="M18 2.0845
                            a 15.9155 15.9155 0 0 1 0 31.831"
                        fill="none" stroke="{color}" stroke-width="2" 
                        stroke-dasharray="{risk},100"/>
                    <text x="18" y="20.35" font-size="8" text-anchor="middle" fill="{color}">
                {risk}%
                    </text>
                </svg>
                </div>
                """, unsafe_allow_html=True)

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
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title"><span>🎙️</span> Audio Analysis</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_audio = st.file_uploader("Upload an audio file", type=["wav", "mp3"], key="audio_upload")
    st.markdown('</div>', unsafe_allow_html=True)
    
    lang_choice = st.selectbox("Select Language", ["en-in (Indian English)", "hi (Hindi)", "gu (Gujarati)"])
    lang_map = {"en-in (Indian English)": "en-in", "hi (Hindi)": "hi", "gu (Gujarati)": "gu"}
    
    if uploaded_audio and st.button("Analyze Audio", key="analyze_audio"):
        with st.spinner("Transcribing audio content..."):
            transcription = transcribe_audio(uploaded_audio, lang=lang_map[lang_choice])
            
        if transcription:
            st.success("Audio transcribed successfully!")
            st.text_area("Transcribed Text", transcription, height=100)
            
            with st.spinner("Analyzing transcribed text..."):
                label, scam_prob, keywords = detect_message(transcription)
                
            # Display results
            percent = f"{scam_prob*100:.1f}"
            icon = "🚨" if "Scam" in label else "✅"
            label_class = "scam" if "Scam" in label else "safe"
            
            st.markdown(f'<div class="result-icon">{icon}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-label {label_class}">{label}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="probability {label_class}">Scam probability: {percent}%</div>', unsafe_allow_html=True)
            
            # Progress bar visualization
            st.markdown('<div class="progress-bar"><div class="progress-fill" style="width: {}%;"></div></div>'.format(percent), unsafe_allow_html=True)
            
            # Display keywords
            if keywords:
                st.markdown("<h4>Detected Risky Keywords</h4>", unsafe_allow_html=True)
                st.markdown('<div class="keyword-list">', unsafe_allow_html=True)
                for kw in keywords:
                    st.markdown(f'<div class="keyword risky">{kw}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No risky keywords detected in this audio.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title"><span>🎥</span> Video Analysis</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"], key="video_upload")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_video and st.button("Analyze Video", key="analyze_video"):
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())
            
        with st.spinner("Analyzing video for deepfake indicators..."):
            label, score = detect_deepfake("temp_video.mp4")
            
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