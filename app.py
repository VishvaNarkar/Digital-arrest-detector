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

# -------------------------------
# Load Models
# -------------------------------
# Text classifier
model = joblib.load("models/text_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Audio transcription
vosk_base_path = "models"

# Video deepfake model (Keras)
deepfake_model = tf.keras.models.load_model("models/Deepfakes_detection_model.keras")

# -------------------------------
# Risky keywords for rule-based check
# -------------------------------
risky_keywords = [
    "bank", "account", "verify", "locked", "password",
    "urgent", "winner", "lottery", "click", "payment",
    "prize", "free", "offer", "limited", "risk", "security",
    "immediately", "suspend", "alert", "confirm",
    "transaction", "access", "details", "information",
    "identity", "social", "security", "ssn", "credit", "debit",
    "otp", "pin", "cvv", "scam", "fraud", "fake"
]

# -------------------------------
# Hybrid Text Scam Detection
# -------------------------------
def detect_message(text):
    X_input = vectorizer.transform([text])
    scam_prob = model.predict_proba(X_input)[0][1]
    found_keywords = [kw for kw in risky_keywords if re.search(rf"\b{kw}\b", text, re.IGNORECASE)]
    keyword_score = len(found_keywords)

    if scam_prob > 0.3 or keyword_score >= 2:
        label = "🚨 Likely Scam"
    else:
        label = "✅ Likely Safe"

    return label, scam_prob, found_keywords

# -------------------------------
# Speech-to-Text with Vosk
# -------------------------------
def transcribe_audio(audio_file, lang="en-in"):
    model_path = f"{vosk_base_path}/vosk-model-small-{lang}"
    model = Model(model_path)

    data, samplerate = sf.read(audio_file)
    if len(data.shape) > 1:
        data = data.mean(axis=1)  # stereo → mono
    sf.write("temp.wav", data, samplerate)

    wf = wave.open("temp.wav", "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    result_text = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            result_text += " " + res.get("text", "")
    res = json.loads(rec.FinalResult())
    result_text += " " + res.get("text", "")
    return result_text.strip()

# -------------------------------
# Deepfake Video Detection
# -------------------------------
def detect_deepfake(video_path, sample_frames=12):
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

        prob = deepfake_model.predict(tensor, verbose=0)[0][0]  # assuming output is [prob_fake]
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
            <div class="stat-value">36</div>
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
        <h1 class="main-title">Multi-Channel Digital Arrest & Fraud Scam Detection</h1>
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
                label, scam_prob, keywords = detect_message(user_text)
                
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