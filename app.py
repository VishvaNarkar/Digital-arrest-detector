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
from torchvision import transforms

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
    "identity", "social", "security", "ssn", "credit", "debit"
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
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Digital Arrest Detector", layout="wide")
st.title("🛡 Digital Arrest Detector")
st.write("Unified detection system for **Text, Audio, and Video deepfakes**")

tab1, tab2, tab3 = st.tabs(["📝 Text Analysis", "🎙 Audio Call Analysis", "🎥 Video Deepfake Analysis"])

# --- Text Tab
with tab1:
    st.header("📝 Text Analysis")
    user_input = st.text_area("Paste text (SMS/Email/Chat) here:")
    if st.button("Analyze Text"):
        if user_input.strip():
            label, scam_prob, keywords = detect_message(user_input)
            st.success("✅ Analysis Complete")
            st.subheader("🔍 Result")
            st.write(f"**Classification:** {label}")
            st.write(f"**ML Scam Probability:** {scam_prob:.2f}")
            st.write(f"**Risky Keywords:** {', '.join(keywords) if keywords else 'None'}")

# --- Audio Tab
with tab2:
    st.header("🎙 Audio Call Analysis")
    uploaded_audio = st.file_uploader("Upload call recording (wav/mp3)", type=["wav", "mp3"])
    lang_choice = st.selectbox("Select Language Model", ["en-in", "gu", "hi"])

    if uploaded_audio and st.button("Analyze Audio"):
        with st.spinner("Transcribing audio..."):
            transcription = transcribe_audio(uploaded_audio, lang=lang_choice)
        st.subheader("🗣 Transcribed Text")
        st.write(transcription if transcription else "❌ No speech detected.")

        if transcription:
            label, scam_prob, keywords = detect_message(transcription)
            st.subheader("🔍 Scam Analysis")
            st.write(f"**Classification:** {label}")
            st.write(f"**ML Scam Probability:** {scam_prob:.2f}")
            st.write(f"**Risky Keywords:** {', '.join(keywords) if keywords else 'None'}")

        # Placeholder for audio deepfake detection
        st.subheader("🤖 Deepfake Audio Detection")
        st.info("⚠️ Audio deepfake detection model integration pending.")

# --- Video Tab
with tab3:
    st.header("🎥 Video Deepfake Detection")
    uploaded_video = st.file_uploader("Upload video (mp4/avi/mov)", type=["mp4", "avi", "mov"])
    if uploaded_video and st.button("Analyze Video"):
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())

        with st.spinner("Analyzing video frames..."):
            label, score = detect_deepfake("temp_video.mp4")

        st.subheader("🔍 Video Deepfake Analysis")
        st.write(f"**Classification:** {label}")
        st.write(f"**Deepfake Probability:** {score:.2f}")
