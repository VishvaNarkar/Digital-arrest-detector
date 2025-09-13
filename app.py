import streamlit as st
import joblib
import re
import wave
import json
import soundfile as sf
from vosk import Model, KaldiRecognizer

# --- Load model + vectorizer ---
model = joblib.load("models/text_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Risky keywords
risky_keywords = ["bank", "account", "verify", "locked", "password",
                  "urgent", "winner", "lottery", "click", "payment",
                  "prize", "free", "offer", "limited", "risk", "security",
                  "immediately", "suspend", "alert", "confirm",
                  "transaction", "access", "details", "information",
                  "identity", "social", "security", "ssn", "credit", "debit"]

# --- Hybrid Detection Function ---
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

# --- Speech-to-Text with Vosk ---
def transcribe_audio(audio_file, lang="en-in"):
    model_path = f"models/vosk-model-small-{lang}"
    model = Model(model_path)

    # Convert to wav if not already
    data, samplerate = sf.read(audio_file)
    if len(data.shape) > 1:  # stereo → mono
        data = data.mean(axis=1)
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

# --- Streamlit UI ---
st.title("📞 Digital Arrest Detector (Calls + Text)")
st.write("Now supports **Text + Audio (EN-IN, HI, GU)** analysis.")

# Tabs: Text / Audio
tab1, tab2 = st.tabs(["📝 Text Analysis", "🎙 Audio Call Analysis"])

with tab1:
    user_input = st.text_area("Paste text (SMS/Email/Chat) here:")
    if st.button("Analyze Text"):
        if user_input.strip():
            label, scam_prob, keywords = detect_message(user_input)
            st.subheader("🔍 Text Analysis Result")
            st.write(f"**Classification:** {label}")
            st.write(f"**ML Scam Probability:** {scam_prob:.2f}")
            st.write(f"**Risky Keywords:** {', '.join(keywords) if keywords else 'None'}")

with tab2:
    uploaded_audio = st.file_uploader("Upload call recording (wav/mp3)", type=["wav", "mp3"])
    lang_choice = st.selectbox("Select Language Model", ["en-in", "gu", "hi"])

    if uploaded_audio and st.button("Analyze Audio"):
        with st.spinner("Transcribing audio..."):
            transcription = transcribe_audio(uploaded_audio, lang=lang_choice)
        st.subheader("🗣 Transcribed Text")
        st.write(transcription if transcription else "❌ No speech detected.")

        if transcription:
            label, scam_prob, keywords = detect_message(transcription)
            st.subheader("🔍 Scam Analysis on Transcription")
            st.write(f"**Classification:** {label}")
            st.write(f"**ML Scam Probability:** {scam_prob:.2f}")
            st.write(f"**Risky Keywords:** {', '.join(keywords) if keywords else 'None'}")

            # --- Placeholder for deepfake detection ---
            st.subheader("🤖 Deepfake Audio Detection")
            st.info("⚠️ Placeholder: Deepfake voice detection model to be integrated here.")
