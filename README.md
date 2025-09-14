# 📞 Digital Arrest Detector

An AI-powered tool to **detect digital fraud, scams, and deepfakes** across multiple channels including **Text, Audio, and Video**.  
Built for real-time prevention, alerts, and awareness against modern cyber scams.

---

## 🚀 Features

- **Text Scam Detection**
  - NLP-based classification (legit vs scam)
  - Keyword spotting for risky terms
  - Sentiment analysis for urgency/fear/threat

- **Audio Scam Detection**
  - Speech-to-text transcription (supports `en-in`, `hi`, `gu`)
  - ML-based scam probability scoring
  - Placeholder for deepfake/voice spoofing detection

- **Video Deepfake Detection**
  - Keras-based deepfake detection model (`Deepfakes_detection_model.keras`)
  - Classifies uploaded videos as **Likely Real / Deepfake**

- **Reputation Analysis**
  - Flags suspicious email domains and spoofed phone numbers
  - Caller/sender trust scoring

- **Feedback Learning**
  - Users can report suspicious communications
  - Models improve adaptively with community feedback

- **Real-Time Prevention**
  - Instant scam alerts
  - Protective advice (e.g., *“Do not share OTP”*, *“Verify caller ID”*)
  - Call blocking, email filtering, or warning overlays (planned)

---

## 🛠 Tech Stack

- **Backend**: Python (Streamlit / FastAPI optional)
- **ML/NLP**: Scikit-learn, TF-IDF, Keras
- **Audio Processing**: Vosk, SoundFile, Wave
- **Video Processing**: OpenCV, TensorFlow/Keras
- **Deployment Ready**: Streamlit (demo) / React or Vue (future)

---

## 📂 Project Structure

```
Digital-arrest-detector/
│── app.py                # Main app (Streamlit UI)
│── models/               # ML models (text, tfidf, deepfake, etc.)
│── detectors.py          # Scam detection logic
│── reputation.py         # Sender/Caller reputation checks
│── feedback.py           # User feedback + adaptive learning
│── blocklist.py          # Real-time prevention & blocking
│── requirements.txt      # Dependencies
│── README.md             # Project documentation
```

---

## ⚙️ Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/VishvaNarkar/Digital-arrest-detector.git
   cd Digital-arrest-detector
   ```

2. Create a virtual environment & install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## 🖥 Usage

- **Text Analysis** → Paste SMS/Email/Chat text to detect scams  
- **Audio Analysis** → Upload call recording (wav/mp3) → Transcription + Scam detection  
- **Video Analysis** → Upload video (mp4/avi) → Deepfake detection  
- **Feedback Tab** → Report suspicious cases to improve model  

---

## 📌 Future Roadmap

- ✅ Add deepfake detection (video)  
- 🔲 Integrate real-time call/email blocking  
- 🔲 Advanced multi-language NLP models  
- 🔲 Deploy full-stack version (FastAPI + React/Vue)  

---

## 👨‍💻 Contributors

- Team Name - **Traceroute Titans**   
- Hackathon Team Members:
- Ajaysinh Chauhan
- Himesh Nayak
- Yug Parmar
- Vishv Narkar

---

## 📜 License

MIT License – free to use and modify with attribution.
