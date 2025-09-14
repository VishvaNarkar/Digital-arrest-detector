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

---

## 🛠 Tech Stack

- **Backend**: Python (Streamlit)
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
│── data/                 # Sample data for training
│── train_text.py         # Script to train text scam detection model
│── temp_video.mp4        # Example video file for deepfake detection
│── temp.wav              # Example audio file for scam detection
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

---

## 📌 Future Roadmap

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
