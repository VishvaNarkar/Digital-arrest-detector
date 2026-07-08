# 📞 Digital Arrest Detector

An AI-powered tool to **detect digital fraud, scams, and deepfakes** across multiple channels including **Text, Audio, and Video**.  
Built for real-time prevention, alerts, and awareness against modern cyber scams.

---

## 🚀 Features

- **Text Scam Detection**
  - NLP-based classification (legit vs scam)
  - Keyword spotting for risky terms (English, Hindi, Gujarati)
  - Sentiment analysis for urgency/fear/threat signals
  - Optional LLM verification via Ollama (Phi3:mini)

- **Audio Scam Detection**
  - Speech-to-text transcription (supports `en-in`, `hi`, `gu`)
  - ML-based scam probability scoring on transcribed text
  - Optional LLM transcription refinement via Ollama

- **Video Deepfake Detection**
  - Keras-based deepfake detection model (`Deepfakes_detection_model.keras`)
  - Classifies uploaded videos as **Likely Real / Deepfake**

---

## 🛠 Tech Stack

- **Backend**: Python (Streamlit)
- **ML/NLP**: Scikit-learn, TF-IDF, Keras / TensorFlow
- **Audio Processing**: Vosk, SoundFile, Wave
- **Video Processing**: OpenCV, TensorFlow/Keras
- **Optional LLM**: Ollama + Phi3:mini (local, offline)

---

## 📂 Project Structure

```
Digital-arrest-detector/
├── app.py                  # Main Streamlit UI
├── train_text.py           # Script to (re)train the text scam model
├── models/                 # ML model files — see models/README.md
├── data/                   # Training dataset (sms_spam.csv)
├── utils/
│   └── rag_utils.py        # Ollama/RAG helper functions
├── tests/                  # Unit tests (pytest)
├── requirements.txt        # Core Streamlit app dependencies
├── requirements-api.txt    # Optional FastAPI backend dependencies
├── LICENSE
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/VishvaNarkar/Digital-arrest-detector.git
cd Digital-arrest-detector
```

### 2. Create a virtual environment

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## 🤖 Optional — LLM Reasoning with Ollama

The app can use a locally running [Ollama](https://ollama.com) server to:
- Provide an LLM-based risk explanation for text scam results.
- Refine Vosk audio transcriptions for better accuracy.

If Ollama is **not running**, the app automatically falls back to
heuristic / ML-only results — no configuration required.

To enable it:

```bash
# 1. Install Ollama: https://ollama.com
# 2. Pull the model (downloads ~2 GB)
ollama pull phi3:mini

# 3. Keep the server running in a separate terminal
ollama serve
```

---

## 🖥 Usage

- **Text Analysis** → Paste SMS/Email/Chat text → get scam probability + keywords
- **Audio Analysis** → Upload a call recording (WAV/MP3) → transcription + scam scoring
- **Video Analysis** → Upload a video (MP4/AVI/MOV) → deepfake classification
- **Sidebar Threshold Slider** → Tune the scam detection sensitivity in real time

---

## 🔁 Retraining the Text Model

```bash
python train_text.py
```

This reads `data/sms_spam.csv`, trains a Logistic Regression classifier over
TF-IDF features, and saves updated `models/text_model.pkl` and
`models/tfidf_vectorizer.pkl`.

---

## 🧪 Running Tests

```bash
pip install pytest
pytest tests/
```

---

## 📌 Future Roadmap

- 🔲 Integrate real-time call/email blocking
- 🔲 Advanced multi-language NLP models
- 🔲 Deploy full-stack version (FastAPI + React/Vue)

---

## 📜 License

[MIT License](LICENSE) — free to use and modify with attribution.
