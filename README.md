# FraudShield AI — Multi-Channel Scam & Deepfake Detector

An AI-powered tool to **detect digital fraud, scams, and deepfakes** across multiple channels including **Text, Audio, and Video**.  
Built for real-time prevention, alerts, and awareness against modern cyber scams.

This project features a modern **React SPA frontend** styled with a premium light theme, backed by a robust **FastAPI backend** handling ML model inference and offline audio/video processing.

---

## Features

- **Text Scam Detection**
  - NLP-based classification (legit vs scam)
  - Keyword spotting for risky terms (300+ weights across English, Hindi, Gujarati)
  - Sentiment analysis for urgency/fear/threat signals
  - Optional LLM verification via local Ollama (Phi3:mini)
- **Audio Scam Detection**
  - Speech-to-text transcription using offline Vosk models (supports `en-in`, `hi`, `gu`)
  - ML-based scam probability scoring on transcribed text
  - Optional LLM transcription refinement via Ollama
- **Video Deepfake Detection**
  - Keras-based deepfake detection model (`Deepfakes_detection_model.keras`)
  - Samples 12 frames and averages probabilities to classify as **Likely Real / Deepfake**

---

## Tech Stack

- **Frontend**: React (Vite, Axios, Vanilla CSS, Inter Typography)
- **Backend**: FastAPI (Python), Uvicorn, Pydantic
- **ML/NLP**: Scikit-learn, TF-IDF, Keras / TensorFlow, NLTK VADER
- **Audio/Video**: Vosk, SoundFile, OpenCV, Wave
- **LLM / RAG**: Ollama + Phi3:mini (local, offline)

---

## Project Structure

```
Digital-arrest-detector/
├── backend/                  # FastAPI Application
│   ├── main.py               # API entrypoint & static serving
│   ├── schemas.py            # Pydantic schemas
│   ├── core/
│   │   ├── detector.py       # Text scam & keyword detection
│   │   ├── transcriber.py    # Vosk audio transcription
│   │   ├── deepfake.py       # OpenCV & Keras video classification
│   │   └── models.py         # Model loading singletons
│   └── routers/
│       ├── text.py           # Text analysis router
│       ├── audio.py          # Audio analysis router
│       └── video.py          # Video analysis router
├── frontend/                 # React SPA (Vite)
│   ├── dist/                 # Built production bundle (ignored)
│   ├── src/
│   │   ├── App.jsx           # Main layout & tab router
│   │   ├── index.css         # Light-theme design system
│   │   ├── api/client.js     # Axios client configuration
│   │   └── components/       # UI Components (Text, Audio, Video tabs)
│   └── vite.config.js        # Vite config with dev-time API proxy
├── models/                   # ML model files — see models/README.md
├── data/                     # Training dataset (sms_spam.csv)
├── utils/
│   └── rag_utils.py          # Ollama/RAG helper functions
├── tests/                    # Unit tests (pytest)
├── app_streamlit.py          # Stale/Legacy Streamlit UI backup
├── requirements.txt          # Core ML/processing dependencies
├── requirements-api.txt      # FastAPI backend dependencies
├── LICENSE
└── README.md
```

---

## Quick Start (Development)

### 1. Backend Setup
Create a virtual environment, install dependencies, and start the FastAPI server:

```bash
# 1. Create and activate venv
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 2. Install core + backend dependencies
pip install -r requirements.txt
pip install -r requirements-api.txt

# 3. Start the FastAPI API server
uvicorn backend.main:app --reload --port 8000
```
*API docs will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)*

### 2. Frontend Setup
In a new terminal window, install npm dependencies and start the Vite dev server:

```bash
cd frontend
npm install
npm run dev
```
*Access the app at [http://localhost:5173](http://localhost:5173) (requests to `/api` proxy automatically to `:8000`)*

---

## Production Build & Run

To run the entire app from a single command/port (FastAPI serving the compiled React build):

```bash
# 1. Compile the React frontend
cd frontend
npm run build
cd ..

# 2. Start the backend (mounts static files from frontend/dist)
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
*Access the production application directly at [http://127.0.0.1:8000/](http://127.0.0.1:8000/)*

---

## Optional — LLM Reasoning with Ollama

Enable Ollama to get detailed AI explanations for scam verdicts and transcript refinement:

```bash
# 1. Install Ollama: https://ollama.com
# 2. Pull the model (downloads ~2.2 GB)
ollama pull phi3:mini

# 3. Start the Ollama server (keep running in separate terminal)
ollama serve
```
*If Ollama is **offline**, FraudShield AI falls back gracefully to local ML/heuristics.*

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Retraining the Text Model

```bash
python train_text.py
```
This reads `data/sms_spam.csv`, trains a Logistic Regression classifier, and saves updated models to the `models/` directory.

---

## License

[MIT License](LICENSE) — free to use and modify with attribution.
