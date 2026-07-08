"""Global model singletons loaded once at FastAPI startup.

Import the module-level variables (text_model, vectorizer, deepfake_model)
from anywhere in the backend; call load_all() inside the lifespan context.
"""
import logging
from pathlib import Path

import joblib
import tensorflow as tf

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = _REPO_ROOT / "models"
TEXT_MODEL_PATH = MODEL_DIR / "text_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
DEEPFAKE_MODEL_PATH = MODEL_DIR / "Deepfakes_detection_model.keras"
VOSK_BASE_PATH = MODEL_DIR

# ── Singletons (populated by load_all) ───────────────────────────────────────
text_model = None
vectorizer = None
deepfake_model = None


def load_all() -> None:
    """Load all models into module-level singletons.  Called once at startup."""
    global text_model, vectorizer, deepfake_model

    # Text / TF-IDF models
    try:
        text_model = joblib.load(TEXT_MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        logger.info("Text model and vectorizer loaded.")
    except Exception as exc:
        logger.error("Failed to load text models: %s", exc)
        text_model = None
        vectorizer = None

    # Deepfake Keras model
    try:
        deepfake_model = tf.keras.models.load_model(
            DEEPFAKE_MODEL_PATH, compile=False
        )
        logger.info("Deepfake model loaded.")
    except Exception as exc:
        logger.error("Failed to load deepfake model: %s", exc)
        deepfake_model = None
