"""Audio transcription — extracted from app_streamlit.py, Streamlit-free."""
import json
import logging
import os
import tempfile
import wave

import soundfile as sf
from vosk import KaldiRecognizer, Model

from backend.core.models import VOSK_BASE_PATH

logger = logging.getLogger(__name__)


def transcribe_audio(audio_bytes: bytes, lang: str = "en-in") -> str:
    """Transcribe audio bytes to text using an offline Vosk model.

    Raises
    ------
    FileNotFoundError
        If the Vosk model for the requested language is missing.
    ValueError
        If the audio data cannot be read or transcription fails.
    """
    model_path = VOSK_BASE_PATH / f"vosk-model-small-{lang}"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Vosk model not found: {model_path}. "
            f"Download it from https://alphacephei.com/vosk/models"
        )

    try:
        vosk_model = Model(str(model_path))
    except Exception as exc:
        raise ValueError(f"Failed to load Vosk model: {exc}") from exc

    # Write incoming bytes to a temp file so soundfile can read it
    tmp_in_fd, tmp_in_path = tempfile.mkstemp(suffix=".audio")
    tmp_wav_fd, tmp_wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_wav_fd)
    result_text = ""

    try:
        with os.fdopen(tmp_in_fd, "wb") as fh:
            fh.write(audio_bytes)

        try:
            data, samplerate = sf.read(tmp_in_path)
        except Exception as exc:
            raise ValueError(f"Could not read audio data: {exc}") from exc

        if len(data.shape) > 1:
            data = data.mean(axis=1)  # stereo → mono

        sf.write(tmp_wav_path, data, samplerate)

        with wave.open(tmp_wav_path, "rb") as wf:
            rec = KaldiRecognizer(vosk_model, wf.getframerate())
            rec.SetWords(False)
            while True:
                chunk = wf.readframes(4000)
                if not chunk:
                    break
                if rec.AcceptWaveform(chunk):
                    result_text += " " + json.loads(rec.Result()).get("text", "")
            result_text += " " + json.loads(rec.FinalResult()).get("text", "")

    except (FileNotFoundError, ValueError):
        raise
    except Exception as exc:
        logger.exception("Vosk transcription error: %s", exc)
        raise ValueError(f"Transcription failed: {exc}") from exc
    finally:
        for p in (tmp_in_path, tmp_wav_path):
            try:
                os.unlink(p)
            except OSError:
                pass

    result_text = result_text.strip()

    # Optional LLM refinement
    if result_text:
        try:
            from utils.rag_utils import is_ollama_available, rag_refine_transcription
            if is_ollama_available():
                refined = rag_refine_transcription(result_text)
                result_text = refined.get("cleaned_text") or result_text
        except Exception:
            pass  # fall back to raw transcription

    return result_text
