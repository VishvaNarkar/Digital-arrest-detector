"""Deepfake video detection — extracted from app_streamlit.py, Streamlit-free."""
import logging
import os
import tempfile
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def detect_deepfake(video_bytes: bytes, sample_frames: int = 12) -> Tuple[str, float]:
    """Detect whether a video is likely a deepfake.

    Parameters
    ----------
    video_bytes:
        Raw bytes of the uploaded video file.
    sample_frames:
        Number of evenly-spaced frames to sample for prediction.

    Returns
    -------
    (label, score)
        label is a human-readable verdict; score is the mean sigmoid probability.

    Raises
    ------
    RuntimeError
        If the deepfake model is not loaded.
    ValueError
        If the video cannot be opened or has no readable frames.
    """
    from backend.core import models as _m
    if _m.deepfake_model is None:
        raise RuntimeError(
            "Deepfake model is not loaded. "
            "Ensure 'models/Deepfakes_detection_model.keras' exists."
        )

    # Write bytes to a temp file so OpenCV can open it
    suffix = ".mp4"  # default; caller should pass correct extension via filename
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(tmp_fd, "wb") as fh:
            fh.write(video_bytes)

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file — unsupported codec or corrupt file.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError("Video has no readable frames.")

        frame_idxs = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
        preds = []

        for idx in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (224, 224)) / 255.0
            tensor = np.expand_dims(frame_resized, axis=0)
            prob = float(_m.deepfake_model.predict(tensor, verbose=0)[0][0])
            preds.append(prob)

        cap.release()

        if not preds:
            raise ValueError("No frames could be decoded from the video.")

        avg_score = float(np.mean(preds))
        label = "Likely Deepfake" if avg_score > 0.5 else "Likely Real"
        return label, avg_score

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
