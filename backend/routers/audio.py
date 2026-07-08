"""Audio analysis router."""
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.core.detector import categorize_scam, detect_message
from backend.core.transcriber import transcribe_audio
from backend.schemas import AudioAnalyzeResponse, TextAnalyzeResponse

router = APIRouter(tags=["Audio"])

SUPPORTED_LANGS = {"en-in", "hi", "gu"}


@router.post("/analyze/audio", response_model=AudioAnalyzeResponse)
async def analyze_audio(
    file: UploadFile = File(..., description="WAV or MP3 audio file"),
    lang: str = Form("en-in", description="Language code: en-in | hi | gu"),
    threshold: float = Form(0.35, ge=0.0, le=1.0),
) -> AudioAnalyzeResponse:
    """Transcribe audio and run scam detection on the transcript."""
    if lang not in SUPPORTED_LANGS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported language '{lang}'. Choose from: {sorted(SUPPORTED_LANGS)}",
        )

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="Uploaded audio file is empty.")

    try:
        transcription = transcribe_audio(audio_bytes, lang=lang)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not transcription:
        raise HTTPException(
            status_code=422,
            detail="Transcription returned no text. Try a different file or language.",
        )

    result = detect_message(transcription, threshold=threshold)
    category = categorize_scam(
        result["keywords"], result["sentiment"]
    ) if result["keywords"] else "No Category"

    analysis = TextAnalyzeResponse(
        label=result["label"],
        category=category,
        ml_prob=result["ml_prob"],
        keyword_score=result["keyword_score"],
        combined_prob=result["combined_prob"],
        keywords=result["keywords"],
        sentiment=result["sentiment"],
        rag=result.get("rag"),
    )
    return AudioAnalyzeResponse(transcription=transcription, analysis=analysis)
