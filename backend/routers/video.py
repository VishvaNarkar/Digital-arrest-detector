"""Video deepfake analysis router."""
from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.core.deepfake import detect_deepfake
from backend.schemas import VideoAnalyzeResponse

router = APIRouter(tags=["Video"])


@router.post("/analyze/video", response_model=VideoAnalyzeResponse)
async def analyze_video(
    file: UploadFile = File(..., description="MP4, AVI, or MOV video file"),
) -> VideoAnalyzeResponse:
    """Run deepfake detection on a video file."""
    video_bytes = await file.read()
    if not video_bytes:
        raise HTTPException(status_code=422, detail="Uploaded video file is empty.")

    try:
        label, score = detect_deepfake(video_bytes)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return VideoAnalyzeResponse(
        label=label,
        score=score,
        percent=round(score * 100, 1),
    )
