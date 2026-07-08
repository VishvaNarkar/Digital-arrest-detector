"""Text analysis router."""
from fastapi import APIRouter, HTTPException

from backend.core.detector import categorize_scam, detect_message
from backend.schemas import TextAnalyzeRequest, TextAnalyzeResponse

router = APIRouter(tags=["Text"])


@router.post("/analyze/text", response_model=TextAnalyzeResponse)
def analyze_text(body: TextAnalyzeRequest) -> TextAnalyzeResponse:
    """Analyse a text message for scam indicators."""
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    result = detect_message(body.text, threshold=body.threshold)
    category = categorize_scam(
        result["keywords"], result["sentiment"]
    ) if result["keywords"] else "No Category"

    return TextAnalyzeResponse(
        label=result["label"],
        category=category,
        ml_prob=result["ml_prob"],
        keyword_score=result["keyword_score"],
        combined_prob=result["combined_prob"],
        keywords=result["keywords"],
        sentiment=result["sentiment"],
        rag=result.get("rag"),
    )
