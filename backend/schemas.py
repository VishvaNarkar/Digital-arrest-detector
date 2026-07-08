"""Pydantic request/response schemas for FraudShield AI API."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Request bodies ──────────────────────────────────────────────────────────

class TextAnalyzeRequest(BaseModel):
    text: str = Field(..., description="Message text to analyse")
    threshold: float = Field(0.35, ge=0.0, le=1.0, description="Scam probability threshold")


# ── Shared sub-models ───────────────────────────────────────────────────────

class SentimentScores(BaseModel):
    neg: float = 0.0
    neu: float = 0.0
    pos: float = 0.0
    compound: float = 0.0


class RagResult(BaseModel):
    llm_prob: Optional[float] = None
    verdict: Optional[str] = None
    risk_percent: Optional[Any] = None
    explanation: Optional[str] = None
    advice: List[str] = []
    raw: Optional[Any] = None


# ── Response bodies ─────────────────────────────────────────────────────────

class TextAnalyzeResponse(BaseModel):
    label: str
    category: str
    ml_prob: float
    keyword_score: int
    combined_prob: float
    keywords: List[str]
    sentiment: Dict[str, float]
    rag: Optional[RagResult] = None


class AudioAnalyzeResponse(BaseModel):
    transcription: str
    analysis: TextAnalyzeResponse


class VideoAnalyzeResponse(BaseModel):
    label: str
    score: float
    percent: float


class HealthResponse(BaseModel):
    ok: bool
    ollama_available: bool
    text_model_loaded: bool
    deepfake_model_loaded: bool
