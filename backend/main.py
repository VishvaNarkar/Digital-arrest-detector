"""FraudShield AI — FastAPI application entry point.

Run (from repo root):
    uvicorn backend.main:app --reload
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.core import models as app_models
from backend.routers import audio, text, video


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML models once at startup; release on shutdown."""
    app_models.load_all()
    yield


app = FastAPI(
    title="FraudShield AI",
    description="Multi-channel fraud, scam & deepfake detection API",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (allow Vite dev server) ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(text.router, prefix="/api")
app.include_router(audio.router, prefix="/api")
app.include_router(video.router, prefix="/api")


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/api/health", tags=["Health"])
def health():
    from utils.rag_utils import is_ollama_available
    return {
        "ok": True,
        "ollama_available": is_ollama_available(),
        "text_model_loaded": app_models.text_model is not None,
        "deepfake_model_loaded": app_models.deepfake_model is not None,
    }


# ── Serve built React SPA (production) ───────────────────────────────────────
_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=_FRONTEND_DIST / "assets"),
        name="assets",
    )

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str):
        """Catch-all: serve React index.html for client-side routing."""
        return FileResponse(_FRONTEND_DIST / "index.html")
