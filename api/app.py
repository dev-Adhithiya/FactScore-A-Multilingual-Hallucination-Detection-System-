"""
api/app.py - FastAPI REST API for the hallucination detection system.
"""

import logging
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Satya Vaani - Hallucination Detection API",
    description="Multilingual hallucination detection and reliability scoring.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ────────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    prompt: str = Field(..., min_length=5, max_length=2000, description="The factual question or prompt.")
    language: str = Field(default="English", description="Target language for generation.")
    config_path: str = Field(default="config.yaml", description="Path to config.yaml.")


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_sec: float


# ── Startup ───────────────────────────────────────────────────────────────────

_start_time = time.time()


@app.on_event("startup")
async def startup_event():
    logger.info("Satya Vaani API starting up.")
    # Models load lazily on first request to keep startup fast.


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
def root():
    return {"message": "Satya Vaani Hallucination Detection API. See /docs for usage."}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(
        status="ok",
        version="1.0.0",
        uptime_sec=round(time.time() - _start_time, 1),
    )


@app.post("/analyze", tags=["Detection"])
async def analyze(request: AnalyzeRequest):
    """
    Run the full hallucination detection pipeline on a prompt.

    Returns:
        - generated_text
        - per-claim risk scores
        - document-level hallucination rate
    """
    try:
        from core.pipeline import run_pipeline

        result = run_pipeline(
            prompt=request.prompt,
            language=request.language,
            config_path=request.config_path,
        )
        return result

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Knowledge base not found. Run `python scripts/build_kb.py` first. Details: {e}",
        )
    except Exception as e:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/build-kb", tags=["Admin"])
async def build_knowledge_base(background_tasks: BackgroundTasks, config_path: str = "config.yaml"):
    """
    Trigger knowledge base (re)build in the background.
    """
    def _build():
        from scripts.build_kb import build_kb
        build_kb(config_path)

    background_tasks.add_task(_build)
    return {"message": "Knowledge base build started in background."}


if __name__ == "__main__":
    import uvicorn
    import yaml

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    uvicorn.run(
        "api.app:app",
        host=cfg["api"]["host"],
        port=cfg["api"]["port"],
        workers=cfg["api"]["workers"],
        reload=False,
    )
