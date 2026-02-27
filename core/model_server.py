"""
model_server.py - Persistent model server.
Loads all models ONCE at startup and keeps them in memory forever.

Run this FIRST before starting the dashboard:
    python core/model_server.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hallucination Detection Model Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline = None


@app.on_event("startup")
async def startup():
    global _pipeline
    logger.info("=" * 60)
    logger.info("Loading all models into memory — do this ONCE.")
    logger.info("Do NOT close this terminal while using the dashboard.")
    logger.info("=" * 60)

    from core.pipeline import Pipeline
    _pipeline = Pipeline("config.yaml")

    # Force-load all models immediately so first prompt is fast
    from core.generation import get_engine
    from core.retrieval import get_index
    from core.entailment import get_checker

    get_engine("config.yaml")._load()
    get_index("config.yaml").load_index()
    get_checker("config.yaml")._load()

    logger.info("=" * 60)
    logger.info("ALL MODELS LOADED. Server is ready.")
    logger.info("Now start the dashboard: python -m streamlit run dashboard/dashboard.py")
    logger.info("=" * 60)


class AnalyzeRequest(BaseModel):
    prompt: str
    language: str = "English"


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if _pipeline is None:
        return {"error": "Models still loading, please wait."}
    return _pipeline.run_pipeline(req.prompt, req.language)


@app.get("/health")
def health():
    return {
        "status": "ready" if _pipeline is not None else "loading",
        "models_loaded": _pipeline is not None,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, workers=1)
