"""
pipeline.py - Orchestrates the full hallucination detection pipeline.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


def _setup_logging(log_dir: str = "logs"):
    import os, sys
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"pipeline_{datetime.now().strftime('%Y%m%d')}.jsonl")

    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        root.addHandler(sh)

    # JSON file handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("pipeline_json").addHandler(fh)


def _log_json(data: dict):
    logging.getLogger("pipeline_json").info(json.dumps(data, ensure_ascii=False))


class Pipeline:
    def __init__(self, config_path: str = "config.yaml"):
        _setup_logging()
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.config_path = config_path
        self.scoring_cfg = self.config["scoring"]
        self.retrieval_cfg = self.config["retrieval"]
        self._models_ready = False

    def _load_models(self):
        if self._models_ready:
            return

        from core.generation import get_engine
        from core.claim_extraction import get_extractor
        from core.retrieval import get_index
        from core.entailment import get_checker
        from core.drift import get_calculator

        logger.info("Pre-loading all models...")
        get_engine(self.config_path)   # lazy — actual load on first call
        get_extractor()
        get_index(self.config_path)
        get_checker(self.config_path)
        get_calculator(self.config_path)
        self._models_ready = True
        logger.info("All models ready.")

    def run_pipeline(self, prompt: str, language: str) -> Dict[str, Any]:
        """
        Execute full hallucination detection pipeline.

        Args:
            prompt: User query.
            language: Target language.

        Returns:
            Structured result dict.
        """
        self._load_models()
        from core.generation import get_engine
        from core.claim_extraction import get_extractor
        from core.retrieval import get_index
        from core.entailment import get_checker
        from core.drift import get_calculator
        from core.scoring import (
            compute_retrieval_score,
            compute_risk_score,
            load_scoring_config,
        )

        t_start = time.time()
        scoring_cfg = load_scoring_config(self.config_path)

        # ── Step 1: Generate text ─────────────────────────────────────────
        logger.info(f"[1/5] Generating text in {language}...")
        generated_text = get_engine(self.config_path).generate_text(prompt, language)

        # ── Step 2: Extract claims ────────────────────────────────────────
        logger.info("[2/5] Extracting claims...")
        claims = get_extractor().extract_claims(generated_text, language)

        if not claims:
            logger.warning("No claims extracted.")
            return {
                "generated_text": generated_text,
                "claims": [],
                "document_hallucination_rate": 0.0,
                "elapsed_sec": round(time.time() - t_start, 2),
            }

        # ── Step 3: Compute cross-lingual drift (batch) ───────────────────
        logger.info("[3/5] Computing cross-lingual drift...")
        sentence_indices = list({c.sentence_index for c in claims})
        drift_map = get_calculator(self.config_path).compute_drift(
            generated_text, language, sentence_indices
        )

        # ── Steps 4 & 5: Per-claim retrieval, entailment, scoring ─────────
        logger.info("[4/5] Retrieving evidence and scoring claims...")
        index = get_index(self.config_path)
        checker = get_checker(self.config_path)
        k = self.retrieval_cfg.get("top_k", 5)

        claim_results = []
        for claim in claims:
            # Retrieve
            passages = index.retrieve_evidence(claim.text, k=k)
            passage_texts = [p.text for p in passages]

            # Entailment
            entailment_score = checker.compute_entailment(claim.text, passage_texts)

            # Retrieval score
            retrieval_score = compute_retrieval_score(
                claim, passages, scoring_cfg["retrieval_weights"]
            )

            # Drift
            drift_score = drift_map.get(claim.sentence_index, 0.0)

            # Risk
            risk_score = compute_risk_score(
                entailment_score,
                retrieval_score,
                drift_score,
                scoring_cfg["weights"],
            )

            is_hallucinated = risk_score > scoring_cfg["hallucination_threshold"]

            claim_results.append({
                "claim_id": claim.claim_id,
                "claim": claim.text,
                "entities": claim.entities,
                "numbers": claim.numbers,
                "sentence_index": claim.sentence_index,
                "entailment_score": entailment_score,
                "retrieval_score": retrieval_score,
                "drift_score": drift_score,
                "risk_score": risk_score,
                "hallucinated": is_hallucinated,
                "evidence": [p.to_dict() for p in passages[:3]],  # Top 3 for display
            })

        # ── Step 6: Aggregate ─────────────────────────────────────────────
        n_hallucinated = sum(1 for c in claim_results if c["hallucinated"])
        doc_hallucination_rate = round(n_hallucinated / len(claim_results), 4)
        elapsed = round(time.time() - t_start, 2)

        result = {
            "prompt": prompt,
            "language": language,
            "generated_text": generated_text,
            "claims": claim_results,
            "document_hallucination_rate": doc_hallucination_rate,
            "total_claims": len(claim_results),
            "hallucinated_claims": n_hallucinated,
            "elapsed_sec": elapsed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        _log_json(result)
        logger.info(
            f"Pipeline complete in {elapsed}s. "
            f"Hallucination rate: {doc_hallucination_rate:.1%} "
            f"({n_hallucinated}/{len(claim_results)} claims)"
        )
        return result


# Singleton
_pipeline: Optional[Pipeline] = None


def get_pipeline(config_path: str = "config.yaml") -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline(config_path)
    return _pipeline


def run_pipeline(prompt: str, language: str, config_path: str = "config.yaml") -> dict:
    return get_pipeline(config_path).run_pipeline(prompt, language)
