"""
scoring.py - Compute composite hallucination risk scores per claim.
"""

import re
import logging
from typing import List

import yaml

from core.claim_extraction import Claim
from core.retrieval import Passage

logger = logging.getLogger(__name__)


def _entity_overlap_ratio(claim_entities: List[str], passage_text: str) -> float:
    """Fraction of claim entities found in the passage text."""
    if not claim_entities:
        return 0.0
    passage_lower = passage_text.lower()
    hits = sum(1 for e in claim_entities if e.lower() in passage_lower)
    return hits / len(claim_entities)


def _number_match_ratio(claim_numbers: List[str], passage_text: str) -> float:
    """Fraction of claim numbers found in the passage text."""
    if not claim_numbers:
        return 1.0  # No numbers to verify — don't penalize
    hits = sum(1 for n in claim_numbers if n in passage_text)
    return hits / len(claim_numbers)


def compute_retrieval_score(
    claim: Claim,
    passages: List[Passage],
    weights: dict,
) -> float:
    """
    Compute composite retrieval score for a claim.

    retrieval_score = 0.5 * cosine + 0.3 * entity_overlap + 0.2 * number_match
    """
    if not passages:
        return 0.0

    best = max(passages, key=lambda p: p.similarity)
    cosine = float(best.similarity)
    entity = _entity_overlap_ratio(claim.entities, best.text)
    number = _number_match_ratio(claim.numbers, best.text)

    score = (
        weights["cosine"] * cosine
        + weights["entity_overlap"] * entity
        + weights["number_match"] * number
    )
    return round(min(max(score, 0.0), 1.0), 4)


def compute_risk_score(
    entailment_score: float,
    retrieval_score: float,
    drift_score: float,
    weights: dict,
) -> float:
    """
    Compute final hallucination risk score.

    Risk = 0.4*(1 - entailment) + 0.3*(1 - retrieval) + 0.3*drift
    """
    risk = (
        weights["entailment"] * (1.0 - entailment_score)
        + weights["retrieval"] * (1.0 - retrieval_score)
        + weights["drift"] * drift_score
    )
    return round(min(max(risk, 0.0), 1.0), 4)


def load_scoring_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)["scoring"]
