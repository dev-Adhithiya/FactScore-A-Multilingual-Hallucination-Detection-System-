"""
entailment.py - NLI-based entailment scoring using XLM-RoBERTa.
Optimized: batched inference instead of one forward pass per passage.
"""

import logging
from typing import List, Optional

import yaml

logger = logging.getLogger(__name__)


class EntailmentChecker:
    def __init__(self, config: dict):
        self.config = config["nli"]
        self._model = None
        self._tokenizer = None
        self._device = None
        self._label2id: dict = {}

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        model_id = self.config["model"]
        logger.info(f"Loading NLI model: {model_id}")
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        self._model.eval()

        labels = self._model.config.id2label
        self._label2id = {v.lower(): k for k, v in labels.items()}
        logger.info(f"NLI model loaded. Labels: {labels}")

    def compute_entailment(self, claim: str, passages: List[str]) -> float:
        """
        Compute max entailment probability — BATCHED for speed.
        All passages processed in a single forward pass instead of one-by-one.
        """
        if not passages:
            return 0.0

        self._load()
        import torch

        entail_key = next((k for k in self._label2id if "entail" in k), "entailment")
        entail_idx = self._label2id.get(entail_key, 0)

        # Batch all (passage, claim) pairs together
        encoding = self._tokenizer(
            passages,                          # premises (batch)
            [claim] * len(passages),           # hypothesis repeated
            return_tensors="pt",
            max_length=self.config.get("max_length", 512),
            truncation=True,
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**encoding).logits          # [N, 3]
            probs  = torch.softmax(logits, dim=-1)           # [N, 3]
            entail_probs = probs[:, entail_idx]              # [N]
            max_prob = entail_probs.max().item()

        return round(max_prob, 4)

    def compute_entailment_batch(
        self, claims: List[str], passages_per_claim: List[List[str]]
    ) -> List[float]:
        """
        Score multiple claims at once — one giant batch for all claims × passages.
        Dramatically faster than calling compute_entailment() in a loop.
        """
        if not claims:
            return []

        self._load()
        import torch

        entail_key = next((k for k in self._label2id if "entail" in k), "entailment")
        entail_idx = self._label2id.get(entail_key, 0)

        # Build flat list of (passage, claim) pairs, track which claim each belongs to
        pairs_premise   = []
        pairs_hypothesis = []
        claim_indices   = []  # which claim index each pair belongs to

        for claim_idx, (claim, passages) in enumerate(zip(claims, passages_per_claim)):
            for passage in passages:
                pairs_premise.append(passage)
                pairs_hypothesis.append(claim)
                claim_indices.append(claim_idx)

        if not pairs_premise:
            return [0.0] * len(claims)

        # Run in mini-batches to avoid OOM
        BATCH_SIZE = 16
        all_probs = []

        for start in range(0, len(pairs_premise), BATCH_SIZE):
            end = start + BATCH_SIZE
            encoding = self._tokenizer(
                pairs_premise[start:end],
                pairs_hypothesis[start:end],
                return_tensors="pt",
                max_length=self.config.get("max_length", 512),
                truncation=True,
                padding=True,
            ).to(self._device)

            with torch.no_grad():
                logits = self._model(**encoding).logits
                probs  = torch.softmax(logits, dim=-1)
                all_probs.extend(probs[:, entail_idx].tolist())

        # Aggregate: max entailment per claim
        results = [0.0] * len(claims)
        for prob, claim_idx in zip(all_probs, claim_indices):
            if prob > results[claim_idx]:
                results[claim_idx] = prob

        return [round(r, 4) for r in results]


# Singleton
_checker: Optional[EntailmentChecker] = None


def get_checker(config_path: str = "config.yaml") -> EntailmentChecker:
    global _checker
    if _checker is None:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        _checker = EntailmentChecker(config)
    return _checker


def compute_entailment(
    claim: str, passages: List[str], config_path: str = "config.yaml"
) -> float:
    return get_checker(config_path).compute_entailment(claim, passages)
