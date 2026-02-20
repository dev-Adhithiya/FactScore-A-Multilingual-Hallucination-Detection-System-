"""
entailment.py - NLI-based entailment scoring using XLM-RoBERTa.
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

        # Map label string → index (handles label order variations)
        labels = self._model.config.id2label
        self._label2id = {v.lower(): k for k, v in labels.items()}
        logger.info(f"NLI model loaded. Labels: {labels}")

    def compute_entailment(self, claim: str, passages: List[str]) -> float:
        """
        Compute maximum entailment probability of claim against passages.

        Args:
            claim: The claim string (hypothesis).
            passages: List of evidence passages (premises).

        Returns:
            Max entailment probability in [0, 1].
        """
        if not passages:
            return 0.0

        self._load()
        import torch

        max_prob = 0.0
        entail_key = next(
            (k for k in self._label2id if "entail" in k), "entailment"
        )
        entail_idx = self._label2id.get(entail_key, 0)

        for passage in passages:
            encoding = self._tokenizer(
                passage,
                claim,
                return_tensors="pt",
                max_length=self.config.get("max_length", 512),
                truncation=True,
                padding=True,
            ).to(self._device)

            with torch.no_grad():
                logits = self._model(**encoding).logits
                probs = torch.softmax(logits, dim=-1)[0]
                entail_prob = probs[entail_idx].item()

            if entail_prob > max_prob:
                max_prob = entail_prob

        return round(max_prob, 4)


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
