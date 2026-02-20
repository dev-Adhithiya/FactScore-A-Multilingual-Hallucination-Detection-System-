"""
drift.py - Cross-lingual semantic drift measurement.
"""

import logging
from typing import List, Optional, Dict

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class DriftCalculator:
    def __init__(self, config: dict):
        self.config = config
        self.translation_cfg = config["translation"]
        self._translator = None
        self._embed_model = None

    def _load_translator(self):
        if self._translator is not None:
            return
        from transformers import pipeline as hf_pipeline
        logger.info(f"Loading translation model: {self.translation_cfg['model']}")
        self._translator = hf_pipeline(
            "translation",
            model=self.translation_cfg["model"],
            max_length=self.translation_cfg.get("max_length", 512),
        )

    def _load_embedder(self):
        if self._embed_model is not None:
            return
        from sentence_transformers import SentenceTransformer
        logger.info("Loading LaBSE for drift embedding.")
        self._embed_model = SentenceTransformer(self.config["embedding"]["model"])

    def translate_to_english(self, text: str) -> str:
        """Translate any language text to English."""
        self._load_translator()
        try:
            result = self._translator(text)
            if isinstance(result, list) and result:
                return result[0].get("translation_text", text)
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
        return text

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def compute_drift(
        self, original_text: str, language: str, sentence_indices: List[int]
    ) -> Dict[int, float]:
        """
        Compute cross-lingual drift for each sentence index.

        Args:
            original_text: Original generated text.
            language: Source language.
            sentence_indices: Sentence indices present in claims.

        Returns:
            Dict mapping sentence_index → drift score [0, 1].
        """
        if language.lower() == "english":
            # No cross-lingual drift for English
            return {idx: 0.0 for idx in sentence_indices}

        self._load_embedder()

        # Split original into sentences (simple split)
        import re
        sentences = re.split(r"(?<=[।.!?])\s+", original_text.strip())

        drift_map: Dict[int, float] = {}

        for idx in sentence_indices:
            if idx >= len(sentences):
                drift_map[idx] = 0.5  # Unknown
                continue

            src_sent = sentences[idx]
            translated_sent = self.translate_to_english(src_sent)

            src_vec = self._embed_model.encode(src_sent, normalize_embeddings=True)
            tgt_vec = self._embed_model.encode(translated_sent, normalize_embeddings=True)

            similarity = self._cosine_similarity(src_vec, tgt_vec)
            drift = round(1.0 - similarity, 4)
            drift_map[idx] = drift

        logger.info(f"Drift computed for {len(drift_map)} sentences. Mean drift: {np.mean(list(drift_map.values())):.3f}")
        return drift_map


# Singleton
_calculator: Optional[DriftCalculator] = None


def get_calculator(config_path: str = "config.yaml") -> DriftCalculator:
    global _calculator
    if _calculator is None:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        _calculator = DriftCalculator(config)
    return _calculator
