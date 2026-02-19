"""
retrieval.py - Evidence retrieval using LaBSE embeddings + FAISS.
"""

import json
import logging
import os
import pickle
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class Passage:
    passage_id: str
    language: str
    text: str
    source: str
    similarity: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class EmbeddingIndex:
    """LaBSE + FAISS retrieval index."""

    def __init__(self, config: dict):
        self.config = config
        self.embed_cfg = config["embedding"]
        self.retrieval_cfg = config["retrieval"]
        self._model = None
        self._index = None
        self._passages: List[Passage] = []

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.embed_cfg['model']}")
            self._model = SentenceTransformer(self.embed_cfg["model"])
        return self._model

    def encode(self, texts: List[str]) -> np.ndarray:
        model = self._load_model()
        return model.encode(
            texts,
            batch_size=self.embed_cfg.get("batch_size", 64),
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def build_index(self, passages: List[dict], save: bool = True):
        """
        Build FAISS index from passage dicts.

        Args:
            passages: List of passage dicts with 'text', 'passage_id', 'language', 'source'.
            save: Whether to persist index to disk.
        """
        import faiss

        logger.info(f"Building FAISS index for {len(passages)} passages.")
        self._passages = [Passage(**p) for p in passages]
        texts = [p.text for p in self._passages]

        embeddings = self.encode(texts)
        dim = embeddings.shape[1]

        self._index = faiss.IndexFlatIP(dim)  # Inner product with normalized vecs = cosine
        self._index.add(embeddings.astype(np.float32))

        if save:
            self._save_index(embeddings)

        logger.info("FAISS index built successfully.")

    def _save_index(self, embeddings: np.ndarray):
        import faiss

        index_path = self.retrieval_cfg["faiss_index_path"]
        passages_path = self.retrieval_cfg["passages_path"]
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        faiss.write_index(self._index, index_path)
        with open(passages_path, "w", encoding="utf-8") as f:
            for p in self._passages:
                f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
        logger.info(f"Index saved to {index_path}")

    def load_index(self):
        """Load persisted FAISS index and passages from disk."""
        import faiss

        index_path = self.retrieval_cfg["faiss_index_path"]
        passages_path = self.retrieval_cfg["passages_path"]

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}. Run build_kb.py first.")

        self._index = faiss.read_index(index_path)
        self._passages = []
        with open(passages_path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self._passages.append(Passage(**data))

        logger.info(f"Loaded FAISS index with {len(self._passages)} passages.")

    def retrieve_evidence(self, claim_text: str, k: int = 5) -> List[Passage]:
        """
        Retrieve top-k evidence passages for a claim.

        Args:
            claim_text: The claim string to search for.
            k: Number of passages to retrieve.

        Returns:
            List of Passage objects with similarity scores.
        """
        if self._index is None:
            self.load_index()

        query_vec = self.encode([claim_text]).astype(np.float32)
        scores, indices = self._index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._passages):
                continue
            passage = self._passages[idx]
            passage.similarity = float(score)
            results.append(passage)

        return results


# Module-level singleton
_index: Optional[EmbeddingIndex] = None


def get_index(config_path: str = "config.yaml") -> EmbeddingIndex:
    global _index
    if _index is None:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        _index = EmbeddingIndex(config)
    return _index


def retrieve_evidence(
    claim_text: str, k: int = 5, config_path: str = "config.yaml"
) -> List[Passage]:
    return get_index(config_path).retrieve_evidence(claim_text, k)
