"""
retrieval.py - Evidence retrieval using LaBSE embeddings + FAISS.
Optimized: batch encode all claims at once instead of one-by-one.
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import List, Optional

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
    def __init__(self, config: dict):
        self.config = config
        self.embed_cfg    = config["embedding"]
        self.retrieval_cfg = config["retrieval"]
        self._model    = None
        self._index    = None
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
        import faiss
        logger.info(f"Building FAISS index for {len(passages)} passages.")
        self._passages = [Passage(**p) for p in passages]
        texts = [p.text for p in self._passages]
        embeddings = self.encode(texts)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype(np.float32))
        if save:
            self._save_index()
        logger.info("FAISS index built successfully.")

    def _save_index(self):
        import faiss
        index_path   = self.retrieval_cfg["faiss_index_path"]
        passages_path = self.retrieval_cfg["passages_path"]
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self._index, index_path)
        with open(passages_path, "w", encoding="utf-8") as f:
            for p in self._passages:
                f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
        logger.info(f"Index saved to {index_path}")

    def load_index(self):
        import faiss
        index_path    = self.retrieval_cfg["faiss_index_path"]
        passages_path = self.retrieval_cfg["passages_path"]
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}. Run build_kb.py first.")
        self._index = faiss.read_index(index_path)
        self._passages = []
        with open(passages_path, encoding="utf-8") as f:
            for line in f:
                self._passages.append(Passage(**json.loads(line)))
        logger.info(f"Loaded FAISS index with {len(self._passages)} passages.")

    def retrieve_evidence(self, claim_text: str, k: int = 5) -> List[Passage]:
        """Single claim retrieval."""
        if self._index is None:
            self.load_index()
        query_vec = self.encode([claim_text]).astype(np.float32)
        scores, indices = self._index.search(query_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._passages):
                continue
            p = Passage(**asdict(self._passages[idx]))
            p.similarity = float(score)
            results.append(p)
        return results

    def retrieve_evidence_batch(self, claim_texts: List[str], k: int = 5) -> List[List[Passage]]:
        """
        Retrieve evidence for ALL claims in a single FAISS search.
        Much faster than calling retrieve_evidence() in a loop.
        """
        if self._index is None:
            self.load_index()

        if not claim_texts:
            return []

        # Encode all claims at once
        query_vecs = self.encode(claim_texts).astype(np.float32)
        all_scores, all_indices = self._index.search(query_vecs, k)

        batch_results = []
        for scores, indices in zip(all_scores, all_indices):
            passages = []
            for score, idx in zip(scores, indices):
                if idx < 0 or idx >= len(self._passages):
                    continue
                p = Passage(**asdict(self._passages[idx]))
                p.similarity = float(score)
                passages.append(p)
            batch_results.append(passages)

        return batch_results


# Singleton
_index: Optional[EmbeddingIndex] = None


def get_index(config_path: str = "config.yaml") -> EmbeddingIndex:
    global _index
    if _index is None:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        _index = EmbeddingIndex(config)
    return _index


def retrieve_evidence(claim_text: str, k: int = 5, config_path: str = "config.yaml") -> List[Passage]:
    return get_index(config_path).retrieve_evidence(claim_text, k)
