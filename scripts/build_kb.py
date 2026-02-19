"""
scripts/build_kb.py - Download Wikipedia dumps, chunk, embed, and index.

Usage:
    python scripts/build_kb.py [--config config.yaml] [--languages en hi ta] [--limit 5000]
"""

import argparse
import json
import logging
import os
import re
import sys
import uuid
from typing import Iterator, List

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def iter_wikipedia_passages(
    language_code: str, chunk_size: int = 250, limit: int = 10_000
) -> Iterator[dict]:
    """
    Stream Wikipedia articles via the MediaWiki API and yield chunked passages.

    Args:
        language_code: ISO 639-1 language code.
        chunk_size: Approximate token count per chunk (split by words).
        limit: Max number of articles to fetch.
    """
    import requests

    api_url = f"https://{language_code}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "random",
        "rnnamespace": 0,
        "rnlimit": min(limit, 50),
        "format": "json",
        "prop": "extracts",
        "explaintext": True,
    }

    fetched = 0
    session = requests.Session()
    session.headers.update({"User-Agent": "SatyaVaani/1.0 (hallucination-detection-research)"})

    while fetched < limit:
        try:
            rand_resp = session.get(api_url, params=params, timeout=10)
            rand_resp.raise_for_status()
            pages = rand_resp.json()["query"]["random"]
        except Exception as e:
            logger.error(f"Failed to fetch random pages for {language_code}: {e}")
            break

        for page in pages:
            if fetched >= limit:
                break
            page_id = page["id"]
            title = page["title"]

            try:
                extract_resp = session.get(api_url, params={
                    "action": "query",
                    "prop": "extracts",
                    "pageids": page_id,
                    "explaintext": True,
                    "format": "json",
                }, timeout=10)
                extract_resp.raise_for_status()
                pages_data = extract_resp.json()["query"]["pages"]
                extract = pages_data[str(page_id)].get("extract", "")
            except Exception as e:
                logger.warning(f"Skipping page {title}: {e}")
                continue

            # Clean
            extract = re.sub(r"\n{3,}", "\n\n", extract)
            extract = re.sub(r"==+[^=]+=+\n?", "", extract)
            extract = extract.strip()

            if len(extract) < 100:
                continue

            # Chunk by words
            words = extract.split()
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i : i + chunk_size])
                if len(chunk.split()) < 20:
                    continue
                yield {
                    "passage_id": str(uuid.uuid4()),
                    "language": language_code,
                    "text": chunk,
                    "source": f"Wikipedia:{language_code}:{title}",
                }

            fetched += 1
            if fetched % 100 == 0:
                logger.info(f"[{language_code}] Fetched {fetched} articles...")


def build_kb(config_path: str = "config.yaml", languages: List[str] = None, limit: int = 1000):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    kb_cfg = config.get("knowledge_base", {})
    chunk_size = kb_cfg.get("chunk_size", 250)
    lang_list = languages or kb_cfg.get("languages", ["en"])
    passages_path = config["retrieval"]["passages_path"]

    os.makedirs(os.path.dirname(passages_path), exist_ok=True)

    all_passages = []
    for lang in lang_list:
        logger.info(f"Fetching Wikipedia passages for language: {lang}")
        passages = list(iter_wikipedia_passages(lang, chunk_size=chunk_size, limit=limit))
        logger.info(f"  → {len(passages)} passages collected for {lang}")
        all_passages.extend(passages)

    logger.info(f"Total passages: {len(all_passages)}")

    # Write JSONL
    with open(passages_path, "w", encoding="utf-8") as f:
        for p in all_passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    logger.info(f"Passages saved to {passages_path}")

    # Build FAISS index
    from core.retrieval import EmbeddingIndex
    index = EmbeddingIndex(config)
    index.build_index(all_passages, save=True)
    logger.info("Knowledge base built successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Satya Vaani knowledge base.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--languages", nargs="+", default=None, help="e.g. en hi ta")
    parser.add_argument("--limit", type=int, default=1000, help="Articles per language.")
    args = parser.parse_args()

    build_kb(args.config, args.languages, args.limit)
