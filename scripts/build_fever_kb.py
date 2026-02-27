"""
scripts/build_fever_kb.py - Build knowledge base from Wikipedia using Poly-FEVER claim topics.

Optimized: parallel fetching with ThreadPoolExecutor instead of sequential requests.

Usage:
    python scripts/build_fever_kb.py --max_articles 5000 --workers 20
"""

import csv
import json
import logging
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TSV_PATH = os.path.join(_PROJECT_ROOT, "Poly-FEVER", "Poly-FEVER.tsv")
if not os.path.exists(TSV_PATH):
    TSV_PATH = "F:/Project/Poly-FEVER/Poly-FEVER.tsv"


def extract_entities_from_claims(max_rows=None):
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm")

    entities = set()
    n = 0

    logger.info(f"Extracting entities from {TSV_PATH}...")
    with open(TSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if max_rows and n >= max_rows:
                break
            claim = row.get("en", "").strip()
            if not claim:
                continue
            doc = nlp(claim)
            for ent in doc.ents:
                if ent.label_ in ("PERSON", "ORG", "GPE", "WORK_OF_ART", "EVENT", "FAC", "LOC", "PRODUCT"):
                    entities.add(ent.text.strip())
            for chunk in doc.noun_chunks:
                if chunk.text[0].isupper() and len(chunk.text.split()) <= 4:
                    entities.add(chunk.text.strip())
            n += 1

    logger.info(f"Extracted {len(entities)} unique entities from {n} claims.")
    return list(entities)


def fetch_wikipedia_article(args):
    """Fetch a single Wikipedia article — designed for thread pool use."""
    title, lang = args
    import requests
    session = requests.Session()
    session.headers.update({"User-Agent": "HallucinationDetector/1.0"})
    api_url = f"https://{lang}.wikipedia.org/w/api.php"
    try:
        resp = session.get(api_url, params={
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "format": "json",
            "redirects": 1,
        }, timeout=10)
        pages = resp.json()["query"]["pages"]
        page  = next(iter(pages.values()))
        if "missing" in page:
            return None
        extract = page.get("extract", "").strip()
        actual_title = page.get("title", title)
        return {"title": actual_title, "text": extract} if extract else None
    except Exception:
        return None


def fetch_random_titles(n=500, lang="en"):
    """Fetch a batch of random Wikipedia article titles."""
    import requests
    session = requests.Session()
    session.headers.update({"User-Agent": "HallucinationDetector/1.0"})
    api_url = f"https://{lang}.wikipedia.org/w/api.php"
    titles  = []
    while len(titles) < n:
        try:
            resp = session.get(api_url, params={
                "action": "query",
                "list": "random",
                "rnnamespace": 0,
                "rnlimit": min(500, n - len(titles)),
                "format": "json",
            }, timeout=10)
            titles.extend(p["title"] for p in resp.json()["query"]["random"])
        except Exception:
            break
    return titles


def article_to_passages(title, text, chunk_size=250):
    if not text or len(text) < 100:
        return []
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"==+[^=]+=+\n?", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words    = text.split()
    passages = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) < 20:
            continue
        passages.append({
            "passage_id": str(uuid.uuid4()),
            "language":   "en",
            "text":       chunk,
            "source":     f"Wikipedia:{title}",
        })
    return passages


def fetch_articles_parallel(titles, workers=20, lang="en"):
    """Fetch multiple Wikipedia articles in parallel."""
    all_passages   = []
    fetched_titles = set()
    args           = [(t, lang) for t in titles]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_wikipedia_article, a): a[0] for a in args}
        done = 0
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result:
                passages = article_to_passages(result["title"], result["text"])
                if passages:
                    all_passages.extend(passages)
                    fetched_titles.add(result["title"])
            if done % 200 == 0:
                logger.info(f"  {done}/{len(titles)} fetched | {len(fetched_titles)} articles | {len(all_passages)} passages")

    return all_passages, fetched_titles


def build_fever_kb(config_path="config.yaml", max_articles=5000, workers=20):
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    passages_path = config["retrieval"]["passages_path"]
    os.makedirs(os.path.dirname(passages_path), exist_ok=True)

    all_passages   = []
    fetched_titles = set()

    # ── Phase 1: Targeted fetch for Poly-FEVER entities ───────────────────
    logger.info("Phase 1: Extracting entities and fetching targeted Wikipedia articles...")
    t1 = time.time()

    entities = extract_entities_from_claims()
    logger.info(f"Fetching {len(entities)} entity articles in parallel (workers={workers})...")

    phase1_passages, phase1_titles = fetch_articles_parallel(
        entities[:max_articles // 2], workers=workers
    )
    all_passages.extend(phase1_passages)
    fetched_titles.update(phase1_titles)

    logger.info(f"Phase 1 done in {time.time()-t1:.0f}s — {len(fetched_titles)} articles, {len(all_passages)} passages")

    # ── Phase 2: Random articles to fill quota ────────────────────────────
    remaining = max_articles - len(fetched_titles)
    if remaining > 0:
        logger.info(f"Phase 2: Fetching {remaining} random Wikipedia articles (workers={workers})...")
        t2 = time.time()

        random_titles = fetch_random_titles(n=remaining + 200)  # fetch extra to account for misses
        random_titles = [t for t in random_titles if t not in fetched_titles]

        phase2_passages, phase2_titles = fetch_articles_parallel(
            random_titles[:remaining], workers=workers
        )
        all_passages.extend(phase2_passages)
        fetched_titles.update(phase2_titles)

        logger.info(f"Phase 2 done in {time.time()-t2:.0f}s — {len(phase2_titles)} articles added")

    logger.info(f"Total: {len(fetched_titles)} articles → {len(all_passages)} passages")

    # ── Save passages ─────────────────────────────────────────────────────
    logger.info(f"Saving passages to {passages_path}...")
    with open(passages_path, "w", encoding="utf-8") as f:
        for p in all_passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # ── Build FAISS index ─────────────────────────────────────────────────
    logger.info("Building FAISS index...")
    from core.retrieval import EmbeddingIndex
    index = EmbeddingIndex(config)
    index.build_index(all_passages, save=True)

    logger.info("=" * 55)
    logger.info("Knowledge base built successfully!")
    logger.info(f"  Articles : {len(fetched_titles)}")
    logger.info(f"  Passages : {len(all_passages)}")
    logger.info(f"  Index    : {config['retrieval']['faiss_index_path']}")
    logger.info("=" * 55)
    logger.info("Now run: python tests/test_poly_fever.py --language en --max_samples 50")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="config.yaml")
    parser.add_argument("--max_articles", default=5000, type=int)
    parser.add_argument("--workers",      default=20,   type=int,
                        help="Parallel fetch workers (default 20, increase for faster download)")
    args = parser.parse_args()
    build_fever_kb(args.config, args.max_articles, args.workers)
