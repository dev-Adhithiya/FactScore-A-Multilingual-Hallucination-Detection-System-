"""
claim_extraction.py - Extract atomic, verifiable claims from text.
"""

import re
import uuid
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional

logger = logging.getLogger(__name__)

SUPPORTED_INDIC = {"tamil", "hindi", "bengali", "telugu", "kannada", "malayalam", "marathi"}


@dataclass
class Claim:
    claim_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    entities: List[str] = field(default_factory=list)
    numbers: List[str] = field(default_factory=list)
    sentence_index: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class ClaimExtractor:
    def __init__(self):
        self._spacy_en = None
        self._stanza_models = {}

    def _load_spacy_en(self):
        if self._spacy_en is None:
            import spacy
            try:
                self._spacy_en = spacy.load("en_core_web_sm")
            except OSError:
                import subprocess, sys
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
                self._spacy_en = spacy.load("en_core_web_sm")
        return self._spacy_en

    def _load_stanza(self, lang_code: str):
        if lang_code not in self._stanza_models:
            import stanza
            stanza.download(lang_code, processors="tokenize,ner", verbose=False)
            self._stanza_models[lang_code] = stanza.Pipeline(
                lang_code, processors="tokenize,ner", verbose=False
            )
        return self._stanza_models[lang_code]

    @staticmethod
    def _extract_numbers(text: str) -> List[str]:
        """Extract all numeric tokens from text."""
        return re.findall(r"\b\d[\d,\.]*\b", text)

    def _extract_english(self, text: str) -> List[Claim]:
        nlp = self._load_spacy_en()
        doc = nlp(text)
        claims = []
        for idx, sent in enumerate(doc.sents):
            sent_text = sent.text.strip()
            if len(sent_text.split()) < 4:
                continue
            entities = [ent.text for ent in sent.ents]
            numbers = self._extract_numbers(sent_text)
            claims.append(Claim(
                text=sent_text,
                entities=entities,
                numbers=numbers,
                sentence_index=idx,
            ))
        return claims

    def _extract_indic(self, text: str, lang_code: str) -> List[Claim]:
        try:
            nlp = self._load_stanza(lang_code)
            doc = nlp(text)
            claims = []
            for idx, sent in enumerate(doc.sentences):
                sent_text = sent.text.strip()
                if len(sent_text.split()) < 3:
                    continue
                entities = [ent.text for ent in sent.ents]
                numbers = self._extract_numbers(sent_text)
                claims.append(Claim(
                    text=sent_text,
                    entities=entities,
                    numbers=numbers,
                    sentence_index=idx,
                ))
            return claims
        except Exception as e:
            logger.warning(f"Stanza failed for {lang_code}: {e}. Falling back to regex segmentation.")
            return self._fallback_segment(text)

    def _fallback_segment(self, text: str) -> List[Claim]:
        """Regex-based sentence segmentation fallback."""
        sentences = re.split(r"(?<=[।.!?])\s+", text.strip())
        claims = []
        for idx, sent in enumerate(sentences):
            sent = sent.strip()
            if len(sent.split()) < 3:
                continue
            claims.append(Claim(
                text=sent,
                entities=[],
                numbers=self._extract_numbers(sent),
                sentence_index=idx,
            ))
        return claims

    LANG_CODE_MAP = {
        "tamil": "ta",
        "hindi": "hi",
        "bengali": "bn",
        "telugu": "te",
        "kannada": "kn",
        "malayalam": "ml",
        "marathi": "mr",
    }

    def extract_claims(self, text: str, language: str) -> List[Claim]:
        """
        Extract atomic claims from generated text.

        Args:
            text: Generated text.
            language: Language name (e.g., 'Tamil', 'Hindi', 'English').

        Returns:
            List of Claim objects.
        """
        if not text or not text.strip():
            logger.warning("Empty text passed to claim extractor.")
            return []

        lang_lower = language.lower()

        if lang_lower == "english":
            claims = self._extract_english(text)
        elif lang_lower in SUPPORTED_INDIC:
            lang_code = self.LANG_CODE_MAP.get(lang_lower, "hi")
            claims = self._extract_indic(text, lang_code)
        else:
            # Try English pipeline as a best-effort for unknown languages
            logger.info(f"Unknown language '{language}'; using English NLP pipeline.")
            claims = self._extract_english(text)

        logger.info(f"Extracted {len(claims)} claims from {language} text.")
        return claims


# Module-level singleton
_extractor: Optional[ClaimExtractor] = None


def get_extractor() -> ClaimExtractor:
    global _extractor
    if _extractor is None:
        _extractor = ClaimExtractor()
    return _extractor


def extract_claims(text: str, language: str) -> List[Claim]:
    return get_extractor().extract_claims(text, language)
