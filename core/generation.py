"""
generation.py - Text generation engine using LLM.
"""

import logging
import time
from typing import Optional
import yaml

logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class GenerationEngine:
    """Lazy-loaded generation engine to avoid startup overhead."""

    def __init__(self, config: dict):
        self.config = config["generation"]
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_id = self.config["model"]
        device = self.config.get("device", "auto")

        logger.info(f"Loading generation model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Determine device
        if device == "auto":
            device_map = "auto" if torch.cuda.is_available() else "cpu"
        else:
            device_map = device

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype="auto",
        )
        self._loaded = True
        logger.info(f"Generation model loaded on {device_map}")

    def generate_text(self, prompt: str, language: str) -> str:
        """
        Generate factual text for a given prompt and language.

        Args:
            prompt: The input prompt/question.
            language: Target language for generation.

        Returns:
            Generated text string.
        """
        self._load()
        import torch

        system_message = (
            f"You are a factual assistant. Respond in {language}. "
            "Provide accurate, well-structured factual information."
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]

        # Step 1: Convert chat messages to a single formatted string
        formatted = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Step 2: Tokenize the formatted string properly into tensors
        encoding = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        input_ids = encoding["input_ids"].to(self.model.device)
        attention_mask = encoding["attention_mask"].to(self.model.device)

        params = {
            "max_new_tokens": self.config["max_new_tokens"],
            "temperature": self.config["temperature"],
            "top_p": self.config["top_p"],
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        start = time.time()
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **params,
            )

        elapsed = time.time() - start
        input_len = input_ids.shape[1]
        generated = output[0][input_len:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        logger.info(
            {
                "event": "generation_complete",
                "language": language,
                "prompt_preview": prompt[:80],
                "output_tokens": len(generated),
                "elapsed_sec": round(elapsed, 2),
                "model": self.config["model"],
                "temperature": self.config["temperature"],
                "max_new_tokens": self.config["max_new_tokens"],
                "top_p": self.config["top_p"],
            }
        )

        return text.strip()


# Module-level singleton (instantiated lazily)
_engine: Optional[GenerationEngine] = None


def get_engine(config_path: str = "config.yaml") -> GenerationEngine:
    global _engine
    if _engine is None:
        config = load_config(config_path)
        _engine = GenerationEngine(config)
    return _engine


def generate_text(prompt: str, language: str, config_path: str = "config.yaml") -> str:
    """Convenience wrapper."""
    return get_engine(config_path).generate_text(prompt, language)
