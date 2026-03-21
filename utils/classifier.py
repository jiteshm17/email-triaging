"""Ollama-based email classifier using OpenAI-compatible API and structured output."""

from __future__ import annotations

import logging

from openai import OpenAI

from .prompts import EmailClassification, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen2.5:14b"


def get_ollama_client(base_url: str = DEFAULT_BASE_URL, api_key: str = "ollama") -> OpenAI:
    """Return an OpenAI client configured for local Ollama."""
    return OpenAI(base_url=base_url, api_key=api_key)


def classify_email(
    client: OpenAI,
    subject: str,
    body: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0,
    seed: int = 42,
) -> tuple[str | None, str | None]:
    """
    Classify a single email. Returns (category, reason_short) or (None, None) on failure.
    """
    user_content = f"Subject: {subject}\nBody: {body}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    try:
        comp = client.chat.completions.parse(
            model=model,
            temperature=temperature,
            seed=seed,
            messages=messages,
            response_format=EmailClassification,
        )
        msg = comp.choices[0].message
        return msg.parsed.category, msg.parsed.reason_short
    except Exception as e:
        logger.warning("Classification failed for subject %r: %s", subject[:60], e)
        return None, None
