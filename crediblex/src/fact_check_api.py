"""
Google Fact Check Tools API client for CredibleX v2.

Provides :class:`FactCheckClient` which wraps the Claim Search endpoint,
handles rate-limiting via a token-bucket algorithm, and normalises textual
ratings into a continuous [0, 1] score suitable for downstream model enrichment.

Environment variable:
    GOOGLE_FACT_CHECK_API_KEY: API key for the Fact Check Tools API.
"""

from __future__ import annotations

import os
import re
import time
import logging
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_ENDPOINT = "https://api.openai.com/v1/chat/completions"

# Textual rating -> [0, 1] score mapping (case-insensitive substring matching)
_RATING_MAP: dict[str, float] = {
    # Clearly TRUE
    "true": 1.0,
    "correct": 1.0,
    "verified": 1.0,
    "accurate": 1.0,
    "confirmed": 1.0,
    # Clearly FALSE
    "false": 0.0,
    "fake": 0.0,
    "fabricated": 0.0,
    "debunked": 0.0,
    "wrong": 0.0,
    "incorrect": 0.0,
    # MIXED / PARTIAL
    "misleading": 0.5,
    "half": 0.5,
    "partly": 0.5,
    "partially": 0.5,
    "mixed": 0.5,
    "unverified": 0.5,
    "unproven": 0.5,
    "disputed": 0.5,
    "questionable": 0.5,
}


class FactCheckClient:
    """
    Thin wrapper around the Google Fact Check Tools Claim Search API.

    Attributes:
        api_key:   The Google API key used for authenticated requests.
        rpm_limit: Maximum requests per minute (default: 60 for the free tier).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        rpm_limit: int = 60,
    ) -> None:
        """
        Initialise the client.

        Args:
            api_key:   OpenAI API key. Falls back to the env var
                       ``OPENAI_API_KEY`` if not provided.
            rpm_limit: Token-bucket capacity (calls per minute). Default 60.

        Raises:
            ValueError: If no API key can be resolved from args or environment.
        """
        self.api_key: str = api_key or os.getenv("OPENAI_API_KEY", os.getenv("GOOGLE_FACT_CHECK_API_KEY", ""))
        if not self.api_key:
            raise ValueError(
                "No API key provided. Set OPENAI_API_KEY in your .env file "
                "or pass api_key= to FactCheckClient()."
            )

        self.rpm_limit: int = rpm_limit
        # Token-bucket state
        self._tokens: float = float(rpm_limit)
        self._last_refill: float = time.monotonic()

    # ── Public Methods ────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        language_code: str = "en",
        max_results: int = 5,
    ) -> dict:
        """
        Search the ChatGPT API for claims matching *query*.
        
        Args:
            query:         The claim or headline text to look up.
            language_code: BCP-47 language tag (unused by ChatGPT, but kept for signature).
            max_results:   Unused by ChatGPT.

        Returns:
            A dict with keys::
                {
                    "found":     bool,
                    "rating":    str,
                    "publisher": str,
                    "url":       str,
                    "score":     float,
                }
        """
        self._rate_limit()

        query = query[:1000]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are an expert fact-checker. Determine the veracity of the following claim. Reply with exactly one word from the following: True, False, Misleading, Unverified. Do not explain."},
                {"role": "user", "content": f"Claim: {query}"}
            ],
            "temperature": 0.0,
            "max_tokens": 10
        }

        try:
            response = requests.post(
                _ENDPOINT,
                headers=headers,
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            logger.warning("ChatGPT API request timed out for query: %.80s", query)
            return self._empty_signal()
        except requests.exceptions.RequestException as exc:
            logger.error("ChatGPT API request failed: %s", exc)
            return self._empty_signal()

        data: dict = response.json()
        rating_text: str = data["choices"][0]["message"]["content"].strip()

        return {
            "found": True,
            "rating": rating_text,
            "publisher": "OpenAI ChatGPT",
            "url": "",
            "score": self._rating_to_score(rating_text),
        }

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _rate_limit(self) -> None:
        """
        Token-bucket rate limiter.

        Refills tokens proportionally to elapsed time (up to ``rpm_limit``).
        If the bucket is empty, sleeps until a token becomes available.
        """
        now = time.monotonic()
        elapsed = now - self._last_refill
        # Refill: rpm_limit tokens per 60 seconds
        self._tokens = min(
            float(self.rpm_limit),
            self._tokens + elapsed * (self.rpm_limit / 60.0),
        )
        self._last_refill = now

        if self._tokens < 1.0:
            sleep_duration = (1.0 - self._tokens) / (self.rpm_limit / 60.0)
            logger.debug("Rate limit reached — sleeping %.2fs", sleep_duration)
            time.sleep(sleep_duration)
            self._tokens = 0.0
        else:
            self._tokens -= 1.0

    def _rating_to_score(self, rating: str) -> float:
        """
        Map a textual fact-check rating to a continuous credibility score in [0, 1].

        Matching is case-insensitive and checks for substring containment so
        ratings like ``"Mostly False"`` correctly resolve to ``0.0`` and
        ``"Half True"`` resolves to ``0.5``.

        Args:
            rating: The raw textual rating string from the API response.

        Returns:
            float in [0, 1]. Returns 0.5 (neutral/unknown) if no match found.
        """
        lower = rating.lower()
        # Check all known keywords, longest match wins (more specific first)
        for keyword, score in sorted(_RATING_MAP.items(), key=lambda x: -len(x[0])):
            if re.search(r"\b" + re.escape(keyword) + r"\b", lower):
                return score
        return 0.5  # Unknown / no match → neutral prior

    def _empty_signal(self) -> dict:
        """
        Return a neutral/empty response when the API yields no results.

        Returns:
            dict: ``{ found: False, rating: "Unknown", publisher: "", url: "", score: 0.5 }``
        """
        return {
            "found": False,
            "rating": "Unknown",
            "publisher": "",
            "url": "",
            "score": 0.5,
        }
