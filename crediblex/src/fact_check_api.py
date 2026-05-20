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

_ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

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
            api_key:   Google Fact Check API key. Falls back to the env var
                       ``GOOGLE_FACT_CHECK_API_KEY`` if not provided.
            rpm_limit: Token-bucket capacity (calls per minute). Default 60.

        Raises:
            ValueError: If no API key can be resolved from args or environment.
        """
        self.api_key: str = api_key or os.getenv("GOOGLE_FACT_CHECK_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Set GOOGLE_FACT_CHECK_API_KEY in your .env file "
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
        Search the Fact Check Tools API for claims matching *query*.

        The query is silently truncated to 500 characters (API limit).

        Args:
            query:         The claim or headline text to look up.
            language_code: BCP-47 language tag (e.g. ``"en"``, ``"hi"``).
            max_results:   Maximum number of results to request (1–10).

        Returns:
            A dict with keys::

                {
                    "found":     bool,          # True if at least one result returned
                    "rating":    str,            # Human-readable rating string
                    "publisher": str,            # Fact-checker organisation name
                    "url":       str,            # URL of the fact-check article
                    "score":     float,          # Normalised [0, 1] credibility score
                }

            If nothing is found or an error occurs the :meth:`_empty_signal`
            value is returned so callers never have to guard for ``None``.
        """
        self._rate_limit()

        query = query[:500]  # hard API limit

        params: dict = {
            "key": self.api_key,
            "query": query,
            "languageCode": language_code,
            "pageSize": max(1, min(max_results, 10)),
        }

        try:
            response = requests.get(
                _ENDPOINT,
                params=params,
                timeout=10,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            logger.warning("Fact Check API request timed out for query: %.80s", query)
            return self._empty_signal()
        except requests.exceptions.RequestException as exc:
            logger.error("Fact Check API request failed: %s", exc)
            return self._empty_signal()

        data: dict = response.json()
        claims: list = data.get("claims", [])

        if not claims:
            return self._empty_signal()

        # Use the first (highest-relevance) claim
        claim: dict = claims[0]
        review: dict = {}
        reviews: list = claim.get("claimReview", [])
        if reviews:
            review = reviews[0]

        rating_text: str = review.get("textualRating", "Unknown")
        publisher: str = review.get("publisher", {}).get("name", "Unknown")
        url: str = review.get("url", "")

        return {
            "found": True,
            "rating": rating_text,
            "publisher": publisher,
            "url": url,
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
