"""
Web search via SERP API (SerpApi).

Provides a reusable search function that returns formatted context strings
suitable for injection into LLM prompts.  Gracefully degrades — if the API
key is not configured or the call fails, returns an empty string so agents
can proceed without web context.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _resolve_api_key(config_value: str) -> str:
    """Return the API key from config, falling back to the Dataiku env variable."""
    if config_value and config_value != "YOUR_SERPAPI_KEY":
        return config_value
    # Fallback: Dataiku project variable / OS environment variable
    key = os.environ.get("serp_api_key", "")
    if key:
        logger.info("SERP API key loaded from environment variable 'serp_api_key'")
    return key


class WebSearchClient:
    """Thin wrapper around the SerpApi Google Search endpoint."""

    def __init__(self, config: dict):
        serp_cfg = config.get("serp_api", {})
        self.api_key: str = _resolve_api_key(serp_cfg.get("api_key", ""))
        self.enabled: bool = serp_cfg.get("enabled", False) and bool(self.api_key)
        self.max_results: int = serp_cfg.get("max_results", 5)
        self.engine: str = serp_cfg.get("engine", "google")

    # ------------------------------------------------------------------
    # Public helpers — build a query, run the search, format the output
    # ------------------------------------------------------------------

    def search(self, query: str, num_results: int | None = None) -> str:
        """
        Run a web search and return results formatted as a text block.
        Returns empty string on any failure or if disabled.
        """
        if not self.enabled:
            return ""

        num = num_results or self.max_results
        try:
            results = self._call_serpapi(query, num)
        except Exception as e:
            logger.warning("Web search failed for query '%s': %s", query, e)
            return ""

        if not results:
            return ""

        return self._format_results(results, query)

    def search_for_skill(
        self,
        skill_id: str,
        params: dict,
        extra_terms: str = "",
    ) -> str:
        """
        Build a domain-appropriate search query from skill parameters and
        return formatted results.  Returns empty string if search is
        disabled or fails.
        """
        query = self._build_query(skill_id, params, extra_terms)
        if not query:
            return ""
        return self.search(query)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_serpapi(self, query: str, num: int) -> list[dict]:
        """Call SerpApi and return a list of organic result dicts."""
        try:
            from serpapi import GoogleSearch
        except ImportError:
            logger.warning(
                "serpapi package not installed. "
                "Install with: pip install google-search-results"
            )
            return []

        search = GoogleSearch({
            "q": query,
            "api_key": self.api_key,
            "engine": self.engine,
            "num": num,
        })
        data = search.get_dict()
        return data.get("organic_results", [])

    @staticmethod
    def _format_results(results: list[dict], query: str) -> str:
        """Format organic results into a concise text block for prompt injection."""
        lines = [f"Web search results for: \"{query}\"", ""]
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            source = r.get("displayed_link") or r.get("link", "")
            lines.append(f"{i}. **{title}**")
            if snippet:
                lines.append(f"   {snippet}")
            if source:
                lines.append(f"   Source: {source}")
            lines.append("")
        return "\n".join(lines).strip()

    @staticmethod
    def _build_query(skill_id: str, params: dict, extra_terms: str = "") -> str:
        """Build a search query tailored to the skill and its parameters."""
        indication = params.get("indication", "")
        age_group = params.get("age_group", "")
        phase = params.get("phase", "")

        queries = {
            "trial_benchmarking": (
                f"clinical trial benchmarking {indication} {phase} "
                f"{age_group} enrollment rate site count duration"
            ),
            "drug_reimbursement": (
                f"drug reimbursement HTA assessment {indication} {phase} "
                f"{params.get('countries', '')} approval payer"
            ),
            "enrollment_forecasting": (
                f"clinical trial enrollment rate {indication} {phase} "
                f"{age_group} recruitment timeline benchmark"
            ),
            "protocol_analysis": (
                f"clinical trial protocol design guidance {indication} "
                f"{phase} FDA EMA ICH"
            ),
        }

        query = queries.get(skill_id, "")
        if extra_terms:
            query = f"{query} {extra_terms}"
        return " ".join(query.split())  # normalize whitespace
