"""
Helpers for validating and extracting structured data from LLM JSON responses.
"""
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def require_keys(data: dict, keys: list[str], context: str = "") -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValueError(f"LLM response missing required keys {missing} in {context}: {data}")


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_intent_response(data: dict) -> tuple[str, float, str]:
    """Returns (intent, confidence, reasoning)."""
    require_keys(data, ["intent", "confidence", "reasoning"], context="intent_classification")
    return (
        str(data["intent"]),
        safe_float(data["confidence"]),
        str(data["reasoning"]),
    )


def parse_parameter_extraction(data: dict) -> dict:
    """Returns a dict of param_name -> value (None if not extracted)."""
    return {k: v for k, v in data.items()}


def parse_site_merger_response(data: dict) -> tuple[list, dict]:
    """Returns (merged_sites list, summary dict)."""
    require_keys(data, ["merged_sites", "summary"], context="site_merger")
    return data["merged_sites"], data["summary"]


def parse_benchmarking_response(data: dict) -> dict:
    require_keys(data, ["benchmark_summary", "key_metrics"], context="trial_benchmarking")
    return data


def parse_reimbursement_response(data: dict) -> dict:
    require_keys(data, ["overall_summary", "country_assessments"], context="drug_reimbursement")
    return data


def parse_enrollment_params(data: dict) -> dict:
    require_keys(data, ["moderate", "pessimistic", "optimistic"], context="enrollment_params")
    for scenario in ["moderate", "pessimistic", "optimistic"]:
        require_keys(data[scenario],
                     ["enrollment_rate_per_site_per_month", "site_ramp_period_months", "dropout_rate_monthly_percent"],
                     context=f"enrollment_params.{scenario}")
    return data
