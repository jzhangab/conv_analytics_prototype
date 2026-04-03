"""
LLM-based intent classifier. Returns a skill_id and confidence score.
"""
from __future__ import annotations

import logging

from backend.llm.llm_client import LLMClient
from backend.llm.prompt_templates import (INTENT_CLASSIFIER_SYSTEM,
                                           INTENT_CLASSIFIER_USER)
from backend.llm.response_parser import parse_intent_response

logger = logging.getLogger(__name__)

VALID_INTENTS = {
    "site_list_merger",
    "trial_benchmarking",
    "drug_reimbursement",
    "enrollment_forecasting",
}

CONFIDENCE_THRESHOLD = 0.70


def classify_intent(
    llm: LLMClient,
    user_message: str,
    history: list[dict],
) -> tuple[str | None, float, str]:
    """
    Returns (intent_or_None, confidence, reasoning).
    Returns None if intent is "unknown" or confidence < threshold.
    """
    history_text = _format_history(history)

    messages = [
        {"role": "system", "content": INTENT_CLASSIFIER_SYSTEM},
        {"role": "user", "content": INTENT_CLASSIFIER_USER.format(
            history=history_text,
            user_message=user_message,
        )},
    ]

    try:
        data = llm.complete_json(messages, temperature=llm.temp_classify)
        intent, confidence, reasoning = parse_intent_response(data)
    except Exception as e:
        logger.error("Intent classification failed: %s", e)
        return None, 0.0, "Classification error"

    if intent not in VALID_INTENTS or confidence < CONFIDENCE_THRESHOLD:
        return None, confidence, reasoning

    return intent, confidence, reasoning


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(no prior conversation)"
    lines = []
    for msg in history:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
