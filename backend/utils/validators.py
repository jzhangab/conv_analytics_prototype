"""
Input validation helpers.
"""
from __future__ import annotations

import re
from datetime import datetime


def is_positive_integer(value) -> bool:
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def to_positive_integer(value) -> int | None:
    try:
        v = int(float(str(value)))
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def is_valid_date(value: str) -> bool:
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            datetime.strptime(value.strip(), fmt)
            return True
        except ValueError:
            continue
    return False


def normalize_date(value: str) -> str | None:
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(value.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def is_allowed_file(filename: str, allowed: list[str] = None) -> bool:
    if allowed is None:
        allowed = ["csv", "xlsx", "xls"]
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


def parse_country_list(raw: str) -> list[str]:
    """
    Parse a user-provided country string into a list.
    Handles comma, semicolon, and 'and' separators.
    """
    if not raw:
        return []
    # Replace common separators
    normalized = re.sub(r"\s+and\s+", ",", raw, flags=re.IGNORECASE)
    normalized = normalized.replace(";", ",")
    parts = [p.strip() for p in normalized.split(",")]
    return [p for p in parts if p]
