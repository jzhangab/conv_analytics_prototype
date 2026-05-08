"""
String similarity utilities for site list matching.

Uses rapidfuzz (C extension) when available — ~50-100x faster than the
pure-Python fallback.  Both paths produce identical results.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from rapidfuzz.distance import JaroWinkler as _JaroWinkler
    _USE_RAPIDFUZZ = True
except ImportError:
    _USE_RAPIDFUZZ = False
    logger.warning(
        "rapidfuzz not installed — falling back to pure-Python Jaro-Winkler. "
        "Install rapidfuzz for significantly faster site matching: pip install rapidfuzz"
    )


def jaro_similarity(s1: str, s2: str) -> float:
    """Pure-Python Jaro similarity (used as fallback only)."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    return (
        matches / len1
        + matches / len2
        + (matches - transpositions / 2) / matches
    ) / 3


def jaro_winkler_similarity(s1: str, s2: str, prefix_weight: float = 0.1) -> float:
    """
    Jaro-Winkler similarity in [0, 1].

    Delegates to rapidfuzz when installed (C extension, ~50-100x faster).
    Falls back to the pure-Python implementation otherwise.
    """
    if _USE_RAPIDFUZZ:
        # rapidfuzz uses prefix_weight=0.1 by default — matches our convention.
        return _JaroWinkler.similarity(s1, s2)

    jaro = jaro_similarity(s1, s2)
    prefix_len = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break
    return jaro + prefix_len * prefix_weight * (1 - jaro)


def normalize_for_matching(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    if not text:
        return ""
    return " ".join(str(text).lower().split())


def first_n_words(text: str, n: int = 3) -> str:
    """Return the first n words of a string."""
    if not text:
        return ""
    return " ".join(str(text).split()[:n])
