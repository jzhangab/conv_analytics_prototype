"""
String similarity utilities for site list matching.
Implements Jaro-Winkler similarity without external dependencies.
"""
from __future__ import annotations


def jaro_similarity(s1: str, s2: str) -> float:
    """Compute the Jaro similarity between two strings."""
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
    """Compute the Jaro-Winkler similarity between two strings."""
    jaro = jaro_similarity(s1, s2)

    # Common prefix length (up to 4 characters)
    prefix_len = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break

    return jaro + prefix_len * prefix_weight * (1 - jaro)


def normalize_for_matching(text: str) -> str:
    """Lowercase, strip, collapse whitespace for matching."""
    if not text:
        return ""
    return " ".join(str(text).lower().split())


def first_n_words(text: str, n: int = 3) -> str:
    """Return the first n words of a string."""
    if not text:
        return ""
    return " ".join(str(text).split()[:n])
