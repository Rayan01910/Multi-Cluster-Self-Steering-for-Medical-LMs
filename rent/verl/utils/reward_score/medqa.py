"""Utilities for computing MedQA rewards."""

from __future__ import annotations

import re
from typing import Optional

_VALID_CHOICES = {"A", "B", "C", "D"}

_ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*([A-D])\s*</answer>", re.IGNORECASE)
_LETTER_PATTERN = re.compile(r"\b([A-D])\b")


def extract_choice(solution: str) -> Optional[str]:
    match = _ANSWER_TAG_PATTERN.search(solution)
    if match:
        return match.group(1).upper()

    fallback = _LETTER_PATTERN.findall(solution.upper())
    for letter in fallback[::-1]:
        if letter in _VALID_CHOICES:
            return letter
    return None


def compute_score(solution_str: str, ground_truth: str) -> float:
    expected = str(ground_truth).strip().upper()
    pred = extract_choice(solution_str)
    return 1.0 if pred == expected else 0.0
