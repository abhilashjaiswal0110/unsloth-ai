"""
Sector-Specific Reward Functions for GRPO Training
===================================================
Multi-objective reward functions for healthcare, public utility, and insurance.

Each reward function scores model completions on:
  1. Format      — structured output, numbered steps, clear sections
  2. Correctness — factual accuracy against reference answers
  3. Reasoning   — logical chain, step-by-step explanation
  4. Completeness — coverage of key points from reference
  5. Safety      — absence of harmful/misleading advice (sector-specific)
"""

from __future__ import annotations

import math
import re
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Common helpers
# ─────────────────────────────────────────────────────────────────────────────

_STEP_RE = re.compile(r"(?:\d+[\.\)]\s|step\s*\d+|•\s|[-–]\s)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _extract_thinking(text: str) -> str:
    m = _THINK_RE.search(text)
    return m.group(1).strip() if m else ""


def _extract_numbers(text: str) -> list[float]:
    """Extract all numbers from text."""
    nums = []
    for match in _NUMBER_RE.findall(text):
        try:
            nums.append(float(match.replace(",", "")))
        except ValueError:
            pass
    return nums


def _keyword_overlap(completion: str, reference: str, min_words: int = 3) -> float:
    """Jaccard-like overlap of meaningful words between completion and reference."""
    stop = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "of",
        "in",
        "to",
        "for",
        "with",
        "on",
        "at",
        "by",
        "from",
        "and",
        "or",
        "but",
        "not",
        "this",
        "that",
        "it",
        "as",
        "if",
        "than",
        "then",
        "so",
        "no",
        "yes",
    }
    comp_words = {w.lower().strip(".,;:!?()") for w in completion.split() if len(w) > 2}
    ref_words = {w.lower().strip(".,;:!?()") for w in reference.split() if len(w) > 2}
    comp_words -= stop
    ref_words -= stop
    if len(ref_words) < min_words:
        return 0.5
    intersection = comp_words & ref_words
    return len(intersection) / max(len(ref_words), 1)


def _numeric_accuracy(completion: str, reference: str) -> float:
    """Check if key numbers in reference also appear in completion."""
    ref_nums = _extract_numbers(reference)
    if not ref_nums:
        return 0.5  # no numbers to check
    comp_nums = _extract_numbers(completion)
    matches = 0
    for rn in ref_nums:
        for cn in comp_nums:
            if math.isclose(rn, cn, rel_tol=0.02):
                matches += 1
                break
    return matches / len(ref_nums)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Format Reward (shared)
# ─────────────────────────────────────────────────────────────────────────────


def format_reward(completions: list[str], **kwargs) -> list[float]:
    """Rewards structured formatting: numbered steps, sections, headings."""
    scores = []
    for comp in completions:
        score = 0.0
        # Has numbered steps or bullet points
        steps = len(_STEP_RE.findall(comp))
        score += min(steps * 0.08, 0.4)

        # Has thinking block
        thinking = _extract_thinking(comp)
        if thinking and len(thinking.split()) >= 10:
            score += 0.2

        # Reasonable length (not too short, not excessively long)
        words = len(comp.split())
        if 50 <= words <= 800:
            score += 0.2
        elif 30 <= words < 50 or 800 < words <= 1200:
            score += 0.1

        # Has colons or labeled sections (indicates structure)
        if comp.count(":") >= 2:
            score += 0.1

        # Closing/summary sentence
        if any(
            w in comp.lower()
            for w in ["in summary", "therefore", "in conclusion", "recommendation"]
        ):
            score += 0.1

        scores.append(round(min(score, 1.0), 4))
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# 2. Correctness Reward (sector-aware)
# ─────────────────────────────────────────────────────────────────────────────


def correctness_reward(
    completions: list[str], answers: list[str] | None = None, **kwargs
) -> list[float]:
    """
    Rewards factual correctness by checking:
      - Keyword overlap with reference answer
      - Numerical accuracy for calculation questions
    """
    if not answers:
        return [0.0] * len(completions)

    scores = []
    for comp, ref in zip(completions, answers):
        kw_score = _keyword_overlap(comp, ref)
        num_score = _numeric_accuracy(comp, ref)
        # Weight: 60% keyword, 40% numeric
        score = 0.6 * kw_score + 0.4 * num_score
        scores.append(round(min(score, 1.0), 4))
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# 3. Reasoning Reward
# ─────────────────────────────────────────────────────────────────────────────


def reasoning_reward(completions: list[str], **kwargs) -> list[float]:
    """Rewards step-by-step reasoning, causal connectives, calculations."""
    scores = []
    for comp in completions:
        score = 0.0
        text = _extract_thinking(comp) or comp

        # Explicit steps
        steps = len(_STEP_RE.findall(text))
        score += min(steps * 0.06, 0.3)

        # Causal/logical connectives
        connectives = sum(
            1
            for w in [
                "because",
                "therefore",
                "since",
                "thus",
                "hence",
                "as a result",
                "consequently",
                "this means",
                "due to",
            ]
            if w in text.lower()
        )
        score += min(connectives * 0.06, 0.3)

        # Mathematical operations (=, ×, +, -, /)
        math_ops = sum(text.count(c) for c in ["=", "×", "+", "/", "÷"])
        score += min(math_ops * 0.02, 0.2)

        # Comparisons or conditional reasoning
        conditionals = sum(
            1
            for w in [
                "if ",
                "however",
                "although",
                "whereas",
                "compared to",
                "in contrast",
                "on the other hand",
            ]
            if w in text.lower()
        )
        score += min(conditionals * 0.05, 0.2)

        scores.append(round(min(score, 1.0), 4))
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# 4. Completeness Reward
# ─────────────────────────────────────────────────────────────────────────────


def completeness_reward(
    completions: list[str], answers: list[str] | None = None, **kwargs
) -> list[float]:
    """Rewards coverage of key information from reference answer."""
    if not answers:
        return [0.5] * len(completions)

    scores = []
    for comp, ref in zip(completions, answers):
        overlap = _keyword_overlap(comp, ref)
        # Boost if completion covers most reference content
        if overlap >= 0.7:
            score = 1.0
        elif overlap >= 0.5:
            score = 0.7
        elif overlap >= 0.3:
            score = 0.4
        else:
            score = overlap
        scores.append(round(score, 4))
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# 5. Safety Reward (sector-specific)
# ─────────────────────────────────────────────────────────────────────────────

# Healthcare unsafe patterns
_HEALTHCARE_UNSAFE = [
    r"take\s+\d+\s+pills",
    r"stop\s+(?:taking|your)\s+medication",
    r"no\s+need\s+(?:to|for)\s+(?:see|visit|consult)\s+(?:a\s+)?doctor",
    r"self[\s-]?medicate",
    r"don'?t\s+(?:go|need)\s+(?:to\s+)?(?:the\s+)?(?:hospital|emergency|ER|doctor)",
    r"guaranteed\s+(?:cure|treatment)",
    r"this\s+will\s+(?:definitely|certainly)\s+cure",
]

# Insurance unsafe patterns
_INSURANCE_UNSAFE = [
    r"hide\s+(?:this|the|your)\s+(?:information|damage|claim|condition)",
    r"don'?t\s+(?:report|disclose|tell|mention)",
    r"commit\s+(?:fraud|insurance\s+fraud)",
    r"exaggerate\s+(?:the|your)\s+(?:claim|damage|injury|loss)",
    r"file\s+a\s+fake\s+claim",
    r"misrepresent",
]

# Public utility unsafe patterns
_PUBLIC_UTILITY_UNSAFE = [
    r"bypass\s+(?:the|your)\s+meter",
    r"tamper\s+with",
    r"steal\s+(?:electricity|power|gas|water)",
    r"illegal\s+(?:connection|hook[\s-]?up)",
    r"don'?t\s+pay\s+(?:the|your)\s+(?:bill|utility)",
]


def _check_unsafe(text: str, patterns: list[str]) -> int:
    """Count how many unsafe patterns match in the text."""
    count = 0
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            count += 1
    return count


def safety_reward(
    completions: list[str], sector: str = "healthcare", **kwargs
) -> list[float]:
    """Penalizes unsafe content specific to the sector domain."""
    if sector == "healthcare":
        patterns = _HEALTHCARE_UNSAFE
    elif sector == "insurance":
        patterns = _INSURANCE_UNSAFE
    elif sector == "public_utility":
        patterns = _PUBLIC_UTILITY_UNSAFE
    else:
        patterns = _HEALTHCARE_UNSAFE + _INSURANCE_UNSAFE + _PUBLIC_UTILITY_UNSAFE

    scores = []
    for comp in completions:
        violations = _check_unsafe(comp, patterns)
        if violations == 0:
            score = 1.0
        elif violations == 1:
            score = 0.3
        else:
            score = 0.0

        # Bonus for including safety caveats / disclaimers
        caveats = sum(
            1
            for w in [
                "consult",
                "professional advice",
                "seek medical",
                "contact your",
                "seek legal",
                "disclaimer",
                "not a substitute",
                "qualified",
            ]
            if w in comp.lower()
        )
        if caveats > 0:
            score = min(score + 0.1, 1.0)

        scores.append(round(score, 4))
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Composite Multi-Objective Reward Factory
# ─────────────────────────────────────────────────────────────────────────────


def make_sector_reward(
    sector: str,
    weight_format: float = 0.15,
    weight_correctness: float = 0.35,
    weight_reasoning: float = 0.25,
    weight_completeness: float = 0.15,
    weight_safety: float = 0.10,
):
    """
    Factory: returns a single composite reward function configured for a sector.

    Default weights optimized for domain QA:
      - Correctness dominates (35%) — right answers matter most
      - Reasoning (25%)             — must show work / explain
      - Format (15%)                — structured output
      - Completeness (15%)          — cover all key points
      - Safety (10%)                — sector-appropriate caution
    """

    def composite_reward(completions: list[str], **kwargs) -> list[float]:
        answers = kwargs.pop("answers", None)
        fmt = format_reward(completions, **kwargs)
        cor = correctness_reward(completions, answers=answers, **kwargs)
        rea = reasoning_reward(completions, **kwargs)
        com = completeness_reward(completions, answers=answers, **kwargs)
        saf = safety_reward(completions, sector=sector, **kwargs)

        return [
            round(
                weight_format * f
                + weight_correctness * c
                + weight_reasoning * r
                + weight_completeness * cp
                + weight_safety * s,
                4,
            )
            for f, c, r, cp, s in zip(fmt, cor, rea, com, saf)
        ]

    return composite_reward
