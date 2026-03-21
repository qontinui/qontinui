"""Semantic element matching for natural-language descriptions.

Matches needle descriptions (e.g. "Submit button", "Search field") against
detected element labels and captions using fuzzy string matching and
keyword decomposition.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass
class SemanticMatch:
    """A scored match between a description and a detected element."""

    element_index: int
    score: float
    matched_label: str
    match_type: str  # "exact", "fuzzy", "keyword"


# Common UI element type synonyms
_TYPE_SYNONYMS: dict[str, set[str]] = {
    "button": {"btn", "button", "submit", "click"},
    "text_field": {"input", "text_field", "textbox", "field", "entry", "search"},
    "dropdown": {"dropdown", "select", "combo", "combobox", "picker"},
    "checkbox": {"checkbox", "check", "toggle"},
    "radio": {"radio", "option"},
    "link": {"link", "anchor", "href", "url"},
    "icon": {"icon", "image", "img", "avatar"},
    "tab": {"tab"},
    "menu": {"menu", "menuitem", "nav"},
    "label": {"label", "text", "caption"},
}


def match_element_by_description(
    description: str,
    detected_labels: list[str],
    element_types: list[str | None] | None = None,
    min_similarity: float = 0.4,
) -> list[SemanticMatch]:
    """Match a natural language description against detected element labels.

    Uses a multi-strategy approach:
    1. Exact substring match (highest priority)
    2. Fuzzy string similarity (SequenceMatcher)
    3. Keyword overlap scoring

    Args:
        description: Natural language description, e.g. "blue Submit button".
        detected_labels: List of labels/captions from detected elements.
        element_types: Optional list of element types (parallel to detected_labels).
        min_similarity: Minimum combined score to include in results.

    Returns:
        List of SemanticMatch sorted by score (highest first).
    """
    if not description or not detected_labels:
        return []

    desc_lower = description.lower().strip()
    desc_keywords = _extract_keywords(desc_lower)

    matches: list[SemanticMatch] = []

    for i, label in enumerate(detected_labels):
        if not label:
            continue

        label_lower = label.lower().strip()
        elem_type = element_types[i] if element_types and i < len(element_types) else None

        score, match_type = _score_match(desc_lower, desc_keywords, label_lower, elem_type)

        if score >= min_similarity:
            matches.append(
                SemanticMatch(
                    element_index=i,
                    score=min(score, 1.0),
                    matched_label=label,
                    match_type=match_type,
                )
            )

    matches.sort(key=lambda m: m.score, reverse=True)
    return matches


def _score_match(
    desc: str,
    desc_keywords: set[str],
    label: str,
    element_type: str | None,
) -> tuple[float, str]:
    """Score how well a label matches a description.

    Tuned for Florence-2 captions which tend to be verbose, e.g.:
    - "a blue rectangular button with text 'Submit'"
    - "a magnifying glass search icon"
    - "a text input field with placeholder 'Search...'"

    Returns (score, match_type) tuple.
    """
    # 1. Exact substring match (either direction)
    if desc in label or label in desc:
        return (0.95, "exact")

    # 2. Fuzzy string similarity
    fuzzy_score = SequenceMatcher(None, desc, label).ratio()

    # 3. Keyword overlap (Jaccard)
    label_keywords = _extract_keywords(label)
    keyword_score = _keyword_overlap(desc_keywords, label_keywords)

    # 4. Directional keyword coverage: what fraction of the *description*
    #    keywords appear in the label? This handles the common case where
    #    the label is much longer than the description (Florence-2 verbosity).
    coverage_score = _keyword_coverage(desc_keywords, label_keywords)

    # 5. Type bonus: if description mentions a type synonym matching element_type
    type_bonus = 0.0
    if element_type:
        type_bonus = _type_match_bonus(desc_keywords, element_type)

    # Weighted combination — coverage weighted highest because Florence-2
    # captions are typically much longer than the user's description.
    combined = fuzzy_score * 0.25 + keyword_score * 0.25 + coverage_score * 0.35 + type_bonus * 0.15

    match_type = "fuzzy" if fuzzy_score > keyword_score else "keyword"
    return (combined, match_type)


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from text, filtering stopwords."""
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "and",
        "or",
        "not",
        "no",
        "by",
        "from",
        "it",
        "its",
    }
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in stopwords and len(w) > 1}


def _keyword_overlap(keywords_a: set[str], keywords_b: set[str]) -> float:
    """Jaccard-like keyword overlap score."""
    if not keywords_a or not keywords_b:
        return 0.0
    intersection = keywords_a & keywords_b
    union = keywords_a | keywords_b
    return len(intersection) / len(union) if union else 0.0


def _keyword_coverage(desc_keywords: set[str], label_keywords: set[str]) -> float:
    """What fraction of description keywords appear in the label?

    Asymmetric: optimised for short descriptions vs. long Florence-2 captions.
    "Submit button" (2 keywords) vs. "a blue rectangular button with text Submit" (5 keywords)
    → coverage = 2/2 = 1.0, whereas Jaccard = 2/5 = 0.4.
    """
    if not desc_keywords:
        return 0.0
    if not label_keywords:
        return 0.0
    covered = desc_keywords & label_keywords
    return len(covered) / len(desc_keywords)


def _type_match_bonus(desc_keywords: set[str], element_type: str) -> float:
    """Bonus score when description keywords match the element type synonyms."""
    for canonical, synonyms in _TYPE_SYNONYMS.items():
        if element_type.lower() in synonyms or canonical == element_type.lower():
            if desc_keywords & synonyms:
                return 1.0
    return 0.0
