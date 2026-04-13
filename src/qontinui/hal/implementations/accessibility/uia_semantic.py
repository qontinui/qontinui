"""Semantic search utilities for UIA accessibility trees.

Provides fuzzy matching against AccessibilityNode names, automation IDs,
and roles using stdlib difflib — no external dependencies required.

Inspired by pywinassistant's semantic keyword matching approach:
generate search terms from a natural-language description, then
fuzzy-match them against the UIA tree.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from qontinui_schemas.accessibility import (
    AccessibilityNode,
    AccessibilityRole,
    AccessibilitySnapshot,
)

from .uia_label_utils import infer_spatial_labels as _infer_spatial_labels

logger = logging.getLogger(__name__)

# Map common natural-language hints to accessibility roles
_ROLE_HINTS: dict[str, list[AccessibilityRole]] = {
    "button": [AccessibilityRole.BUTTON],
    "btn": [AccessibilityRole.BUTTON],
    "link": [AccessibilityRole.LINK],
    "input": [AccessibilityRole.TEXTBOX, AccessibilityRole.COMBOBOX],
    "text field": [AccessibilityRole.TEXTBOX],
    "textbox": [AccessibilityRole.TEXTBOX],
    "search bar": [AccessibilityRole.TEXTBOX, AccessibilityRole.SEARCH],
    "search": [AccessibilityRole.SEARCH, AccessibilityRole.TEXTBOX],
    "checkbox": [AccessibilityRole.CHECKBOX],
    "radio": [AccessibilityRole.RADIO],
    "dropdown": [AccessibilityRole.COMBOBOX, AccessibilityRole.LISTBOX],
    "select": [AccessibilityRole.COMBOBOX, AccessibilityRole.LISTBOX],
    "menu": [AccessibilityRole.MENU, AccessibilityRole.MENUBAR],
    "menu item": [AccessibilityRole.MENUITEM],
    "tab": [AccessibilityRole.TAB],
    "slider": [AccessibilityRole.SLIDER],
    "tree": [AccessibilityRole.TREE, AccessibilityRole.TREEITEM],
    "list": [AccessibilityRole.LIST, AccessibilityRole.LISTITEM],
    "table": [AccessibilityRole.TABLE],
    "dialog": [AccessibilityRole.DIALOG],
    "toolbar": [AccessibilityRole.TOOLBAR],
    "scroll bar": [AccessibilityRole.SCROLLBAR],
    "image": [AccessibilityRole.IMG],
    "heading": [AccessibilityRole.HEADING],
}

# Strip non-alphanumeric characters for mode 3/4 matching
_RE_NON_ALNUM = re.compile(r"[^a-z0-9]")


@dataclass
class SemanticMatch:
    """Result of a semantic search against the accessibility tree."""

    node: AccessibilityNode
    score: float  # 0.0 - 1.0
    match_type: str  # "exact_name", "fuzzy_name", "automation_id", "role_hint"
    matched_term: str  # The term that matched

    @property
    def ref(self) -> str:
        return self.node.ref


@dataclass
class SemanticSearchCache:
    """Simple cache for semantic search results and raw SequenceMatcher ratios."""

    _cache: dict[tuple[str, str], list[SemanticMatch]] = field(default_factory=dict)
    _ratio_cache: dict[tuple[str, str], float] = field(default_factory=dict)

    def get(self, app_key: str, description: str) -> list[SemanticMatch] | None:
        return self._cache.get((app_key, description))

    def put(self, app_key: str, description: str, results: list[SemanticMatch]) -> None:
        self._cache[(app_key, description)] = results

    def get_ratio(self, a: str, b: str) -> float | None:
        return self._ratio_cache.get((a, b))

    def put_ratio(self, a: str, b: str, ratio: float) -> None:
        self._ratio_cache[(a, b)] = ratio
        self._ratio_cache[(b, a)] = ratio  # bidirectional

    def invalidate(self, app_key: str | None = None) -> None:
        if app_key is None:
            self._cache.clear()
            self._ratio_cache.clear()
        else:
            self._cache = {k: v for k, v in self._cache.items() if k[0] != app_key}
            self._ratio_cache.clear()


def _sieve_ratio(a: str, b: str, cutoff: float) -> float:
    """Three-pass sieve ratio for a single (a, b) string pair.

    Uses SequenceMatcher's real_quick_ratio() → quick_ratio() → ratio()
    sequence to avoid the expensive full ratio() when the upper bound
    is already below *cutoff*.  Returns 0.0 when the pair is pruned.
    """
    sm = SequenceMatcher(None, a, b)
    if sm.real_quick_ratio() < cutoff:
        return 0.0
    if sm.quick_ratio() < cutoff:
        return 0.0
    return sm.ratio()


def _multi_mode_ratio(
    a: str,
    b: str,
    cutoff: float,
    cache: SemanticSearchCache | None = None,
) -> float:
    """Compute the best similarity between *a* and *b* across four matching modes.

    The four modes mirror pywinauto's string-comparison strategies:

    Mode 1 — raw strings                      weight 1.00
    Mode 2 — case-insensitive                 weight 0.90
    Mode 3 — non-alphanumeric stripped        weight 0.90
    Mode 4 — case-insensitive AND stripped    weight 0.81

    Each mode uses the three-pass sieve (real_quick → quick → full ratio)
    so expensive full-ratio passes are skipped whenever the cheap upper
    bounds already rule out improvement over the current best.

    The cache (if provided) is consulted and populated for every pair
    actually computed so repeated calls with the same strings are free.
    """
    best = 0.0

    def _cached_sieve(s1: str, s2: str, weight: float) -> float:
        """Return weight * sieve_ratio, using cache when available."""
        if cache is not None:
            cached = cache.get_ratio(s1, s2)
            if cached is not None:
                return weight * cached
        r = _sieve_ratio(s1, s2, cutoff / weight if weight > 0 else cutoff)
        if cache is not None:
            cache.put_ratio(s1, s2, r)
        return weight * r

    # Mode 1: raw strings (weight 1.0)
    r1 = _cached_sieve(a, b, 1.0)
    if r1 > best:
        best = r1

    # Mode 2: case-insensitive (weight 0.9)
    a_lower = a.lower()
    b_lower = b.lower()
    if a_lower != a or b_lower != b:
        r2 = _cached_sieve(a_lower, b_lower, 0.9)
        if r2 > best:
            best = r2
    else:
        # a and b are already lower-case; mode 2 is identical to mode 1
        # (already captured above, no need to recompute)
        pass

    # Mode 3: non-alphanumeric stripped, case-sensitive (weight 0.9)
    a_stripped = _RE_NON_ALNUM.sub("", a)
    b_stripped = _RE_NON_ALNUM.sub("", b)
    if a_stripped != a or b_stripped != b:
        r3 = _cached_sieve(a_stripped, b_stripped, 0.9)
        if r3 > best:
            best = r3

    # Mode 4: case-insensitive AND stripped (weight 0.81)
    a_stripped_lower = _RE_NON_ALNUM.sub("", a_lower)
    b_stripped_lower = _RE_NON_ALNUM.sub("", b_lower)
    if a_stripped_lower != a_lower or b_stripped_lower != b_lower:
        r4 = _cached_sieve(a_stripped_lower, b_stripped_lower, 0.81)
        if r4 > best:
            best = r4

    return best


def _node_identifiers(
    node: AccessibilityNode,
    spatial_labels: dict[str, str],
) -> list[tuple[str, float]]:
    """Generate (identifier_text, weight) pairs for a node.

    Up to five identifiers are produced per node, ordered by confidence:

    1. raw name                         weight 1.00
    2. name + role_name                 weight 0.95
    3. automation_id                    weight 0.90
    4. inferred spatial label           weight 0.85
    5. inferred label + role_name       weight 0.80

    Only non-empty strings are included.
    """
    identifiers: list[tuple[str, float]] = []

    role_name = node.role.value if node.role else ""
    raw_name = (node.name or "").strip()
    auto_id = (node.automation_id or "").strip()
    inferred = spatial_labels.get(node.ref, "").strip()

    if raw_name:
        identifiers.append((raw_name, 1.0))
        if role_name:
            identifiers.append((f"{raw_name} {role_name}", 0.95))

    if auto_id:
        identifiers.append((auto_id, 0.9))

    if inferred:
        identifiers.append((inferred, 0.85))
        if role_name:
            identifiers.append((f"{inferred} {role_name}", 0.8))

    return identifiers


def _flatten_nodes(
    node: AccessibilityNode,
    interactive_only: bool = True,
) -> list[AccessibilityNode]:
    """Flatten the accessibility tree into a list of nodes."""
    result: list[AccessibilityNode] = []

    if not interactive_only or node.is_interactive:
        result.append(node)

    for child in node.children:
        result.extend(_flatten_nodes(child, interactive_only))

    return result


def _extract_role_hints(description: str) -> list[AccessibilityRole]:
    """Extract likely roles from a natural-language description."""
    desc_lower = description.lower()
    roles: list[AccessibilityRole] = []

    for hint, hint_roles in _ROLE_HINTS.items():
        if hint in desc_lower:
            for r in hint_roles:
                if r not in roles:
                    roles.append(r)

    return roles


def _extract_name_keywords(description: str) -> list[str]:
    """Extract name-matching keywords from a description.

    Strips role hints and common filler words to get the semantic target.
    E.g., "the blue Submit button" → ["blue", "submit"]
    """
    desc_lower = description.lower()

    # Remove role hint phrases
    for hint in _ROLE_HINTS:
        desc_lower = desc_lower.replace(hint, " ")

    # Remove common filler words
    fillers = {
        "the",
        "a",
        "an",
        "this",
        "that",
        "click",
        "press",
        "tap",
        "on",
        "in",
        "at",
        "to",
        "of",
        "for",
        "with",
        "and",
        "or",
        "find",
        "locate",
        "select",
        "open",
        "close",
        "first",
        "last",
        "main",
        "primary",
        "secondary",
    }

    words = re.findall(r"[a-z0-9]+", desc_lower)
    keywords = [w for w in words if w not in fillers and len(w) > 1]

    return keywords


def fuzzy_match_nodes(
    description: str,
    snapshot: AccessibilitySnapshot,
    *,
    min_score: float = 0.4,
    max_results: int = 5,
    interactive_only: bool = True,
    cache: SemanticSearchCache | None = None,
) -> list[SemanticMatch]:
    """Find accessibility nodes matching a natural-language description.

    Uses a multi-signal scoring approach:
    1. Exact name match (score = 1.0)
    2. Fuzzy name match via SequenceMatcher (progressive ratio sieve)
    3. Automation ID match
    4. Role hint bonus

    Args:
        description: Natural-language element description
            (e.g., "the search bar", "Submit button", "File menu")
        snapshot: Captured accessibility tree
        min_score: Minimum similarity score to include (0.0-1.0)
        max_results: Maximum number of results to return
        interactive_only: Only search interactive elements
        cache: Optional SemanticSearchCache for ratio caching.

    Returns:
        List of SemanticMatch sorted by score (highest first)
    """
    nodes = _flatten_nodes(snapshot.root, interactive_only)
    if not nodes:
        return []

    # Preprocessing: infer labels for unlabeled controls from adjacent text nodes.
    # We run over all nodes (not just interactive) so text_nodes are included.
    all_nodes = _flatten_nodes(snapshot.root, interactive_only=False)
    spatial_labels = _infer_spatial_labels(all_nodes)

    role_hints = _extract_role_hints(description)
    keywords = _extract_name_keywords(description)
    desc_lower = description.lower()

    matches: list[SemanticMatch] = []

    for node in nodes:
        best_score = 0.0
        best_type = ""
        best_term = ""

        node_name = (node.name or "").strip()
        node_name_lower = node_name.lower()
        node_auto_id = (node.automation_id or "").strip()

        # 1. Exact name match (against node.name directly)
        if node_name_lower and node_name_lower == desc_lower:
            best_score = 1.0
            best_type = "exact_name"
            best_term = node_name

        # 2. Name contains all keywords (against node.name directly)
        if best_score < 1.0 and keywords and node_name_lower:
            matched_kw = sum(1 for kw in keywords if kw in node_name_lower)
            if matched_kw == len(keywords):
                score = 0.9
                if score > best_score:
                    best_score = score
                    best_type = "keyword_match"
                    best_term = node_name

        # 3. Fuzzy multi-identifier match — progressive ratio sieve with four modes.
        # Iterate over all identifiers (name, name+role, automation_id, inferred
        # label, inferred label+role) and take the best weighted ratio.
        if best_score < 0.9:
            identifiers = _node_identifiers(node, spatial_labels)
            for ident_text, weight in identifiers:
                ident_lower = ident_text.lower()
                # Match against each keyword
                for kw in keywords:
                    ratio = _multi_mode_ratio(kw, ident_lower, min_score, cache)
                    weighted = ratio * weight
                    if weighted > best_score:
                        best_score = weighted
                        # Distinguish match type by source identifier
                        if ident_text == node_name:
                            best_type = "fuzzy_name"
                            best_term = node_name
                        elif ident_text == node_auto_id:
                            best_type = "automation_id"
                            best_term = node_auto_id
                        elif node_auto_id and ident_text == node_auto_id:
                            best_type = "automation_id"
                            best_term = node_auto_id
                        elif node.ref in spatial_labels and spatial_labels[node.ref] in ident_text:
                            best_type = "spatial_label"
                            best_term = ident_text
                        else:
                            best_type = "fuzzy_name"
                            best_term = ident_text

                # Also match full description against identifier
                full_ratio = _multi_mode_ratio(desc_lower, ident_lower, min_score, cache)
                full_weighted = full_ratio * weight
                if full_weighted > best_score:
                    best_score = full_weighted
                    if ident_text == node_name:
                        best_type = "fuzzy_name"
                        best_term = node_name
                    elif ident_text == node_auto_id:
                        best_type = "automation_id"
                        best_term = node_auto_id
                    elif node.ref in spatial_labels and spatial_labels[node.ref] in ident_text:
                        best_type = "spatial_label"
                        best_term = ident_text
                    else:
                        best_type = "fuzzy_name"
                        best_term = ident_text

        # 4. Role hint bonus — boost score if role matches hints
        if role_hints and node.role in role_hints:
            best_score = min(best_score + 0.15, 1.0)

        if best_score >= min_score:
            matches.append(
                SemanticMatch(
                    node=node,
                    score=best_score,
                    match_type=best_type,
                    matched_term=best_term,
                )
            )

    # Sort by score descending, take top results
    matches.sort(key=lambda m: m.score, reverse=True)
    return matches[:max_results]


def format_nodes_for_llm(
    snapshot: AccessibilitySnapshot,
    *,
    max_elements: int = 80,
    interactive_only: bool = True,
) -> str:
    """Format accessibility nodes as an indexed list for LLM consumption.

    Similar to LLMFormatter for web elements but for accessibility trees.

    Returns:
        Indexed element list, e.g.:
            [0] @e1: button "Save"
            [1] @e3: textbox "Search" = ""
            [2] @e5: link "Help"
    """
    nodes = _flatten_nodes(snapshot.root, interactive_only)
    lines: list[str] = []

    for i, node in enumerate(nodes[:max_elements]):
        name_str = f' "{node.name}"' if node.name else ""
        value_str = f' = "{node.value}"' if node.value else ""
        auto_id = f" [id={node.automation_id}]" if node.automation_id else ""
        lines.append(f"[{i}] {node.ref}: {node.role.value}{name_str}{value_str}{auto_id}")

    if len(nodes) > max_elements:
        lines.append(f"... ({len(nodes)} total, showing first {max_elements})")

    return "\n".join(lines)
