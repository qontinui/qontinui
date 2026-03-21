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
    "radio": [AccessibilityRole.RADIO_BUTTON],
    "dropdown": [AccessibilityRole.COMBOBOX, AccessibilityRole.LISTBOX],
    "select": [AccessibilityRole.COMBOBOX, AccessibilityRole.LISTBOX],
    "menu": [AccessibilityRole.MENU, AccessibilityRole.MENUBAR],
    "menu item": [AccessibilityRole.MENU_ITEM],
    "tab": [AccessibilityRole.TAB],
    "slider": [AccessibilityRole.SLIDER],
    "tree": [AccessibilityRole.TREE, AccessibilityRole.TREE_ITEM],
    "list": [AccessibilityRole.LIST, AccessibilityRole.LIST_ITEM],
    "table": [AccessibilityRole.TABLE],
    "dialog": [AccessibilityRole.DIALOG],
    "toolbar": [AccessibilityRole.TOOLBAR],
    "scroll bar": [AccessibilityRole.SCROLLBAR],
    "image": [AccessibilityRole.IMAGE],
    "heading": [AccessibilityRole.HEADING],
}


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
    """Simple cache for semantic search results."""

    _cache: dict[tuple[str, str], list[SemanticMatch]] = field(default_factory=dict)

    def get(self, app_key: str, description: str) -> list[SemanticMatch] | None:
        return self._cache.get((app_key, description))

    def put(self, app_key: str, description: str, results: list[SemanticMatch]) -> None:
        self._cache[(app_key, description)] = results

    def invalidate(self, app_key: str | None = None) -> None:
        if app_key is None:
            self._cache.clear()
        else:
            self._cache = {k: v for k, v in self._cache.items() if k[0] != app_key}


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
) -> list[SemanticMatch]:
    """Find accessibility nodes matching a natural-language description.

    Uses a multi-signal scoring approach:
    1. Exact name match (score = 1.0)
    2. Fuzzy name match via SequenceMatcher
    3. Automation ID match
    4. Role hint bonus

    Args:
        description: Natural-language element description
            (e.g., "the search bar", "Submit button", "File menu")
        snapshot: Captured accessibility tree
        min_score: Minimum similarity score to include (0.0-1.0)
        max_results: Maximum number of results to return
        interactive_only: Only search interactive elements

    Returns:
        List of SemanticMatch sorted by score (highest first)
    """
    nodes = _flatten_nodes(snapshot.root, interactive_only)
    if not nodes:
        return []

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
        node_auto_id_lower = node_auto_id.lower()

        # 1. Exact name match
        if node_name_lower and node_name_lower == desc_lower:
            best_score = 1.0
            best_type = "exact_name"
            best_term = node_name

        # 2. Name contains all keywords
        if best_score < 1.0 and keywords and node_name_lower:
            matched_kw = sum(1 for kw in keywords if kw in node_name_lower)
            if matched_kw == len(keywords):
                score = 0.9
                if score > best_score:
                    best_score = score
                    best_type = "keyword_match"
                    best_term = node_name

        # 3. Fuzzy name match
        if best_score < 0.9 and node_name_lower:
            # Match against each keyword
            for kw in keywords:
                ratio = SequenceMatcher(None, kw, node_name_lower).ratio()
                if ratio > best_score:
                    best_score = ratio
                    best_type = "fuzzy_name"
                    best_term = node_name

            # Also match full description against name
            full_ratio = SequenceMatcher(None, desc_lower, node_name_lower).ratio()
            if full_ratio > best_score:
                best_score = full_ratio
                best_type = "fuzzy_name"
                best_term = node_name

        # 4. Automation ID match
        if node_auto_id_lower:
            for kw in keywords:
                ratio = SequenceMatcher(None, kw, node_auto_id_lower).ratio()
                if ratio > best_score:
                    best_score = ratio
                    best_type = "automation_id"
                    best_term = node_auto_id

        # 5. Role hint bonus — boost score if role matches hints
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
