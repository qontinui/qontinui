"""Type definitions for self-healing system.

Defines data structures for healing results, element locations, and
healing configuration.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LLMMode(Enum):
    """LLM access mode for self-healing.

    Controls whether and how LLM is used for element healing.
    Default is DISABLED for privacy and offline operation.
    """

    DISABLED = "disabled"
    """Default: No LLM access. Falls back to mechanical retry."""

    LOCAL = "local"
    """Local model via Ollama or llama.cpp. No internet required after download."""

    REMOTE = "remote"
    """Remote API (OpenAI, Anthropic, etc.). Requires API key and internet."""


class HealingStrategy(Enum):
    """Strategy used to heal a failed element lookup."""

    CACHE_HIT = "cache_hit"
    """Element found in cache."""

    VISUAL_SEARCH = "visual_search"
    """Found by searching visible screen area."""

    LLM_VISION = "llm_vision"
    """Found by vision LLM analyzing screenshot."""

    DOM_SELECTOR = "dom_selector"
    """Found by CSS/XPath selector (web only)."""

    TEXT_SEARCH = "text_search"
    """Found by searching for text content."""

    FAILED = "failed"
    """All healing strategies failed."""


@dataclass
class ElementLocation:
    """Location of an element found by healing."""

    x: int
    """X coordinate of element center."""

    y: int
    """Y coordinate of element center."""

    confidence: float
    """Confidence in the location (0.0-1.0)."""

    region: tuple[int, int, int, int] | None = None
    """Optional (x, y, width, height) bounding region."""

    description: str | None = None
    """Optional description of what was found."""


@dataclass
class HealingResult:
    """Result of a healing attempt."""

    success: bool
    """Whether healing succeeded."""

    strategy: HealingStrategy
    """Strategy that succeeded (or FAILED)."""

    location: ElementLocation | None = None
    """Location if successful."""

    message: str = ""
    """Human-readable result message."""

    attempts: list[tuple[HealingStrategy, str]] = field(default_factory=list)
    """List of (strategy, reason) for each attempt."""

    llm_tokens_used: int = 0
    """Number of LLM tokens used (for cost tracking)."""

    duration_ms: float = 0.0
    """Time spent healing in milliseconds."""


@dataclass
class HealingContext:
    """Context for a healing operation."""

    original_description: str
    """Description of the element being sought."""

    action_type: str | None = None
    """Type of action that was attempted (click, type, etc.)."""

    failure_reason: str | None = None
    """Why the original lookup failed."""

    state_id: str | None = None
    """Current state machine state ID."""

    screenshot_shape: tuple[int, int] | None = None
    """Shape of the screenshot (height, width)."""

    additional_context: dict[str, Any] = field(default_factory=dict)
    """Any additional context for the LLM."""
