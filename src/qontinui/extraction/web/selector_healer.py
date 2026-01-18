"""
Selector self-healing module.

This module automatically repairs broken CSS selectors when the DOM changes,
enabling robust automation that survives UI updates. Similar to Stagehand's
selfHeal capability.

Healing Strategies (in priority order)
--------------------------------------
1. **History Lookup**: Check if this selector was healed before
2. **Selector Variations**: Try similar selectors (remove nth-child, classes)
3. **Text Content Matching**: Find by visible text
4. **Aria-Label Matching**: Find by accessibility label
5. **Position Matching**: Find element near original position
6. **LLM Recovery**: Use AI to find semantically similar element

Learning Features
-----------------
- **Healing History**: Persistent storage of successful repairs
- **Strategy Statistics**: Track which strategies work best
- **Pattern Mapping**: Generalize selector patterns for reuse

Classes
-------
SelectorHealer
    Main healer with all strategies and learning.
HealingHistory
    Persistent storage for healing records.
HealingResult
    Result container with healed selector and metadata.
HealingRecord
    Record of a successful healing for learning.

Functions
---------
heal_broken_selector
    Convenience function for one-off healing.

Usage Examples
--------------
Basic selector healing::

    from qontinui.extraction.web import SelectorHealer

    healer = SelectorHealer()
    result = await healer.heal_selector(
        broken_selector="#old-button",
        original_element=saved_element,
        page=page,
    )
    if result.success:
        print(f"Healed: {result.healed_selector}")
        print(f"Strategy: {result.strategy_used}")

With LLM fallback::

    from qontinui.extraction.web import SelectorHealer, AnthropicClient

    healer = SelectorHealer(llm_client=AnthropicClient())
    result = await healer.heal_selector(selector, element, page)

With persistent history::

    healer = SelectorHealer(
        history_storage_path="./healing_history.json"
    )
    # History is loaded from file and saved after each healing

Convenience function::

    from qontinui.extraction.web import heal_broken_selector

    result = await heal_broken_selector(
        "#broken-selector",
        original_element,
        page,
        llm_client=AnthropicClient(),
    )

See Also
--------
- natural_language_selector: AI-powered element selection
- llm_clients: LLM client adapters
- interactive_element_extractor: Source of element data
"""

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from playwright.async_api import ElementHandle, Page

from .models import BoundingBox, InteractiveElement
from .natural_language_selector import LLMClient, NaturalLanguageSelector

logger = logging.getLogger(__name__)


@dataclass
class HealingAttempt:
    """Record of a healing attempt."""

    strategy: str
    selector_tried: str
    success: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "selector_tried": self.selector_tried,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class HealingResult:
    """Result of selector healing."""

    success: bool
    original_selector: str
    healed_selector: str | None
    element: ElementHandle | None
    confidence: float
    strategy_used: str
    attempts: list[HealingAttempt] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "original_selector": self.original_selector,
            "healed_selector": self.healed_selector,
            "confidence": self.confidence,
            "strategy_used": self.strategy_used,
            "attempts": [a.to_dict() for a in self.attempts],
        }


@dataclass
class HealingRecord:
    """
    A record of a successful healing for learning.

    Used to remember what worked so we can try it first next time.
    """

    original_selector: str
    healed_selector: str
    strategy_used: str
    confidence: float
    timestamp: float
    url_pattern: str  # Domain or URL pattern where this worked
    element_signature: str  # Hash of element characteristics
    success_count: int = 1  # How many times this mapping has worked

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_selector": self.original_selector,
            "healed_selector": self.healed_selector,
            "strategy_used": self.strategy_used,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "url_pattern": self.url_pattern,
            "element_signature": self.element_signature,
            "success_count": self.success_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HealingRecord":
        return cls(
            original_selector=data["original_selector"],
            healed_selector=data["healed_selector"],
            strategy_used=data["strategy_used"],
            confidence=data["confidence"],
            timestamp=data["timestamp"],
            url_pattern=data["url_pattern"],
            element_signature=data["element_signature"],
            success_count=data.get("success_count", 1),
        )


class HealingHistory:
    """
    Persistent storage for healing records.

    Learns from past selector repairs to improve future healing:
    - Remembers successful selector transformations
    - Tracks which strategies work best for different selectors
    - Provides lookup by selector pattern and URL
    """

    def __init__(self, storage_path: Path | str | None = None):
        """
        Initialize healing history.

        Args:
            storage_path: Path to JSON file for persistence.
                         If None, history is in-memory only.
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.records: list[HealingRecord] = []
        self._selector_index: dict[str, list[int]] = {}  # selector -> record indices
        self._pattern_index: dict[str, list[int]] = {}  # pattern -> record indices

        if self.storage_path and self.storage_path.exists():
            self._load()

    def add_record(self, result: HealingResult, url: str, element: InteractiveElement) -> None:
        """
        Add a successful healing to history.

        Args:
            result: The successful healing result
            url: URL where the healing occurred
            element: The original element data
        """
        if not result.success or not result.healed_selector:
            return

        # Generate element signature for matching similar elements
        signature = self._compute_element_signature(element)
        url_pattern = self._extract_url_pattern(url)

        # Check if we already have this mapping
        existing = self._find_existing_record(
            result.original_selector,
            result.healed_selector,
            url_pattern,
        )

        if existing:
            # Increment success count
            existing.success_count += 1
            existing.timestamp = time.time()
        else:
            # Create new record
            record = HealingRecord(
                original_selector=result.original_selector,
                healed_selector=result.healed_selector,
                strategy_used=result.strategy_used,
                confidence=result.confidence,
                timestamp=time.time(),
                url_pattern=url_pattern,
                element_signature=signature,
            )
            self._add_record_to_index(record)

        self._save()

    def lookup(
        self,
        selector: str,
        url: str | None = None,
        element: InteractiveElement | None = None,
    ) -> list[HealingRecord]:
        """
        Look up past healing records for a selector.

        Args:
            selector: The broken selector
            url: Optional URL to filter by domain
            element: Optional element to match by signature

        Returns:
            List of relevant healing records, sorted by relevance
        """
        candidates: list[tuple[HealingRecord, float]] = []
        seen_indices: set[int] = set()

        # Look up by exact selector match
        if selector in self._selector_index:
            for idx in self._selector_index[selector]:
                record = self.records[idx]
                score = 1.0 * record.success_count
                candidates.append((record, score))
                seen_indices.add(idx)

        # Look up by selector pattern (skip records already added by exact match)
        pattern = self._extract_selector_pattern(selector)
        if pattern in self._pattern_index:
            for idx in self._pattern_index[pattern]:
                if idx in seen_indices:
                    continue
                record = self.records[idx]
                # Pattern match gets lower score
                score = 0.5 * record.success_count
                candidates.append((record, score))

        # Filter by URL if provided
        if url:
            url_pattern = self._extract_url_pattern(url)
            candidates = [
                (r, s * 1.5) if r.url_pattern == url_pattern else (r, s * 0.8)
                for r, s in candidates
            ]

        # Filter by element signature if provided
        if element:
            signature = self._compute_element_signature(element)
            candidates = [
                (r, s * 2.0) if r.element_signature == signature else (r, s)
                for r, s in candidates
            ]

        # Sort by score (relevance)
        candidates.sort(key=lambda x: x[1], reverse=True)

        return [r for r, _ in candidates]

    def get_strategy_stats(self) -> dict[str, dict[str, Any]]:
        """
        Get statistics on which strategies work best.

        Returns:
            Dict of strategy -> {success_count, avg_confidence, etc.}
        """
        stats: dict[str, dict[str, Any]] = {}

        for record in self.records:
            strategy = record.strategy_used
            if strategy not in stats:
                stats[strategy] = {
                    "success_count": 0,
                    "total_uses": 0,
                    "avg_confidence": 0.0,
                }

            stats[strategy]["success_count"] += record.success_count
            stats[strategy]["total_uses"] += 1
            # Running average
            n = stats[strategy]["total_uses"]
            old_avg = stats[strategy]["avg_confidence"]
            stats[strategy]["avg_confidence"] = (
                old_avg * (n - 1) + record.confidence
            ) / n

        return stats

    def clear(self) -> None:
        """Clear all history."""
        self.records = []
        self._selector_index = {}
        self._pattern_index = {}
        self._save()

    def _add_record_to_index(self, record: HealingRecord) -> None:
        """Add a record and update indices."""
        idx = len(self.records)
        self.records.append(record)

        # Index by selector
        if record.original_selector not in self._selector_index:
            self._selector_index[record.original_selector] = []
        self._selector_index[record.original_selector].append(idx)

        # Index by pattern
        pattern = self._extract_selector_pattern(record.original_selector)
        if pattern not in self._pattern_index:
            self._pattern_index[pattern] = []
        self._pattern_index[pattern].append(idx)

    def _find_existing_record(
        self,
        original: str,
        healed: str,
        url_pattern: str,
    ) -> HealingRecord | None:
        """Find existing record with same mapping."""
        if original not in self._selector_index:
            return None

        for idx in self._selector_index[original]:
            record = self.records[idx]
            if (
                record.healed_selector == healed
                and record.url_pattern == url_pattern
            ):
                return record

        return None

    def _compute_element_signature(self, element: InteractiveElement) -> str:
        """Compute a signature hash for an element."""
        sig_parts = [
            element.tag_name,
            element.element_type or "",
            element.text or "",
            element.aria_label or "",
        ]
        sig_str = "|".join(sig_parts)
        return hashlib.md5(sig_str.encode()).hexdigest()[:12]

    def _extract_url_pattern(self, url: str) -> str:
        """Extract domain from URL as pattern."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc or url
        except Exception:
            return url

    def _extract_selector_pattern(self, selector: str) -> str:
        """
        Extract a generalized pattern from a selector.

        Removes specific IDs and nth-child indices to create
        a pattern that matches similar selectors.
        """
        # Remove specific nth-child numbers
        pattern = re.sub(r":nth-(?:child|of-type)\(\d+\)", ":nth-*", selector)
        # Keep structure but generalize IDs
        pattern = re.sub(r"#[\w-]+", "#*", pattern)
        return pattern

    def _save(self) -> None:
        """Save history to file."""
        if not self.storage_path:
            return

        try:
            data = {
                "version": 1,
                "records": [r.to_dict() for r in self.records],
            }
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save healing history: {e}")

    def _load(self) -> None:
        """Load history from file."""
        if not self.storage_path or not self.storage_path.exists():
            return

        try:
            with open(self.storage_path) as f:
                data = json.load(f)

            for record_data in data.get("records", []):
                record = HealingRecord.from_dict(record_data)
                self._add_record_to_index(record)

        except Exception as e:
            logger.warning(f"Failed to load healing history: {e}")


class SelectorHealer:
    """
    Automatically repair broken CSS selectors.

    When a selector no longer finds an element, this class tries
    multiple strategies to find the element that moved/changed.

    Strategies (in order):
    1. History lookup - check if this selector was healed before
    2. Selector variations - try similar selectors
    3. Text content match - find by visible text
    4. Attribute match - find by aria-label, data-* attrs
    5. Position match - find element near saved position
    6. LLM recovery - use AI to find similar element

    Learning features:
    - Records successful healings for future reference
    - Uses past repairs to try most likely fix first
    - Tracks strategy effectiveness over time
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        max_attempts: int = 10,
        position_tolerance: int = 50,
        history: HealingHistory | None = None,
        history_storage_path: Path | str | None = None,
    ):
        """
        Initialize the healer.

        Args:
            llm_client: Optional LLM client for AI-based recovery
            max_attempts: Maximum healing attempts per strategy
            position_tolerance: Pixel tolerance for position matching
            history: Optional HealingHistory for learning from past repairs
            history_storage_path: Path to persist healing history (creates new history if none provided)
        """
        self.llm_client = llm_client
        self.max_attempts = max_attempts
        self.position_tolerance = position_tolerance

        # Set up healing history for learning
        if history:
            self.history = history
        elif history_storage_path:
            self.history = HealingHistory(history_storage_path)
        else:
            self.history = HealingHistory()  # In-memory only

    async def heal_selector(
        self,
        broken_selector: str,
        original_element: InteractiveElement,
        page: Page,
        url: str | None = None,
    ) -> HealingResult:
        """
        Attempt to heal a broken selector.

        Args:
            broken_selector: The selector that no longer works
            original_element: The original element data (for context)
            page: Playwright Page to search
            url: Optional URL for history lookup (uses page.url if not provided)

        Returns:
            HealingResult with healed selector if successful
        """
        attempts: list[HealingAttempt] = []
        current_url = url or page.url

        # Strategy 0: History lookup (learned from past repairs)
        result = await self._try_history_lookup(
            broken_selector, original_element, current_url, page, attempts
        )
        if result:
            healing_result = HealingResult(
                success=True,
                original_selector=broken_selector,
                healed_selector=result[0],
                element=result[1],
                confidence=0.95,  # High confidence since it worked before
                strategy_used="history_lookup",
                attempts=attempts,
            )
            # Record to history to increase success count
            self.history.add_record(healing_result, current_url, original_element)
            return healing_result

        # Strategy 1: Selector variations
        result = await self._try_selector_variations(
            broken_selector, page, attempts
        )
        if result:
            healing_result = HealingResult(
                success=True,
                original_selector=broken_selector,
                healed_selector=result[0],
                element=result[1],
                confidence=0.9,
                strategy_used="selector_variation",
                attempts=attempts,
            )
            # Record to history for future learning
            self.history.add_record(healing_result, current_url, original_element)
            return healing_result

        # Strategy 2: Text content match
        if original_element.text:
            result = await self._try_text_match(
                original_element.text, original_element.tag_name, page, attempts
            )
            if result:
                healing_result = HealingResult(
                    success=True,
                    original_selector=broken_selector,
                    healed_selector=result[0],
                    element=result[1],
                    confidence=0.85,
                    strategy_used="text_match",
                    attempts=attempts,
                )
                self.history.add_record(healing_result, current_url, original_element)
                return healing_result

        # Strategy 3: Aria-label match
        if original_element.aria_label:
            result = await self._try_aria_match(
                original_element.aria_label, page, attempts
            )
            if result:
                healing_result = HealingResult(
                    success=True,
                    original_selector=broken_selector,
                    healed_selector=result[0],
                    element=result[1],
                    confidence=0.85,
                    strategy_used="aria_match",
                    attempts=attempts,
                )
                self.history.add_record(healing_result, current_url, original_element)
                return healing_result

        # Strategy 4: Position-based match
        if original_element.bbox:
            result = await self._try_position_match(
                original_element.bbox,
                original_element.tag_name,
                page,
                attempts,
            )
            if result:
                healing_result = HealingResult(
                    success=True,
                    original_selector=broken_selector,
                    healed_selector=result[0],
                    element=result[1],
                    confidence=0.7,
                    strategy_used="position_match",
                    attempts=attempts,
                )
                self.history.add_record(healing_result, current_url, original_element)
                return healing_result

        # Strategy 5: LLM recovery
        if self.llm_client:
            result = await self._try_llm_recovery(
                original_element, page, attempts
            )
            if result:
                healing_result = HealingResult(
                    success=True,
                    original_selector=broken_selector,
                    healed_selector=result[0],
                    element=result[1],
                    confidence=0.75,
                    strategy_used="llm_recovery",
                    attempts=attempts,
                )
                self.history.add_record(healing_result, current_url, original_element)
                return healing_result

        # All strategies failed
        return HealingResult(
            success=False,
            original_selector=broken_selector,
            healed_selector=None,
            element=None,
            confidence=0.0,
            strategy_used="none",
            attempts=attempts,
        )

    async def _try_history_lookup(
        self,
        selector: str,
        element: InteractiveElement,
        url: str,
        page: Page,
        attempts: list[HealingAttempt],
    ) -> tuple[str, ElementHandle] | None:
        """Try selectors from healing history."""
        records = self.history.lookup(selector, url, element)

        for record in records[: self.max_attempts]:
            attempt = HealingAttempt(
                strategy="history_lookup",
                selector_tried=record.healed_selector,
                success=False,
            )
            try:
                element_handle = await page.query_selector(record.healed_selector)
                if element_handle:
                    attempt.success = True
                    attempts.append(attempt)
                    logger.debug(
                        f"History lookup succeeded: {selector} -> {record.healed_selector}"
                    )
                    return record.healed_selector, element_handle
            except Exception as e:
                attempt.error = str(e)

            attempts.append(attempt)

        return None

    async def _try_selector_variations(
        self,
        selector: str,
        page: Page,
        attempts: list[HealingAttempt],
    ) -> tuple[str, ElementHandle] | None:
        """Try variations of the original selector."""
        variations = self._generate_selector_variations(selector)

        for var in variations[: self.max_attempts]:
            attempt = HealingAttempt(
                strategy="selector_variation",
                selector_tried=var,
                success=False,
            )
            try:
                element = await page.query_selector(var)
                if element:
                    attempt.success = True
                    attempts.append(attempt)
                    return var, element
            except Exception as e:
                attempt.error = str(e)

            attempts.append(attempt)

        return None

    async def _try_text_match(
        self,
        text: str,
        tag_name: str,
        page: Page,
        attempts: list[HealingAttempt],
    ) -> tuple[str, ElementHandle] | None:
        """Try to find element by text content."""
        # Escape special characters for XPath
        escaped_text = text.replace("'", "\\'")

        selectors = [
            # Exact text match
            f"//{tag_name}[normalize-space(text())='{escaped_text}']",
            f"//*[normalize-space(text())='{escaped_text}']",
            # Contains text
            f"//{tag_name}[contains(text(), '{escaped_text}')]",
            f"//*[contains(text(), '{escaped_text}')]",
            # CSS with text content (for short unique text)
            f"{tag_name}:has-text('{text}')" if len(text) < 30 else None,
        ]

        for selector in filter(None, selectors):
            attempt = HealingAttempt(
                strategy="text_match",
                selector_tried=selector,
                success=False,
            )
            try:
                element = await page.query_selector(selector)
                if element:
                    attempt.success = True
                    attempts.append(attempt)
                    return selector, element
            except Exception as e:
                attempt.error = str(e)

            attempts.append(attempt)

        return None

    async def _try_aria_match(
        self,
        aria_label: str,
        page: Page,
        attempts: list[HealingAttempt],
    ) -> tuple[str, ElementHandle] | None:
        """Try to find element by aria-label."""
        selectors = [
            f'[aria-label="{aria_label}"]',
            f'[aria-label*="{aria_label}"]',
            f'//*[@aria-label="{aria_label}"]',
        ]

        for selector in selectors:
            attempt = HealingAttempt(
                strategy="aria_match",
                selector_tried=selector,
                success=False,
            )
            try:
                element = await page.query_selector(selector)
                if element:
                    attempt.success = True
                    attempts.append(attempt)
                    return selector, element
            except Exception as e:
                attempt.error = str(e)

            attempts.append(attempt)

        return None

    async def _try_position_match(
        self,
        bbox: BoundingBox,
        tag_name: str,
        page: Page,
        attempts: list[HealingAttempt],
    ) -> tuple[str, ElementHandle] | None:
        """Try to find element by position."""
        attempt = HealingAttempt(
            strategy="position_match",
            selector_tried=f"element_at({bbox.center})",
            success=False,
        )

        try:
            # Find all elements of the same tag
            elements = await page.query_selector_all(tag_name)

            for element in elements:
                elem_bbox = await element.bounding_box()
                if elem_bbox:
                    # Check if center is within tolerance
                    elem_center_x = elem_bbox["x"] + elem_bbox["width"] / 2
                    elem_center_y = elem_bbox["y"] + elem_bbox["height"] / 2

                    dx = abs(elem_center_x - bbox.center[0])
                    dy = abs(elem_center_y - bbox.center[1])

                    if dx <= self.position_tolerance and dy <= self.position_tolerance:
                        # Found element near original position
                        # Generate a selector for it
                        selector = await self._generate_selector_for_element(
                            element, page
                        )
                        attempt.success = True
                        attempt.selector_tried = selector
                        attempts.append(attempt)
                        return selector, element

        except Exception as e:
            attempt.error = str(e)

        attempts.append(attempt)
        return None

    async def _try_llm_recovery(
        self,
        original_element: InteractiveElement,
        page: Page,
        attempts: list[HealingAttempt],
    ) -> tuple[str, ElementHandle] | None:
        """Use LLM to find similar element."""
        from .interactive_element_extractor import InteractiveElementExtractor

        attempt = HealingAttempt(
            strategy="llm_recovery",
            selector_tried="llm_search",
            success=False,
        )

        try:
            # Extract current page elements
            extractor = InteractiveElementExtractor()
            current_elements = await extractor.extract_interactive_elements(
                page, "heal_screenshot"
            )

            if not current_elements:
                attempt.error = "No elements found on page"
                attempts.append(attempt)
                return None

            # Build description of original element
            description = self._build_element_description(original_element)

            # Use LLM to find matching element
            selector = NaturalLanguageSelector(self.llm_client)
            result = await selector.find_element(description, current_elements)

            if result.found and result.element:
                # Get the actual element handle
                element = await page.query_selector(result.element.selector)
                if element:
                    attempt.success = True
                    attempt.selector_tried = result.element.selector
                    attempts.append(attempt)
                    return result.element.selector, element

        except Exception as e:
            attempt.error = str(e)

        attempts.append(attempt)
        return None

    def _generate_selector_variations(self, selector: str) -> list[str]:
        """Generate variations of a CSS selector."""
        variations: list[str] = []

        # Remove nth-child/nth-of-type variations
        no_nth = re.sub(r":nth-(?:child|of-type)\(\d+\)", "", selector)
        if no_nth != selector:
            variations.append(no_nth)

        # Try parent selector only (if has child combinator)
        if " > " in selector:
            parts = selector.rsplit(" > ", 1)
            variations.append(parts[0])
            # Also try with descendant combinator
            variations.append(f"{parts[0]} {parts[1]}")

        # Try without last class
        class_match = re.search(r"\.[\w-]+$", selector)
        if class_match:
            without_last_class = selector[: class_match.start()]
            variations.append(without_last_class)

        # Try with only tag name (if has classes)
        tag_match = re.match(r"^(\w+)", selector)
        if tag_match and "." in selector:
            variations.append(tag_match.group(1))

        # Try with ID only (if present)
        id_match = re.search(r"#[\w-]+", selector)
        if id_match:
            variations.append(id_match.group())

        # Remove duplicates while preserving order
        seen: set[str] = set()
        unique_variations: list[str] = []
        for v in variations:
            if v and v not in seen:
                seen.add(v)
                unique_variations.append(v)

        return unique_variations

    async def _generate_selector_for_element(
        self,
        element: ElementHandle,
        page: Page,
    ) -> str:
        """Generate a CSS selector for an element."""
        try:
            selector = await page.evaluate(
                """(el) => {
                if (el.id) return '#' + CSS.escape(el.id);

                let path = [];
                let current = el;

                while (current && current !== document.body) {
                    let selector = current.tagName.toLowerCase();

                    if (current.id) {
                        selector = '#' + CSS.escape(current.id);
                        path.unshift(selector);
                        break;
                    }

                    if (current.className && typeof current.className === 'string') {
                        const classes = current.className.split(' ')
                            .filter(c => c && !c.includes(':'))
                            .slice(0, 2);
                        if (classes.length > 0) {
                            selector += '.' + classes.map(c => CSS.escape(c)).join('.');
                        }
                    }

                    path.unshift(selector);
                    current = current.parentElement;
                }

                return path.join(' > ');
            }""",
                element,
            )
            return str(selector)
        except Exception:
            return "unknown"

    def _build_element_description(self, element: InteractiveElement) -> str:
        """Build a natural language description of an element for LLM."""
        parts: list[str] = []

        # Element type
        if element.tag_name == "button" or element.element_type == "button":
            parts.append("a button")
        elif element.tag_name == "a" or element.element_type == "link":
            parts.append("a link")
        elif element.tag_name == "input":
            parts.append("an input field")
        else:
            parts.append(f"a {element.tag_name} element")

        # Text content
        if element.text:
            parts.append(f'with text "{element.text}"')

        # Aria label
        if element.aria_label and element.aria_label != element.text:
            parts.append(f'labeled "{element.aria_label}"')

        # Href for links
        if element.href:
            parts.append(f'pointing to "{element.href}"')

        return " ".join(parts)


async def heal_broken_selector(
    selector: str,
    element_data: InteractiveElement,
    page: Page,
    llm_client: LLMClient | None = None,
) -> HealingResult:
    """
    Convenience function to heal a broken selector.

    Args:
        selector: The broken selector
        element_data: Original element data
        page: Playwright Page
        llm_client: Optional LLM client for AI recovery

    Returns:
        HealingResult with healed selector if successful
    """
    healer = SelectorHealer(llm_client=llm_client)
    return await healer.heal_selector(selector, element_data, page)
