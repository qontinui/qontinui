"""
Natural language element selector.

This module enables AI-powered element selection using natural language
descriptions. Similar to OpenManus and Stagehand's approaches, it uses
LLMs to match human descriptions to DOM elements.

Key Features
------------
- **Natural Language Matching**: Find elements by description
- **Multi-Element Search**: Find all elements matching a pattern
- **Action Selection**: Determine both element and action from instruction
- **Confidence Scoring**: Each match includes confidence and alternatives
- **Fallback Support**: Text-based matching when LLM unavailable

Classes
-------
NaturalLanguageSelector
    Main selector using LLM for matching.
FallbackSelector
    Simple text/role-based matching without LLM.
SelectionResult
    Result container with element, confidence, reasoning.

Functions
---------
find_element_by_description
    Convenience function with automatic fallback.

Usage Examples
--------------
Find single element::

    from qontinui.extraction.web import NaturalLanguageSelector, AnthropicClient

    client = AnthropicClient()
    selector = NaturalLanguageSelector(client)

    result = await selector.find_element("the blue submit button", elements)
    if result.found:
        print(f"Found: {result.element.text} (confidence: {result.confidence})")
        print(f"Reasoning: {result.reasoning}")

Find multiple matching elements::

    results = await selector.find_multiple("all navigation links", elements)
    for result in results:
        print(f"[{result.index}] {result.element.text} ({result.confidence})")

Select element AND action::

    result, action = await selector.select_action(
        "click the login button",
        elements
    )
    # action = "click", result.element = login button

With fallback (no LLM)::

    from qontinui.extraction.web import find_element_by_description

    # Uses text matching if no client provided
    result = await find_element_by_description("Submit", elements)

Using FallbackSelector directly::

    from qontinui.extraction.web import FallbackSelector

    fallback = FallbackSelector()
    result = fallback.find_by_text("Login", elements)
    buttons = fallback.find_by_role("button", elements, text_hint="Submit")

See Also
--------
- llm_clients: LLM client adapters (Anthropic, OpenAI, LiteLLM)
- selector_healer: Automatic selector repair
- llm_formatter: Format elements for LLM consumption
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .frame_manager import FrameAwareElement
from .llm_formatter import LLMFormatter
from .models import InteractiveElement

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Protocol for LLM client implementations."""

    async def complete(self, prompt: str) -> str:
        """Complete a prompt and return the response text."""
        ...


@dataclass
class SelectionResult:
    """Result of a natural language element selection."""

    element: InteractiveElement | FrameAwareElement | None
    index: int | None
    confidence: float
    reasoning: str
    alternatives: list[int] = field(default_factory=list)

    @property
    def found(self) -> bool:
        """Whether an element was found."""
        return self.element is not None

    def to_dict(self) -> dict[str, Any]:
        elem_dict = None
        if self.element:
            if hasattr(self.element, "to_dict"):
                elem_dict = self.element.to_dict()
            else:
                elem_dict = {"id": str(self.element)}

        return {
            "found": self.found,
            "index": self.index,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "alternatives": self.alternatives,
            "element": elem_dict,
        }


class NaturalLanguageSelector:
    """
    Select elements using natural language descriptions.

    Uses an LLM to match natural language descriptions to indexed elements,
    similar to how OpenManus and Stagehand work.

    Example:
        selector = NaturalLanguageSelector(llm_client)
        result = await selector.find_element(
            "the login button",
            elements
        )
        if result.found:
            await result.element.click()
    """

    def __init__(
        self,
        llm_client: LLMClient,
        formatter: LLMFormatter | None = None,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize the selector.

        Args:
            llm_client: LLM client for natural language processing
            formatter: Optional custom formatter (uses default if None)
            confidence_threshold: Minimum confidence for selection
        """
        self.llm = llm_client
        self.formatter = formatter or LLMFormatter()
        self.confidence_threshold = confidence_threshold

    async def find_element(
        self,
        description: str,
        elements: list[InteractiveElement] | list[FrameAwareElement],
        context: str = "",
    ) -> SelectionResult:
        """
        Find an element matching a natural language description.

        Args:
            description: Natural language description of the element
            elements: List of elements to search
            context: Optional additional context about the page/task

        Returns:
            SelectionResult with the matched element and metadata
        """
        if not elements:
            return SelectionResult(
                element=None,
                index=None,
                confidence=0.0,
                reasoning="No elements provided",
            )

        # Format elements for LLM
        formatted = self.formatter.format_elements(elements)

        # Build the prompt
        prompt = self._build_selection_prompt(
            description=description,
            element_list=formatted.text,
            context=context,
        )

        # Get LLM response
        try:
            response = await self.llm.complete(prompt)
            result = self._parse_selection_response(response, elements)

            if result.confidence < self.confidence_threshold:
                logger.warning(f"Low confidence ({result.confidence:.2f}) for: {description}")

            return result

        except Exception as e:
            logger.error(f"LLM selection failed: {e}")
            return SelectionResult(
                element=None,
                index=None,
                confidence=0.0,
                reasoning=f"LLM error: {e}",
            )

    async def find_multiple(
        self,
        description: str,
        elements: list[InteractiveElement] | list[FrameAwareElement],
        max_results: int = 5,
    ) -> list[SelectionResult]:
        """
        Find multiple elements matching a description.

        Useful for finding all buttons of a certain type, all links
        in a navigation area, etc.

        Args:
            description: Natural language description
            elements: List of elements to search
            max_results: Maximum number of results to return

        Returns:
            List of SelectionResult objects, sorted by confidence
        """
        if not elements:
            return []

        formatted = self.formatter.format_elements(elements)

        prompt = self._build_multi_selection_prompt(
            description=description,
            element_list=formatted.text,
            max_results=max_results,
        )

        try:
            response = await self.llm.complete(prompt)
            results = self._parse_multi_response(response, elements)
            return sorted(results, key=lambda r: r.confidence, reverse=True)[:max_results]

        except Exception as e:
            logger.error(f"LLM multi-selection failed: {e}")
            return []

    async def select_action(
        self,
        instruction: str,
        elements: list[InteractiveElement] | list[FrameAwareElement],
        available_actions: list[str] | None = None,
    ) -> tuple[SelectionResult, str]:
        """
        Select an element AND determine the action to perform.

        Similar to Stagehand's act() method - given an instruction,
        determine both what element to interact with and how.

        Args:
            instruction: Natural language instruction (e.g., "click the submit button")
            elements: List of elements
            available_actions: List of possible actions (defaults to click, type, hover)

        Returns:
            Tuple of (SelectionResult, action_name)
        """
        if available_actions is None:
            available_actions = ["click", "type", "hover", "focus", "select"]

        if not elements:
            return (
                SelectionResult(
                    element=None,
                    index=None,
                    confidence=0.0,
                    reasoning="No elements provided",
                ),
                "none",
            )

        formatted = self.formatter.format_elements(elements)

        prompt = self._build_action_prompt(
            instruction=instruction,
            element_list=formatted.text,
            actions=available_actions,
        )

        try:
            response = await self.llm.complete(prompt)
            result, action = self._parse_action_response(response, elements)
            return result, action

        except Exception as e:
            logger.error(f"LLM action selection failed: {e}")
            return (
                SelectionResult(
                    element=None,
                    index=None,
                    confidence=0.0,
                    reasoning=f"LLM error: {e}",
                ),
                "none",
            )

    def _build_selection_prompt(
        self,
        description: str,
        element_list: str,
        context: str = "",
    ) -> str:
        """Build the prompt for single element selection."""
        context_section = f"\nContext: {context}\n" if context else ""

        return f"""You are an element selector. Given a list of interactive elements and a description, find the element that best matches.

{context_section}
Interactive Elements:
{element_list}

Description: "{description}"

Instructions:
1. Analyze the description carefully - look for:
   - Exact text matches (highest priority)
   - aria-label matches (high priority)
   - Element type hints (button, link, input, etc.)
   - Position hints (first, main, primary)
2. Match by semantic meaning, not just keywords
3. Consider the element tag: <a> = link, <button> = button, <input> = input field
4. Return your answer in this exact format:
   INDEX: <number or "none">
   CONFIDENCE: <0.0 to 1.0>
   REASONING: <brief explanation>
   ALTERNATIVES: <comma-separated indices of other possible matches, or "none">

Confidence guidelines:
- 0.95-1.0: Exact text/aria-label match
- 0.80-0.94: Strong semantic match (correct type + related text)
- 0.60-0.79: Partial match (either type or text matches)
- 0.40-0.59: Weak match (only vaguely related)
- Below 0.40: No good match found

Example response:
INDEX: 3
CONFIDENCE: 0.95
REASONING: Element 3 is a button with text "Submit" which matches the description of a submit button.
ALTERNATIVES: 7, 12

Your response:"""

    def _build_multi_selection_prompt(
        self,
        description: str,
        element_list: str,
        max_results: int,
    ) -> str:
        """Build the prompt for multiple element selection."""
        return f"""You are an element selector. Find ALL elements matching the description, up to {max_results} results.

Interactive Elements:
{element_list}

Description: "{description}"

Instructions:
1. Find all elements that match the description
2. List each match with its confidence score
3. Return in this format (one per line):
   MATCH: <index>, <confidence 0.0-1.0>, <brief reason>

Example:
MATCH: 3, 0.95, Button with "Submit" text
MATCH: 7, 0.80, Similar button in footer
MATCH: 12, 0.60, Generic button that could match

Your response:"""

    def _build_action_prompt(
        self,
        instruction: str,
        element_list: str,
        actions: list[str],
    ) -> str:
        """Build the prompt for action selection."""
        actions_str = ", ".join(actions)

        return f"""You are a web automation assistant. Given an instruction, select the element and action.

Interactive Elements:
{element_list}

Instruction: "{instruction}"

Available Actions: {actions_str}

Instructions:
1. Parse the instruction to identify:
   - The target element description
   - The intended action (click, type, hover, etc.)
2. Find the element that best matches the description
3. If the action is not explicit, infer from context:
   - Buttons/links -> "click"
   - Input fields with "type", "enter", "fill" -> "type"
   - Menus with "hover" -> "hover"
   - Form fields with "focus" -> "focus"
   - Dropdowns with "select" -> "select"
4. Return in this exact format:
   INDEX: <number or "none">
   ACTION: <action name from available actions>
   CONFIDENCE: <0.0 to 1.0>
   REASONING: <brief explanation>

Action hints in instructions:
- "click", "press", "tap" -> click
- "type", "enter", "fill", "input" -> type
- "hover", "mouse over" -> hover
- "focus", "tab to" -> focus
- "select", "choose", "pick" -> select

Example:
INDEX: 5
ACTION: click
CONFIDENCE: 0.95
REASONING: Element 5 is the submit button mentioned in the instruction.

Your response:"""

    def _parse_selection_response(
        self,
        response: str,
        elements: list[InteractiveElement] | list[FrameAwareElement],
    ) -> SelectionResult:
        """Parse the LLM response for single element selection."""
        # Extract index
        index_match = re.search(r"INDEX:\s*(\d+|none)", response, re.IGNORECASE)
        index = None
        if index_match:
            val = index_match.group(1).lower()
            if val != "none":
                index = int(val)

        # Extract confidence
        conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", response, re.IGNORECASE)
        confidence = float(conf_match.group(1)) if conf_match else 0.5

        # Extract reasoning
        reason_match = re.search(
            r"REASONING:\s*(.+?)(?=\n[A-Z]+:|$)", response, re.IGNORECASE | re.DOTALL
        )
        reasoning = reason_match.group(1).strip() if reason_match else ""

        # Extract alternatives
        alt_match = re.search(r"ALTERNATIVES:\s*(.+)", response, re.IGNORECASE)
        alternatives: list[int] = []
        if alt_match:
            alt_str = alt_match.group(1).strip().lower()
            if alt_str != "none":
                alternatives = [int(x.strip()) for x in alt_str.split(",") if x.strip().isdigit()]

        # Get the element
        element = None
        if index is not None and 0 <= index < len(elements):
            element = elements[index]

        return SelectionResult(
            element=element,
            index=index,
            confidence=confidence,
            reasoning=reasoning,
            alternatives=alternatives,
        )

    def _parse_multi_response(
        self,
        response: str,
        elements: list[InteractiveElement] | list[FrameAwareElement],
    ) -> list[SelectionResult]:
        """Parse the LLM response for multiple element selection."""
        results: list[SelectionResult] = []

        # Find all MATCH lines
        matches = re.findall(
            r"MATCH:\s*(\d+)\s*,\s*([\d.]+)\s*,\s*(.+?)(?=\nMATCH:|$)",
            response,
            re.IGNORECASE,
        )

        for idx_str, conf_str, reason in matches:
            idx = int(idx_str)
            confidence = float(conf_str)

            element = None
            if 0 <= idx < len(elements):
                element = elements[idx]

            results.append(
                SelectionResult(
                    element=element,
                    index=idx,
                    confidence=confidence,
                    reasoning=reason.strip(),
                )
            )

        return results

    def _parse_action_response(
        self,
        response: str,
        elements: list[InteractiveElement] | list[FrameAwareElement],
    ) -> tuple[SelectionResult, str]:
        """Parse the LLM response for action selection."""
        # Parse selection result
        result = self._parse_selection_response(response, elements)

        # Extract action
        action_match = re.search(r"ACTION:\s*(\w+)", response, re.IGNORECASE)
        action = action_match.group(1).lower() if action_match else "click"

        return result, action


class FallbackSelector:
    """
    Fallback element selector using simple text matching.

    Used when LLM is unavailable or for quick local matching.
    """

    def find_by_text(
        self,
        text: str,
        elements: list[InteractiveElement] | list[FrameAwareElement],
        case_sensitive: bool = False,
    ) -> SelectionResult:
        """
        Find element by exact or partial text match.

        Args:
            text: Text to search for
            elements: List of elements
            case_sensitive: Whether to match case

        Returns:
            SelectionResult with best match
        """
        if not case_sensitive:
            text = text.lower()

        best_match = None
        best_index = None
        best_score = 0.0

        for i, elem in enumerate(elements):
            # Get the inner element if frame-aware
            inner = elem.element if isinstance(elem, FrameAwareElement) else elem

            # Check text content
            elem_text = inner.text or ""
            aria_label = inner.aria_label or ""

            if not case_sensitive:
                elem_text = elem_text.lower()
                aria_label = aria_label.lower()

            # Exact match on text
            if elem_text == text:
                return SelectionResult(
                    element=elem,
                    index=i,
                    confidence=1.0,
                    reasoning="Exact text match",
                )

            # Exact match on aria-label
            if aria_label == text:
                return SelectionResult(
                    element=elem,
                    index=i,
                    confidence=0.95,
                    reasoning="Exact aria-label match",
                )

            # Partial match
            if text in elem_text or text in aria_label:
                score = len(text) / max(len(elem_text), len(aria_label), 1)
                if score > best_score:
                    best_score = score
                    best_match = elem
                    best_index = i

        if best_match:
            return SelectionResult(
                element=best_match,
                index=best_index,
                confidence=best_score * 0.8,  # Partial matches get lower confidence
                reasoning="Partial text match",
            )

        return SelectionResult(
            element=None,
            index=None,
            confidence=0.0,
            reasoning="No text match found",
        )

    def find_by_role(
        self,
        role: str,
        elements: list[InteractiveElement] | list[FrameAwareElement],
        text_hint: str | None = None,
    ) -> list[SelectionResult]:
        """
        Find all elements with a specific role.

        Args:
            role: Element role (button, link, etc.)
            elements: List of elements
            text_hint: Optional text hint for filtering

        Returns:
            List of matching SelectionResults
        """
        results: list[SelectionResult] = []
        role_lower = role.lower()

        for i, elem in enumerate(elements):
            inner = elem.element if isinstance(elem, FrameAwareElement) else elem

            # Check element type and aria role
            matches_role = (
                inner.element_type.lower() == role_lower
                or inner.tag_name.lower() == role_lower
                or (inner.aria_role and inner.aria_role.lower() == role_lower)
            )

            if matches_role:
                # Calculate confidence based on text hint
                confidence = 0.8
                if text_hint:
                    elem_text = (inner.text or "").lower()
                    aria_label = (inner.aria_label or "").lower()
                    hint_lower = text_hint.lower()

                    if hint_lower in elem_text or hint_lower in aria_label:
                        confidence = 0.95
                    else:
                        confidence = 0.6

                results.append(
                    SelectionResult(
                        element=elem,
                        index=i,
                        confidence=confidence,
                        reasoning=f"Matches role '{role}'",
                    )
                )

        return sorted(results, key=lambda r: r.confidence, reverse=True)


async def find_element_by_description(
    description: str,
    elements: list[InteractiveElement] | list[FrameAwareElement],
    llm_client: LLMClient | None = None,
) -> SelectionResult:
    """
    Convenience function to find an element by description.

    Uses LLM if available, falls back to text matching otherwise.

    Args:
        description: Natural language description
        elements: List of elements
        llm_client: Optional LLM client

    Returns:
        SelectionResult with best match
    """
    if llm_client:
        selector = NaturalLanguageSelector(llm_client)
        return await selector.find_element(description, elements)
    else:
        # Fallback to simple text matching
        fallback = FallbackSelector()
        return fallback.find_by_text(description, elements)
