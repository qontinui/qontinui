"""
LLM-friendly element formatter.

Formats extracted elements in a way that's easy for LLMs to understand and
reference. Uses numeric indices (like OpenManus) for unambiguous element
selection in natural language interactions.

Example output:
    [0]<button>Submit Form</button>
    [1]<input placeholder="Username">
    [2]<a href="/login">Sign In</a>
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from .frame_manager import FrameAwareElement
from .models import InteractiveElement

logger = logging.getLogger(__name__)


@dataclass
class IndexedElement:
    """Element with a numeric index for LLM reference."""

    index: int
    element: InteractiveElement | FrameAwareElement
    formatted: str  # The formatted string representation

    def to_dict(self) -> dict[str, Any]:
        elem_dict = (
            self.element.to_dict()
            if hasattr(self.element, "to_dict")
            else {"id": str(self.element)}
        )
        return {
            "index": self.index,
            "element": elem_dict,
            "formatted": self.formatted,
        }


@dataclass
class FormattedElementList:
    """A list of elements formatted for LLM consumption."""

    elements: list[IndexedElement] = field(default_factory=list)
    text: str = ""  # The full formatted text

    # Index lookup
    _by_index: dict[int, IndexedElement] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._build_index()

    def _build_index(self) -> None:
        self._by_index = {e.index: e for e in self.elements}

    def get_by_index(self, index: int) -> IndexedElement | None:
        """Get element by its numeric index."""
        return self._by_index.get(index)

    def get_element_by_index(
        self, index: int
    ) -> InteractiveElement | FrameAwareElement | None:
        """Get the underlying element by index."""
        indexed = self._by_index.get(index)
        return indexed.element if indexed else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "elements": [e.to_dict() for e in self.elements],
            "text": self.text,
        }


class LLMFormatter:
    """
    Formats elements for LLM consumption.

    Creates numbered element lists that are easy for LLMs to reference:
    - Simple index-based selection (no complex selectors)
    - Relevant context (tag, text, key attributes)
    - Truncated for token efficiency

    Based on OpenManus's approach of using numeric indices for element
    identification, which is much simpler for LLMs than CSS selectors.
    """

    def __init__(
        self,
        max_text_length: int = 50,
        max_attr_length: int = 30,
        include_href: bool = True,
        include_aria: bool = True,
        include_placeholder: bool = True,
        include_type: bool = True,
    ):
        """
        Initialize the formatter.

        Args:
            max_text_length: Maximum length for element text
            max_attr_length: Maximum length for attribute values
            include_href: Include href attribute for links
            include_aria: Include aria-label attribute
            include_placeholder: Include placeholder for inputs
            include_type: Include input type attribute
        """
        self.max_text_length = max_text_length
        self.max_attr_length = max_attr_length
        self.include_href = include_href
        self.include_aria = include_aria
        self.include_placeholder = include_placeholder
        self.include_type = include_type

    def format_elements(
        self,
        elements: list[InteractiveElement] | list[FrameAwareElement],
    ) -> FormattedElementList:
        """
        Format a list of elements with numeric indices.

        Args:
            elements: List of InteractiveElement or FrameAwareElement

        Returns:
            FormattedElementList with indexed elements and full text
        """
        indexed_elements: list[IndexedElement] = []
        lines: list[str] = []

        for i, element in enumerate(elements):
            # Handle both InteractiveElement and FrameAwareElement
            if isinstance(element, FrameAwareElement):
                inner_elem = element.element
            else:
                inner_elem = element

            formatted = self._format_single_element(i, inner_elem)
            lines.append(formatted)

            indexed_elements.append(
                IndexedElement(
                    index=i,
                    element=element,
                    formatted=formatted,
                )
            )

        result = FormattedElementList(
            elements=indexed_elements,
            text="\n".join(lines),
        )

        return result

    def _format_single_element(self, index: int, element: InteractiveElement) -> str:
        """
        Format a single element.

        Format: [index]<tag attrs>text</tag>
        Example: [0]<button>Submit Form</button>
        """
        tag = element.tag_name
        attrs = self._format_attributes(element)
        text = self._truncate(element.text or element.aria_label or "", self.max_text_length)

        # Build the formatted string
        if attrs:
            return f"[{index}]<{tag} {attrs}>{text}</{tag}>"
        else:
            return f"[{index}]<{tag}>{text}</{tag}>"

    def _format_attributes(self, element: InteractiveElement) -> str:
        """Format relevant attributes for the element."""
        attrs: list[str] = []

        # href for links
        if self.include_href and element.href:
            href = self._truncate(element.href, self.max_attr_length)
            attrs.append(f'href="{href}"')

        # aria-label if different from text
        if self.include_aria and element.aria_label:
            if element.aria_label != element.text:
                label = self._truncate(element.aria_label, self.max_attr_length)
                attrs.append(f'aria-label="{label}"')

        # ARIA role if present
        if element.aria_role:
            attrs.append(f'role="{element.aria_role}"')

        # Element type for classification
        if self.include_type and element.element_type:
            # Only include if it adds information beyond tag name
            if element.element_type not in [element.tag_name, f"aria_{element.aria_role}"]:
                attrs.append(f'type="{element.element_type}"')

        return " ".join(attrs)

    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text to max length, adding ellipsis if needed."""
        if not text:
            return ""
        text = text.strip()
        if len(text) <= max_length:
            return text
        return text[: max_length - 3] + "..."

    def format_for_context(
        self,
        elements: list[InteractiveElement] | list[FrameAwareElement],
        page_url: str = "",
        page_title: str = "",
        include_header: bool = True,
    ) -> str:
        """
        Format elements with page context for LLM consumption.

        Creates a complete context block suitable for LLM prompts:

        ```
        Page: Example Page (https://example.com)
        Interactive Elements:
        [0]<button>Submit</button>
        [1]<a href="/home">Home</a>
        ...
        ```

        Args:
            elements: List of elements to format
            page_url: Current page URL
            page_title: Current page title
            include_header: Whether to include page context header

        Returns:
            Formatted string ready for LLM consumption
        """
        formatted = self.format_elements(elements)

        if not include_header:
            return formatted.text

        lines: list[str] = []

        if page_title or page_url:
            if page_title and page_url:
                lines.append(f"Page: {page_title} ({page_url})")
            elif page_title:
                lines.append(f"Page: {page_title}")
            else:
                lines.append(f"URL: {page_url}")
            lines.append("")

        lines.append(f"Interactive Elements ({len(elements)}):")
        lines.append(formatted.text)

        return "\n".join(lines)

    def format_with_grouping(
        self,
        elements: list[FrameAwareElement],
        group_by_frame: bool = True,
    ) -> str:
        """
        Format frame-aware elements with frame grouping.

        Groups elements by their source frame for clearer organization:

        ```
        Main Frame:
        [0]<button>Submit</button>
        [1]<a href="/home">Home</a>

        Frame: sidebar (iframe#sidebar):
        [2]<button>Menu</button>
        [3]<a href="/settings">Settings</a>
        ```

        Args:
            elements: List of FrameAwareElement
            group_by_frame: Whether to group by frame

        Returns:
            Formatted string with frame grouping
        """
        if not group_by_frame:
            return self.format_elements(elements).text

        # Group elements by frame
        by_frame: dict[int, list[tuple[int, FrameAwareElement]]] = {}
        for i, elem in enumerate(elements):
            frame_id = elem.frame_id
            if frame_id not in by_frame:
                by_frame[frame_id] = []
            by_frame[frame_id].append((i, elem))

        # Format each group
        lines: list[str] = []
        for frame_id in sorted(by_frame.keys()):
            frame_elements = by_frame[frame_id]

            # Frame header
            if frame_id == 0:
                lines.append("Main Frame:")
            else:
                # Get frame selector from first element
                frame_selector = frame_elements[0][1].frame_selector
                lines.append(f"Frame ({frame_selector}):")

            # Format elements in this frame
            for global_index, elem in frame_elements:
                formatted = self._format_single_element(global_index, elem.element)
                lines.append(formatted)

            lines.append("")  # Blank line between groups

        return "\n".join(lines).strip()


def format_for_llm(
    elements: list[InteractiveElement] | list[FrameAwareElement],
    page_url: str = "",
    page_title: str = "",
) -> str:
    """
    Convenience function to format elements for LLM consumption.

    Args:
        elements: List of elements to format
        page_url: Current page URL
        page_title: Current page title

    Returns:
        Formatted string ready for LLM prompts
    """
    formatter = LLMFormatter()
    return formatter.format_for_context(
        elements=elements,
        page_url=page_url,
        page_title=page_title,
    )


def get_element_by_index(
    elements: list[InteractiveElement] | list[FrameAwareElement],
    index: int,
) -> InteractiveElement | FrameAwareElement | None:
    """
    Get an element by its LLM-assigned index.

    Args:
        elements: The original element list
        index: Index returned by LLM

    Returns:
        The element at that index, or None if out of bounds
    """
    if 0 <= index < len(elements):
        return elements[index]
    return None


def parse_llm_element_reference(response: str) -> int | None:
    """
    Parse an LLM's element reference to get the index.

    Handles various formats:
    - "5" -> 5
    - "[5]" -> 5
    - "element 5" -> 5
    - "index 5" -> 5

    Args:
        response: LLM's response containing an element reference

    Returns:
        The index, or None if unparseable
    """
    import re

    # Clean the response
    response = response.strip().lower()

    # Try direct number
    if response.isdigit():
        return int(response)

    # Try [number] format
    match = re.search(r"\[(\d+)\]", response)
    if match:
        return int(match.group(1))

    # Try "element N" or "index N" format
    match = re.search(r"(?:element|index|item)\s*(\d+)", response)
    if match:
        return int(match.group(1))

    # Try any number in the response
    match = re.search(r"(\d+)", response)
    if match:
        return int(match.group(1))

    return None
