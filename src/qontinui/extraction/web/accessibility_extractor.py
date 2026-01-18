"""
Accessibility tree extractor.

This module extracts the accessibility tree from web pages using Chrome DevTools
Protocol (CDP) and merges it with DOM extraction data for enhanced semantic
understanding of page structure.

Key Features
------------
- **Accessibility Tree Extraction**: Full a11y tree via CDP
- **Element Enrichment**: Merge a11y data with DOM elements
- **Role-Based Search**: Find elements by ARIA role
- **Name-Based Lookup**: Fast name-based node lookup
- **Retry Logic**: Configurable retries for transient CDP failures
- **Graceful Degradation**: Return empty tree on failures

Classes
-------
A11yNode
    A node in the accessibility tree with role, name, state.
A11yTree
    Complete accessibility tree with lookup indexes.
EnrichedElement
    InteractiveElement merged with accessibility data.
AccessibilityExtractor
    Main extractor class for a11y tree operations.

Functions
---------
extract_accessibility_tree
    Convenience function for tree extraction.
enrich_with_accessibility
    Convenience function to enrich elements.
a11y_tree_to_text
    Convert tree to LLM-friendly text format.

Usage Examples
--------------
Extract accessibility tree::

    from qontinui.extraction.web import extract_accessibility_tree

    tree = await extract_accessibility_tree(page)
    print(f"Tree has {tree.node_count} nodes")
    print(tree.to_text())

Search by role and name::

    # Find all buttons
    buttons = tree.find_by_role("button")

    # Find by accessible name
    matches = tree.find_by_name("Submit")

Enrich DOM elements with a11y data::

    from qontinui.extraction.web import enrich_with_accessibility

    enriched = await enrich_with_accessibility(elements, page)
    for elem in enriched:
        print(f"{elem.a11y_role}: {elem.a11y_name} (confidence: {elem.match_confidence})")

With AccessibilityExtractor::

    from qontinui.extraction.web import AccessibilityExtractor

    extractor = AccessibilityExtractor()
    tree = await extractor.extract_tree(page)

    # Get only interactive nodes
    interactive = await extractor.extract_interactive_nodes(page)

See Also
--------
- hybrid_extractor: Combined DOM + screenshot + a11y context
- interactive_element_extractor: Base DOM extraction
- llm_formatter: Format elements for LLM consumption
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from playwright.async_api import Page

from .exceptions import ExtractionTimeoutError, with_timeout
from .models import InteractiveElement

logger = logging.getLogger(__name__)

# Default settings
DEFAULT_CDP_TIMEOUT = 30.0
DEFAULT_CDP_RETRIES = 3
DEFAULT_RETRY_DELAY = 0.5


@dataclass
class A11yNode:
    """A node in the accessibility tree."""

    role: str  # ARIA role (button, link, heading, etc.)
    name: str  # Accessible name
    description: str  # Accessible description

    # Value for inputs/controls
    value: str | int | float | bool | None = None

    # State
    checked: bool | None = None
    disabled: bool | None = None
    expanded: bool | None = None
    focused: bool = False
    pressed: bool | None = None
    selected: bool | None = None

    # Structure
    level: int | None = None  # Heading level, tree level
    children: list["A11yNode"] = field(default_factory=list)

    # For matching with DOM elements
    dom_selector: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "name": self.name,
            "description": self.description,
            "value": self.value,
            "checked": self.checked,
            "disabled": self.disabled,
            "expanded": self.expanded,
            "focused": self.focused,
            "pressed": self.pressed,
            "selected": self.selected,
            "level": self.level,
            "children": [c.to_dict() for c in self.children],
            "dom_selector": self.dom_selector,
        }

    @classmethod
    def from_playwright(cls, pw_node: dict[str, Any]) -> "A11yNode":
        """Create from Playwright accessibility snapshot node."""
        return cls(
            role=pw_node.get("role", ""),
            name=pw_node.get("name", ""),
            description=pw_node.get("description", ""),
            value=pw_node.get("value"),
            checked=pw_node.get("checked"),
            disabled=pw_node.get("disabled"),
            expanded=pw_node.get("expanded"),
            focused=pw_node.get("focused", False),
            pressed=pw_node.get("pressed"),
            selected=pw_node.get("selected"),
            level=pw_node.get("level"),
            children=[cls.from_playwright(c) for c in pw_node.get("children", [])],
        )


@dataclass
class A11yTree:
    """Complete accessibility tree for a page."""

    root: A11yNode | None = None
    node_count: int = 0

    # Flattened lookup by role+name for matching
    _by_name: dict[str, list[A11yNode]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.root:
            self._build_index(self.root)

    def _build_index(self, node: A11yNode) -> None:
        """Build lookup index from the tree."""
        self.node_count += 1

        if node.name:
            key = node.name.lower().strip()
            if key not in self._by_name:
                self._by_name[key] = []
            self._by_name[key].append(node)

        for child in node.children:
            self._build_index(child)

    def find_by_name(self, name: str) -> list[A11yNode]:
        """Find nodes by accessible name (case-insensitive)."""
        key = name.lower().strip()
        return self._by_name.get(key, [])

    def find_by_role(self, role: str) -> list[A11yNode]:
        """Find all nodes with a specific role."""
        results: list[A11yNode] = []
        if self.root:
            self._find_by_role_recursive(self.root, role, results)
        return results

    def _find_by_role_recursive(self, node: A11yNode, role: str, results: list[A11yNode]) -> None:
        if node.role == role:
            results.append(node)
        for child in node.children:
            self._find_by_role_recursive(child, role, results)

    def to_text(self, indent: int = 0) -> str:
        """Convert tree to indented text representation."""
        if not self.root:
            return "(empty)"
        return self._node_to_text(self.root, indent)

    def _node_to_text(self, node: A11yNode, indent: int) -> str:
        """Convert a single node to text."""
        prefix = "  " * indent
        parts = [node.role]

        if node.name:
            parts.append(f'"{node.name}"')
        if node.description:
            parts.append(f"({node.description})")
        if node.value is not None:
            parts.append(f"value={node.value}")
        if node.checked is not None:
            parts.append("checked" if node.checked else "unchecked")
        if node.expanded is not None:
            parts.append("expanded" if node.expanded else "collapsed")
        if node.disabled:
            parts.append("disabled")

        line = prefix + " ".join(parts)
        lines = [line]

        for child in node.children:
            lines.append(self._node_to_text(child, indent + 1))

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root.to_dict() if self.root else None,
            "node_count": self.node_count,
        }


@dataclass
class EnrichedElement:
    """InteractiveElement enriched with accessibility data."""

    element: InteractiveElement

    # Accessibility data
    a11y_role: str | None = None
    a11y_name: str | None = None
    a11y_description: str | None = None
    a11y_value: str | int | float | bool | None = None

    # State from accessibility tree
    a11y_checked: bool | None = None
    a11y_disabled: bool | None = None
    a11y_expanded: bool | None = None
    a11y_pressed: bool | None = None
    a11y_selected: bool | None = None

    # Confidence of the match
    match_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "element": self.element.to_dict(),
            "a11y_role": self.a11y_role,
            "a11y_name": self.a11y_name,
            "a11y_description": self.a11y_description,
            "a11y_value": self.a11y_value,
            "a11y_checked": self.a11y_checked,
            "a11y_disabled": self.a11y_disabled,
            "a11y_expanded": self.a11y_expanded,
            "a11y_pressed": self.a11y_pressed,
            "a11y_selected": self.a11y_selected,
            "match_confidence": self.match_confidence,
        }


class AccessibilityExtractor:
    """
    Extracts and processes accessibility trees.

    Uses Playwright's accessibility API to get the full accessibility tree,
    then provides methods to match DOM elements to accessibility nodes.

    Production features:
    - CDP session caching (reuse session across extractions)
    - Retry logic for transient CDP failures
    - Configurable timeouts
    - Graceful degradation on failures
    """

    def __init__(
        self,
        cache_cdp_session: bool = True,
        max_retries: int = DEFAULT_CDP_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        timeout_seconds: float = DEFAULT_CDP_TIMEOUT,
    ) -> None:
        """
        Initialize the extractor.

        Args:
            cache_cdp_session: Whether to cache CDP sessions for reuse (default: True).
            max_retries: Maximum number of retry attempts for CDP calls.
            retry_delay: Initial delay between retries in seconds.
            timeout_seconds: Timeout for CDP operations.
        """
        self._cache: dict[str, A11yTree] = {}
        self._cache_cdp_session = cache_cdp_session
        self._cached_client: Any = None
        self._cached_page_id: int | None = None
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._timeout_seconds = timeout_seconds

    async def _get_cdp_client(self, page: Page) -> Any:
        """
        Get or create a CDP client for the page.

        If caching is enabled and a cached client exists for this page,
        returns the cached client. Otherwise creates a new one.
        """
        page_id = id(page)

        # Check if we have a cached client for this page
        if self._cache_cdp_session and self._cached_client and self._cached_page_id == page_id:
            return self._cached_client

        # Create new client
        client = await page.context.new_cdp_session(page)

        # Cache if enabled
        if self._cache_cdp_session:
            # Detach old client if exists
            if self._cached_client:
                try:
                    await self._cached_client.detach()
                except Exception:
                    pass
            self._cached_client = client
            self._cached_page_id = page_id

        return client

    async def close(self) -> None:
        """Close any cached CDP session."""
        if self._cached_client:
            try:
                await self._cached_client.detach()
            except Exception:
                pass
            self._cached_client = None
            self._cached_page_id = None

    async def extract_tree(
        self,
        page: Page,
        timeout: float | None = None,
    ) -> A11yTree:
        """
        Extract the full accessibility tree from a page.

        Uses CDP (Chrome DevTools Protocol) to get the accessibility tree
        since page.accessibility was removed from Playwright Python.

        Production features:
        - Reuses CDP session if caching is enabled
        - Retry logic for transient failures
        - Configurable timeout

        Args:
            page: Playwright Page to extract from.
            timeout: Timeout in seconds (uses default if not specified).

        Returns:
            A11yTree representing the page's accessibility structure.

        Raises:
            ExtractionTimeoutError: If extraction times out.
            CDPError: If CDP operations fail after retries.
        """
        actual_timeout = timeout if timeout is not None else self._timeout_seconds

        try:
            return await with_timeout(
                self._extract_tree_with_retry(page),
                timeout_seconds=actual_timeout,
                operation_name="extract_accessibility_tree",
            )
        except ExtractionTimeoutError:
            logger.warning("Accessibility tree extraction timed out, returning empty tree")
            return A11yTree()

    async def _extract_tree_with_retry(self, page: Page) -> A11yTree:
        """
        Extract accessibility tree with retry logic.
        """
        for attempt in range(self._max_retries + 1):
            client = None
            should_detach = not self._cache_cdp_session

            try:
                client = await self._get_cdp_client(page)
                result = await client.send("Accessibility.getFullAXTree")
                nodes = result.get("nodes", [])

                if not nodes:
                    logger.warning("No accessibility nodes found")
                    return A11yTree()

                # Build tree from flat CDP nodes
                root = self._build_tree_from_cdp_nodes(nodes)
                tree = A11yTree(root=root)

                logger.info(f"Extracted accessibility tree with {tree.node_count} nodes")
                return tree

            except Exception as e:
                # Invalidate cache on error
                if self._cache_cdp_session:
                    self._cached_client = None
                    self._cached_page_id = None

                if attempt == self._max_retries:
                    logger.error(
                        f"Failed to extract accessibility tree after "
                        f"{self._max_retries + 1} attempts: {e}"
                    )
                    # Return empty tree for graceful degradation
                    return A11yTree()

                logger.warning(
                    f"Accessibility extraction attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {self._retry_delay}s..."
                )
                await asyncio.sleep(self._retry_delay)

            finally:
                # Only detach if not caching
                if should_detach and client:
                    try:
                        await client.detach()
                    except Exception:
                        pass

        # Should never reach here, but satisfy type checker
        return A11yTree()

    def _build_tree_from_cdp_nodes(self, nodes: list[dict]) -> A11yNode | None:
        """Build A11yTree from flat CDP AXNode list."""
        if not nodes:
            return None

        # Create node lookup by nodeId
        node_map: dict[str, dict] = {}
        for node in nodes:
            node_id = node.get("nodeId")
            if node_id:
                node_map[node_id] = node

        # Find root node (first one or one with no parent)
        root_data = nodes[0] if nodes else None

        def convert_node(cdp_node: dict) -> A11yNode:
            """Convert CDP AXNode to A11yNode."""
            # Extract properties
            props = {}
            for prop in cdp_node.get("properties", []):
                props[prop.get("name", "")] = prop.get("value", {}).get("value")

            # Get name and description from properties or name field
            name_obj = cdp_node.get("name", {})
            name = name_obj.get("value", "") if isinstance(name_obj, dict) else ""

            desc_obj = cdp_node.get("description", {})
            description = desc_obj.get("value", "") if isinstance(desc_obj, dict) else ""

            # Get role
            role_obj = cdp_node.get("role", {})
            role = role_obj.get("value", "") if isinstance(role_obj, dict) else ""

            # Get value
            value_obj = cdp_node.get("value", {})
            value = value_obj.get("value") if isinstance(value_obj, dict) else None

            # Build children recursively
            children = []
            for child_id in cdp_node.get("childIds", []):
                if child_id in node_map:
                    children.append(convert_node(node_map[child_id]))

            return A11yNode(
                role=role,
                name=name,
                description=description,
                value=value,
                checked=props.get("checked"),
                disabled=props.get("disabled"),
                expanded=props.get("expanded"),
                focused=props.get("focused", False),
                pressed=props.get("pressed"),
                selected=props.get("selected"),
                level=props.get("level"),
                children=children,
            )

        return convert_node(root_data) if root_data else None

    async def extract_interactive_nodes(self, page: Page) -> list[A11yNode]:
        """
        Extract only interactive nodes from the accessibility tree.

        These are nodes that users can interact with (buttons, links, etc.)
        """
        tree = await self.extract_tree(page)

        interactive_roles = {
            "button",
            "link",
            "checkbox",
            "radio",
            "switch",
            "slider",
            "spinbutton",
            "combobox",
            "listbox",
            "textbox",
            "searchbox",
            "menuitem",
            "menuitemcheckbox",
            "menuitemradio",
            "tab",
            "treeitem",
            "option",
        }

        results: list[A11yNode] = []
        for role in interactive_roles:
            results.extend(tree.find_by_role(role))

        return results

    def match_element_to_a11y(
        self,
        element: InteractiveElement,
        tree: A11yTree,
    ) -> tuple[A11yNode | None, float]:
        """
        Find the accessibility node that matches a DOM element.

        Matching strategy:
        1. Match by text content (exact)
        2. Match by aria-label (exact)
        3. Match by text content (fuzzy)

        Args:
            element: The DOM element to match
            tree: The accessibility tree to search

        Returns:
            Tuple of (matching A11yNode or None, confidence score)
        """
        if not tree.root:
            return None, 0.0

        # Try exact text match
        if element.text:
            matches = tree.find_by_name(element.text)
            if matches:
                # Prefer matching role
                for match in matches:
                    if self._roles_compatible(element, match):
                        return match, 1.0
                return matches[0], 0.8

        # Try aria-label match
        if element.aria_label:
            matches = tree.find_by_name(element.aria_label)
            if matches:
                for match in matches:
                    if self._roles_compatible(element, match):
                        return match, 1.0
                return matches[0], 0.8

        # Try role-based matching
        role = self._element_to_a11y_role(element)
        if role:
            role_matches = tree.find_by_role(role)
            for match in role_matches:
                if self._fuzzy_name_match(element, match):
                    return match, 0.6

        return None, 0.0

    def _roles_compatible(self, element: InteractiveElement, node: A11yNode) -> bool:
        """Check if element type and a11y role are compatible."""
        role_map = {
            "button": {"button"},
            "a": {"link"},
            "input": {"textbox", "searchbox", "checkbox", "radio", "spinbutton"},
            "select": {"combobox", "listbox"},
            "textarea": {"textbox"},
            "aria_button": {"button"},
            "aria_link": {"link"},
            "aria_checkbox": {"checkbox"},
            "aria_radio": {"radio"},
            "aria_tab": {"tab"},
            "aria_menuitem": {"menuitem"},
        }

        expected_roles = role_map.get(element.element_type, set())
        return node.role in expected_roles

    def _element_to_a11y_role(self, element: InteractiveElement) -> str | None:
        """Convert element type to expected a11y role."""
        type_to_role = {
            "button": "button",
            "a": "link",
            "aria_button": "button",
            "aria_link": "link",
            "aria_checkbox": "checkbox",
            "aria_radio": "radio",
            "aria_tab": "tab",
            "aria_menuitem": "menuitem",
        }
        return type_to_role.get(element.element_type)

    def _fuzzy_name_match(self, element: InteractiveElement, node: A11yNode) -> bool:
        """Check if element and node names match fuzzily."""
        elem_text = (element.text or element.aria_label or "").lower().strip()
        node_name = node.name.lower().strip()

        if not elem_text or not node_name:
            return False

        # Check containment
        return elem_text in node_name or node_name in elem_text

    async def enrich_elements(
        self,
        elements: list[InteractiveElement],
        page: Page,
    ) -> list[EnrichedElement]:
        """
        Enrich DOM elements with accessibility data.

        Args:
            elements: List of InteractiveElement from DOM extraction
            page: Playwright Page for a11y extraction

        Returns:
            List of EnrichedElement with merged data
        """
        tree = await self.extract_tree(page)
        enriched: list[EnrichedElement] = []

        for element in elements:
            a11y_node, confidence = self.match_element_to_a11y(element, tree)

            enriched_elem = EnrichedElement(
                element=element,
                match_confidence=confidence,
            )

            if a11y_node:
                enriched_elem.a11y_role = a11y_node.role
                enriched_elem.a11y_name = a11y_node.name
                enriched_elem.a11y_description = a11y_node.description
                enriched_elem.a11y_value = a11y_node.value
                enriched_elem.a11y_checked = a11y_node.checked
                enriched_elem.a11y_disabled = a11y_node.disabled
                enriched_elem.a11y_expanded = a11y_node.expanded
                enriched_elem.a11y_pressed = a11y_node.pressed
                enriched_elem.a11y_selected = a11y_node.selected

            enriched.append(enriched_elem)

        matched_count = sum(1 for e in enriched if e.match_confidence > 0)
        logger.info(
            f"Enriched {len(elements)} elements with a11y data " f"({matched_count} matched)"
        )

        return enriched


async def extract_accessibility_tree(page: Page) -> A11yTree:
    """
    Convenience function to extract accessibility tree.

    Args:
        page: Playwright Page

    Returns:
        A11yTree for the page
    """
    extractor = AccessibilityExtractor()
    return await extractor.extract_tree(page)


async def enrich_with_accessibility(
    elements: list[InteractiveElement],
    page: Page,
) -> list[EnrichedElement]:
    """
    Convenience function to enrich elements with accessibility data.

    Args:
        elements: List of InteractiveElement
        page: Playwright Page

    Returns:
        List of EnrichedElement with merged a11y data
    """
    extractor = AccessibilityExtractor()
    return await extractor.enrich_elements(elements, page)


def a11y_tree_to_text(tree: A11yTree) -> str:
    """
    Convert accessibility tree to LLM-friendly text representation.

    This format is similar to what Stagehand sends to LLMs for element
    understanding.

    Args:
        tree: The accessibility tree

    Returns:
        Indented text representation of the tree
    """
    return tree.to_text()
