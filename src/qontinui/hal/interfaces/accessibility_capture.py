"""Accessibility capture interface definition.

This module defines the interface for capturing and interacting with
accessibility trees from various sources (browsers via CDP, Windows
apps via UIA, etc.).
"""

from abc import ABC, abstractmethod

from qontinui_schemas.accessibility import (
    AccessibilityCaptureOptions,
    AccessibilityNode,
    AccessibilitySelector,
    AccessibilitySnapshot,
)


class IAccessibilityCapture(ABC):
    """Interface for accessibility tree capture and interaction.

    This interface provides methods for:
    - Connecting to accessibility sources (browsers, native apps)
    - Capturing accessibility tree snapshots
    - Finding and interacting with elements by ref or selector
    - Generating AI-friendly context from tree snapshots

    The ref system (@e1, @e2, etc.) provides stable identifiers for
    elements that can be used for AI-driven automation without
    re-querying the accessibility tree.

    Example:
        >>> capture = CDPAccessibilityCapture()
        >>> await capture.connect(cdp_port=9222)
        >>> snapshot = await capture.capture_tree()
        >>> await capture.click_by_ref("@e3")
        >>> await capture.type_by_ref("@e2", "hello@example.com")
    """

    @abstractmethod
    async def connect(
        self,
        target: str | int | None = None,
        *,
        host: str = "localhost",
        port: int = 9222,
        timeout: float = 30.0,
    ) -> bool:
        """Connect to an accessibility source.

        Args:
            target: Target identifier (URL, window title, handle, or None for auto)
            host: Host for network-based connections (CDP)
            port: Port for network-based connections
            timeout: Connection timeout in seconds

        Returns:
            True if connection succeeded, False otherwise
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the current accessibility source.

        Should be called to release resources when done capturing.
        Safe to call multiple times.
        """
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to an accessibility source.

        Returns:
            True if currently connected
        """
        ...

    @abstractmethod
    async def capture_tree(
        self,
        options: AccessibilityCaptureOptions | None = None,
    ) -> AccessibilitySnapshot:
        """Capture the accessibility tree.

        Captures the full accessibility tree from the connected source,
        assigns refs to all nodes, and returns a snapshot.

        Args:
            options: Capture options (filters, limits, etc.)

        Returns:
            AccessibilitySnapshot with the captured tree

        Raises:
            RuntimeError: If not connected to a source
        """
        ...

    @abstractmethod
    async def get_node_by_ref(self, ref: str) -> AccessibilityNode | None:
        """Get a node by its ref ID.

        Uses the most recent snapshot to look up a node by ref.

        Args:
            ref: Reference ID (e.g., "@e1", "@e5")

        Returns:
            The node if found, None otherwise
        """
        ...

    @abstractmethod
    async def find_nodes(
        self,
        selector: AccessibilitySelector,
    ) -> list[AccessibilityNode]:
        """Find nodes matching a selector.

        Searches the most recent snapshot for nodes matching the selector.

        Args:
            selector: Selection criteria

        Returns:
            List of matching nodes (may be empty)
        """
        ...

    @abstractmethod
    async def click_by_ref(self, ref: str) -> bool:
        """Click an element by ref.

        Performs a click action on the element with the given ref.
        Uses the element's center coordinates from its bounds.

        Args:
            ref: Reference ID of element to click

        Returns:
            True if click succeeded, False if element not found

        Raises:
            RuntimeError: If element has no bounds
        """
        ...

    @abstractmethod
    async def type_by_ref(
        self,
        ref: str,
        text: str,
        *,
        clear_first: bool = False,
    ) -> bool:
        """Type text into an element by ref.

        Focuses the element (if not already focused) and types text.
        Optionally clears existing content first.

        Args:
            ref: Reference ID of element to type into
            text: Text to type
            clear_first: If True, clear existing content before typing

        Returns:
            True if typing succeeded, False if element not found

        Raises:
            RuntimeError: If element is not editable
        """
        ...

    @abstractmethod
    async def focus_by_ref(self, ref: str) -> bool:
        """Focus an element by ref.

        Moves keyboard focus to the element with the given ref.

        Args:
            ref: Reference ID of element to focus

        Returns:
            True if focus succeeded, False if element not found

        Raises:
            RuntimeError: If element is not focusable
        """
        ...

    @abstractmethod
    def get_backend_name(self) -> str:
        """Get the name of this backend implementation.

        Returns:
            Backend name (e.g., "cdp", "uia", "atspi")
        """
        ...

    def to_ai_context(
        self,
        snapshot: AccessibilitySnapshot,
        *,
        max_elements: int = 100,
        interactive_only: bool = True,
    ) -> str:
        """Generate AI-friendly text representation of the tree.

        Creates a concise text format suitable for including in AI
        prompts. Lists interactive elements with their refs, roles,
        and names.

        Args:
            snapshot: The snapshot to convert
            max_elements: Maximum number of elements to include
            interactive_only: If True, only include interactive elements

        Returns:
            AI-friendly text representation

        Example output:
            ## Accessibility Tree
            URL: https://example.com/login
            Title: Login Page

            ### Interactive Elements
            - @e1: button "Sign In"
            - @e2: textbox "Email"
            - @e3: textbox "Password"
            - @e4: link "Forgot Password?"
        """
        lines: list[str] = []
        lines.append("## Accessibility Tree")

        if snapshot.url:
            lines.append(f"URL: {snapshot.url}")
        if snapshot.title:
            lines.append(f"Title: {snapshot.title}")

        lines.append("")
        lines.append("### Interactive Elements")

        element_count = 0

        def walk(node: AccessibilityNode, depth: int = 0) -> None:
            nonlocal element_count
            if element_count >= max_elements:
                return

            # Skip non-interactive if filter is enabled
            if interactive_only and not node.is_interactive:
                for child in node.children:
                    walk(child, depth)
                return

            # Format element entry
            name_str = f' "{node.name}"' if node.name else ""
            value_str = f" = {node.value}" if node.value else ""
            state_strs: list[str] = []

            if node.state.is_disabled:
                state_strs.append("disabled")
            if node.state.is_focused:
                state_strs.append("focused")
            if node.state.is_checked is True:
                state_strs.append("checked")
            if node.state.is_expanded is True:
                state_strs.append("expanded")

            state_suffix = f" [{', '.join(state_strs)}]" if state_strs else ""
            lines.append(
                f"- {node.ref}: {node.role.value}{name_str}{value_str}{state_suffix}"
            )
            element_count += 1

            # Recurse into children
            for child in node.children:
                walk(child, depth + 1)

        walk(snapshot.root)

        if element_count >= max_elements:
            lines.append(f"... (truncated, {snapshot.total_nodes} total)")

        return "\n".join(lines)
