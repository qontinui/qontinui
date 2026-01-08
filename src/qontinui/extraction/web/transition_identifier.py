"""
Transition Identifier for function-based transition detection.

Identifies transitions from element function data (href, onclick, form actions).
Maps navigation targets to states to create the transition graph.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlparse

from .models import (
    ElementFunction,
    ExtractedPageV2,
    FunctionType,
    IdentifiedState,
    IdentifiedTransition,
    RawElement,
)
from .state_identifier import PageStateMapping

logger = logging.getLogger(__name__)


@dataclass
class TransitionCandidate:
    """A potential transition identified from element function."""

    element_id: str
    source_page_url: str
    source_screenshot_id: str
    target_url: str | None
    function_type: FunctionType
    trigger_type: str  # "click", "submit", "hover", etc.
    confidence: float
    element_function: ElementFunction
    metadata: dict[str, Any] = field(default_factory=dict)


class TransitionIdentifier:
    """
    Identifies transitions from element function data.

    Process:
    1. Extract ElementFunction from each RawElement
    2. Resolve target URLs for navigation functions
    3. Map target URLs to states
    4. Create transitions between states

    Supports:
    - Link navigation (href)
    - Form submission (form action)
    - JavaScript navigation (onclick with URL patterns)
    - ARIA controls (aria-controls, aria-expanded)
    """

    # Patterns for detecting navigation in onclick handlers
    ONCLICK_URL_PATTERNS = [
        r"window\.location\s*=\s*['\"]([^'\"]+)['\"]",
        r"location\.href\s*=\s*['\"]([^'\"]+)['\"]",
        r"navigate\(['\"]([^'\"]+)['\"]",
        r"router\.push\(['\"]([^'\"]+)['\"]",
        r"history\.pushState\([^,]+,\s*[^,]+,\s*['\"]([^'\"]+)['\"]",
    ]

    # Patterns for detecting modal/dialog triggers
    MODAL_PATTERNS = [
        r"modal",
        r"dialog",
        r"popup",
        r"overlay",
        r"drawer",
        r"sheet",
    ]

    # Patterns for detecting toggle/expand actions
    TOGGLE_PATTERNS = [
        r"toggle",
        r"expand",
        r"collapse",
        r"accordion",
        r"dropdown",
        r"menu",
    ]

    def __init__(
        self,
        base_url: str | None = None,
        include_same_page: bool = False,
    ) -> None:
        """
        Initialize the transition identifier.

        Args:
            base_url: Base URL for resolving relative URLs.
            include_same_page: Whether to include same-page transitions
                (anchor links, toggles, etc.).
        """
        self.base_url = base_url
        self.include_same_page = include_same_page

    def extract_element_functions(
        self,
        elements: list[RawElement],
        page_url: str,
    ) -> list[ElementFunction]:
        """
        Extract function data from elements.

        Args:
            elements: List of RawElement objects.
            page_url: URL of the page (for resolving relative URLs).

        Returns:
            List of ElementFunction objects.
        """
        functions: list[ElementFunction] = []

        for element in elements:
            func = self._analyze_element_function(element, page_url)
            if func:
                functions.append(func)

        logger.debug(f"Extracted {len(functions)} element functions")
        return functions

    def _analyze_element_function(
        self,
        element: RawElement,
        page_url: str,
    ) -> ElementFunction | None:
        """Analyze an element to determine its function."""
        # Check for navigation (href)
        if element.href:
            target_url = self._resolve_url(element.href, page_url)

            # Skip anchor-only links unless include_same_page
            if target_url and target_url.startswith("#"):
                if not self.include_same_page:
                    return None
                func_type = FunctionType.TOGGLE
            else:
                func_type = FunctionType.NAVIGATE

            return ElementFunction(
                element_id=element.id,
                function_type=func_type,
                href=element.href,
                target_url=target_url,
            )

        # Check for form submission
        if element.form_action:
            return ElementFunction(
                element_id=element.id,
                function_type=FunctionType.SUBMIT,
                form_action=element.form_action,
                form_method=element.form_method,
                target_url=self._resolve_url(element.form_action, page_url),
            )

        # Check for onclick handler
        if element.onclick:
            func = self._analyze_onclick(element, page_url)
            if func:
                return func

        # Check for ARIA controls
        if element.aria_controls:
            func_type = FunctionType.TOGGLE
            if element.aria_expanded:
                func_type = FunctionType.EXPAND

            return ElementFunction(
                element_id=element.id,
                function_type=func_type,
                aria_controls=element.aria_controls,
                aria_expanded=element.aria_expanded,
            )

        # Check data attributes for navigation hints
        for key, value in element.data_attributes.items():
            if "href" in key or "url" in key or "link" in key:
                return ElementFunction(
                    element_id=element.id,
                    function_type=FunctionType.NAVIGATE,
                    target_url=self._resolve_url(value, page_url),
                    metadata={"source": f"data-{key}"},
                )

            if "toggle" in key or "target" in key:
                return ElementFunction(
                    element_id=element.id,
                    function_type=FunctionType.TOGGLE,
                    aria_controls=value,
                    metadata={"source": f"data-{key}"},
                )

        return None

    def _analyze_onclick(
        self,
        element: RawElement,
        page_url: str,
    ) -> ElementFunction | None:
        """Analyze onclick handler for navigation patterns."""
        onclick = element.onclick.lower() if element.onclick else ""

        # Check for URL patterns
        for pattern in self.ONCLICK_URL_PATTERNS:
            match = re.search(pattern, onclick, re.IGNORECASE)
            if match:
                url = match.group(1)
                return ElementFunction(
                    element_id=element.id,
                    function_type=FunctionType.NAVIGATE,
                    onclick=element.onclick,
                    target_url=self._resolve_url(url, page_url),
                )

        # Check for modal patterns
        for pattern in self.MODAL_PATTERNS:
            if pattern in onclick:
                return ElementFunction(
                    element_id=element.id,
                    function_type=FunctionType.OPEN_MODAL,
                    onclick=element.onclick,
                )

        # Check for toggle patterns
        for pattern in self.TOGGLE_PATTERNS:
            if pattern in onclick:
                return ElementFunction(
                    element_id=element.id,
                    function_type=FunctionType.TOGGLE,
                    onclick=element.onclick,
                )

        # Unknown onclick - still record it
        if element.onclick:
            return ElementFunction(
                element_id=element.id,
                function_type=FunctionType.UNKNOWN,
                onclick=element.onclick,
            )

        return None

    def _resolve_url(self, url: str, base_url: str) -> str | None:
        """Resolve a URL relative to the base URL."""
        if not url:
            return None

        # Handle anchor links
        if url.startswith("#"):
            return url

        # Handle javascript: URLs
        if url.startswith("javascript:"):
            return None

        # Handle mailto: and tel: URLs
        if url.startswith(("mailto:", "tel:")):
            return None

        try:
            # Use base_url if provided, otherwise page_url
            base = self.base_url or base_url
            return urljoin(base, url)
        except Exception:
            return url

    def identify_transitions(
        self,
        pages: list[ExtractedPageV2],
        states: list[IdentifiedState],
        page_mappings: list[PageStateMapping],
    ) -> list[IdentifiedTransition]:
        """
        Identify transitions between states.

        Args:
            pages: Extracted pages with elements.
            states: Identified states.
            page_mappings: Mapping of pages to states.

        Returns:
            List of IdentifiedTransition objects.
        """
        if not pages or not states:
            return []

        logger.info(f"Identifying transitions from {len(pages)} pages...")

        # Build lookup maps
        url_to_state = self._build_url_state_map(page_mappings)
        screenshot_to_state = {pm.screenshot_id: pm.state_id for pm in page_mappings}

        # Find transition candidates
        candidates: list[TransitionCandidate] = []

        for page in pages:
            # Get element functions for this page
            functions = self.extract_element_functions(page.elements, page.url)
            page.element_functions = functions

            # Get source state
            source_state_id = screenshot_to_state.get(page.screenshot_id)
            if not source_state_id:
                continue

            # Check each function for potential transitions
            for func in functions:
                candidate = self._evaluate_transition_candidate(
                    func,
                    page,
                    source_state_id,
                    url_to_state,
                )
                if candidate:
                    candidates.append(candidate)

        # Convert candidates to transitions
        transitions = self._create_transitions(candidates, states)

        logger.info(f"Identified {len(transitions)} transitions")
        return transitions

    def _build_url_state_map(
        self,
        page_mappings: list[PageStateMapping],
    ) -> dict[str, str]:
        """Build mapping from URL to state ID."""
        url_to_state: dict[str, str] = {}

        for pm in page_mappings:
            # Normalize URL
            normalized = self._normalize_url(pm.page_url)
            url_to_state[normalized] = pm.state_id

            # Also store without query string
            parsed = urlparse(pm.page_url)
            base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            url_to_state[self._normalize_url(base_url)] = pm.state_id

        return url_to_state

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for comparison."""
        if not url:
            return ""

        # Remove trailing slash
        url = url.rstrip("/")

        # Remove common tracking parameters
        try:
            parsed = urlparse(url)
            # Keep only path for internal comparison
            return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        except Exception:
            return url

    def _evaluate_transition_candidate(
        self,
        func: ElementFunction,
        page: ExtractedPageV2,
        source_state_id: str,
        url_to_state: dict[str, str],
    ) -> TransitionCandidate | None:
        """Evaluate an element function as a transition candidate."""
        # Only navigation and submit functions create cross-state transitions
        if func.function_type not in (
            FunctionType.NAVIGATE,
            FunctionType.SUBMIT,
        ):
            return None

        target_url = func.target_url
        if not target_url:
            return None

        # Skip anchor-only links
        if target_url.startswith("#"):
            return None

        # Determine trigger type
        trigger_type = "click"
        if func.function_type == FunctionType.SUBMIT:
            trigger_type = "submit"

        # Calculate confidence based on evidence
        confidence = 0.8  # Default for href
        if func.form_action:
            confidence = 0.9  # Forms are reliable
        if func.onclick:
            confidence = 0.6  # onclick patterns are less reliable

        return TransitionCandidate(
            element_id=func.element_id,
            source_page_url=page.url,
            source_screenshot_id=page.screenshot_id,
            target_url=target_url,
            function_type=func.function_type,
            trigger_type=trigger_type,
            confidence=confidence,
            element_function=func,
        )

    def _create_transitions(
        self,
        candidates: list[TransitionCandidate],
        states: list[IdentifiedState],
    ) -> list[IdentifiedTransition]:
        """Convert transition candidates to IdentifiedTransition objects."""
        transitions: list[IdentifiedTransition] = []
        seen_transitions: set[tuple[str, str, str]] = set()

        # Build state lookup by various identifiers
        state_by_url = self._build_state_url_lookup(states)

        for candidate in candidates:
            # Find target state
            target_state_id = self._resolve_target_state(
                candidate.target_url,
                state_by_url,
            )

            if not target_state_id:
                logger.debug(f"No state found for target URL: {candidate.target_url}")
                continue

            # Find source state
            source_state_id = self._resolve_source_state(
                candidate.source_screenshot_id,
                states,
            )

            if not source_state_id:
                continue

            # Skip self-loops unless explicitly requested
            if source_state_id == target_state_id:
                continue

            # Create unique transition key
            trans_key = (source_state_id, target_state_id, candidate.element_id)
            if trans_key in seen_transitions:
                continue
            seen_transitions.add(trans_key)

            # Create transition
            transition = IdentifiedTransition(
                id=f"trans_{len(transitions):04d}",
                from_state_id=source_state_id,
                to_state_id=target_state_id,
                trigger_element_id=candidate.element_id,
                trigger_type=candidate.trigger_type,
                element_function=candidate.element_function,
                source_url=candidate.source_page_url,
                target_url=candidate.target_url or "",
            )
            transitions.append(transition)

        return transitions

    def _build_state_url_lookup(
        self,
        states: list[IdentifiedState],
    ) -> dict[str, str]:
        """Build lookup from URL to state ID."""
        lookup: dict[str, str] = {}

        for state in states:
            for url in state.page_urls:
                normalized = self._normalize_url(url)
                lookup[normalized] = state.id

                # Also add path-only version
                try:
                    parsed = urlparse(url)
                    lookup[parsed.path] = state.id
                except Exception:
                    pass

        return lookup

    def _resolve_target_state(
        self,
        target_url: str | None,
        state_by_url: dict[str, str],
    ) -> str | None:
        """Resolve target URL to a state ID."""
        if not target_url:
            return None

        # Try exact match
        normalized = self._normalize_url(target_url)
        if normalized in state_by_url:
            return state_by_url[normalized]

        # Try path only
        try:
            parsed = urlparse(target_url)
            if parsed.path in state_by_url:
                return state_by_url[parsed.path]
        except Exception:
            pass

        return None

    def _resolve_source_state(
        self,
        screenshot_id: str,
        states: list[IdentifiedState],
    ) -> str | None:
        """Find state containing the given screenshot."""
        for state in states:
            if screenshot_id in state.screenshot_ids:
                return state.id
        return None


def identify_transitions(
    pages: list[ExtractedPageV2],
    states: list[IdentifiedState],
    page_mappings: list[PageStateMapping],
    base_url: str | None = None,
) -> list[IdentifiedTransition]:
    """
    Convenience function to identify transitions.

    Args:
        pages: Extracted pages.
        states: Identified states.
        page_mappings: Page to state mappings.
        base_url: Optional base URL.

    Returns:
        List of transitions.
    """
    identifier = TransitionIdentifier(base_url)
    return identifier.identify_transitions(pages, states, page_mappings)
