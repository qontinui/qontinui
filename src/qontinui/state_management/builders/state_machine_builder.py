"""
State Machine Builder for Web Extraction.

This module implements the algorithm to create a State Machine from web extraction results.
The key insight is that states are defined by image co-occurrence across screens:

- States = clusters of images that appear together across screens
- Transitions = actions that change which states are visible

Example:
    Screen 1: images a, b, c, d -> states {state1: a,b}, {state2: c,d}
    Screen 2: images a, b, c, d -> states {state1: a,b}, {state2: c,d}
    Screen 3: images a, b, e    -> states {state1: a,b}, {state3: e}

    Transition: click on c (from state2) navigates from screen 2 to screen 3
                This removes state2 and makes state3 active.

Usage:
    from qontinui.state_management.builders import build_state_machine_from_extraction

    states, transitions = build_state_machine_from_extraction(
        annotations,  # List of extraction annotation dicts
        transitions   # Optional list of inferred transition dicts
    )
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class ExtractedImage:
    """Represents an image/element extracted from a screenshot."""

    id: str
    element_id: str  # Original element ID from extraction
    name: str
    text: str | None
    element_type: str
    bbox: dict[str, float]  # {x, y, width, height}
    screenshot_id: str
    source_url: str
    selector: str | None = None
    signature: str = ""  # Computed signature for matching
    extraction_category: str = ""  # Why this element was extracted (debugging)

    def __post_init__(self):
        """Compute signature for matching elements across screens."""
        # Signature is based on element type + text content + relative dimensions
        # We use relative position (normalized) to handle different viewport sizes
        text_sig = (self.text or "").strip().lower()[:50]  # First 50 chars
        type_sig = self.element_type.lower()
        # Normalize dimensions to ratio for viewport-independent matching
        width = self.bbox.get("width", 0)
        height = self.bbox.get("height", 0)
        ratio = round(width / max(height, 1), 2) if height > 0 else 0
        self.signature = f"{type_sig}|{text_sig}|{ratio}"


@dataclass
class ScreenInfo:
    """Information about a screen/page from extraction."""

    screenshot_id: str
    source_url: str
    images: list[ExtractedImage] = field(default_factory=list)
    image_signatures: set[str] = field(default_factory=set)


@dataclass
class StateMachineState:
    """A state in the state machine, derived from image clustering."""

    id: str
    name: str
    description: str
    image_signatures: frozenset[str]  # The signature cluster defining this state
    screen_ids: set[str]  # Screens where this state appears
    images: list[ExtractedImage] = field(default_factory=list)  # Representative images
    # Track all instances of each image across screens for position analysis
    all_image_instances: dict[str, list[ExtractedImage]] = field(default_factory=dict)

    def to_config(self, graph_position: tuple[int, int] = (0, 0)) -> dict[str, Any]:
        """Convert to state machine configuration format.

        Only generates stateImages from extracted elements.
        - Sets `fixed: true` when the image position is consistent across screens
        - Populates `searchRegions` with the bounding box where the image was found

        regions, locations, and strings are left empty for user definition:
        - regions: Semantic areas (e.g., where an island name appears)
        - locations: Click points for variable-appearance elements in fixed positions
        - strings: Text with special meaning (e.g., "i" for inventory)
        """
        state_images = []

        for img in self.images:
            # Check if this image has a fixed position across all screens
            # by looking at all instances with the same signature
            is_fixed = False
            search_regions = []

            instances = self.all_image_instances.get(img.signature, [img])

            if len(instances) > 0:
                # Check if position is consistent across all instances
                first_bbox = instances[0].bbox
                position_consistent = all(
                    abs(inst.bbox.get("x", 0) - first_bbox.get("x", 0)) < 10
                    and abs(inst.bbox.get("y", 0) - first_bbox.get("y", 0)) < 10
                    for inst in instances
                )
                is_fixed = position_consistent

                # Create search region from the bounding box
                # Use the first instance's bbox as the search region
                bbox = first_bbox
                if bbox.get("width", 0) > 0 and bbox.get("height", 0) > 0:
                    search_regions.append(
                        {
                            "id": str(uuid4()),
                            "name": f"Search region for {img.name or img.text or img.element_type}",
                            "x": int(bbox.get("x", 0)),
                            "y": int(bbox.get("y", 0)),
                            "width": int(bbox.get("width", 1)),
                            "height": int(bbox.get("height", 1)),
                            "offsetX": 0,
                            "offsetY": 0,
                        }
                    )

            state_image = {
                "id": str(uuid4()),
                "name": img.name or img.text or f"Image from {img.element_type}",
                "patterns": [
                    {
                        "id": str(uuid4()),
                        "name": "Default pattern",
                        "imageId": None,  # Will be populated when user uploads/captures image
                        "searchRegions": search_regions,
                        "fixed": is_fixed,
                        "similarity": 0.8,
                    }
                ],
                "shared": False,
                "source": "web-extraction",
                "searchMode": "default",
                "searchRegions": search_regions,  # Also at StateImage level
                "screenshotId": img.screenshot_id,
                "sourceUrl": img.source_url,
                "extractionCategory": img.extraction_category,  # Debugging info
            }
            state_images.append(state_image)

        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "stateImages": state_images,
            "regions": [],  # User-defined semantic areas
            "locations": [],  # User-defined click points
            "strings": [],  # User-defined text with special meaning
            "position": {"x": graph_position[0], "y": graph_position[1]},
            "initial": False,
            "isFinal": False,
        }


@dataclass
class StateMachineTransition:
    """A transition between states, derived from navigation actions."""

    id: str
    from_state_id: str
    to_state_id: str
    trigger_element: ExtractedImage | None
    source_url: str
    target_url: str
    trigger_type: str = "click"

    def to_config(self) -> dict[str, Any]:
        """Convert to transition configuration format."""
        name = "Navigation"
        if self.trigger_element:
            name = self.trigger_element.text or self.trigger_element.name or "Click"

        return {
            "id": self.id,
            "type": "OutgoingTransition",
            "name": f"Transition: {name}",
            "description": f"Navigate from {self.source_url} to {self.target_url}",
            "fromState": self.from_state_id,
            "toState": self.to_state_id,
            "workflows": [],  # User can add workflows later
            "timeout": 10000,
            "retryCount": 3,
            "staysVisible": False,
            "activateStates": [self.to_state_id] if self.to_state_id else [],
            "deactivateStates": [self.from_state_id],
        }


class StateMachineBuilder:
    """
    Builds a State Machine from web extraction results.

    The algorithm:
    1. Extract all elements/images from all screenshots
    2. Compute signatures for each element (based on type, text, dimensions)
    3. Build co-occurrence matrix (which signatures appear on which screens)
    4. Cluster signatures by which screens they appear on
    5. Each unique cluster = one State
    6. Derive transitions from navigation actions (InferredTransitions)

    Example:
        >>> builder = StateMachineBuilder(annotations, transitions)
        >>> states_config, transitions_config = builder.build()
    """

    def __init__(
        self,
        annotations: list[dict[str, Any]],
        transitions: list[dict[str, Any]] | None = None,
    ):
        """
        Initialize the builder with extraction data.

        Args:
            annotations: List of ExtractionAnnotation dicts from the extraction session.
                Each annotation should have:
                - screenshot_id: str
                - source_url: str
                - elements: list of element dicts with id, elementType, text, bbox
            transitions: List of InferredTransition dicts (optional).
                Each transition should have:
                - sourceUrl, targetUrl
                - triggerType, triggerText, triggerSelector
        """
        self.annotations = annotations
        self.raw_transitions = transitions or []
        self.screens: dict[str, ScreenInfo] = {}
        self.all_images: list[ExtractedImage] = []
        self.signature_to_images: dict[str, list[ExtractedImage]] = defaultdict(list)
        self.signature_to_screens: dict[str, set[str]] = defaultdict(set)
        self.states: list[StateMachineState] = []
        self.transitions: list[StateMachineTransition] = []

    def build(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Build the state machine from extraction data.

        Returns:
            Tuple of (states_config, transitions_config) ready for project configuration
        """
        logger.info(
            "Starting state machine build with %d annotations and %d transitions",
            len(self.annotations),
            len(self.raw_transitions),
        )

        # Step 1: Extract all images/elements from annotations
        self._extract_images()

        # Step 2: Build signature→screens mapping
        self._build_cooccurrence_map()

        # Step 3: Cluster signatures into states
        self._cluster_into_states()

        # Step 4: Derive transitions
        self._derive_transitions()

        # Step 5: Generate configuration
        states_config = self._generate_states_config()
        transitions_config = self._generate_transitions_config()

        logger.info(
            "State machine build complete: %d states, %d transitions",
            len(states_config),
            len(transitions_config),
        )

        return states_config, transitions_config

    def _extract_images(self) -> None:
        """Extract all images/elements from annotations."""
        for annotation in self.annotations:
            screenshot_id = annotation.get("screenshot_id", annotation.get("screenshotId", ""))
            source_url = annotation.get("source_url", annotation.get("sourceUrl", ""))

            screen = ScreenInfo(screenshot_id=screenshot_id, source_url=source_url)
            self.screens[screenshot_id] = screen

            elements = annotation.get("elements", [])
            for elem in elements:
                # Parse bbox - handle both dict and nested formats
                bbox_raw = elem.get("bbox", {})
                if isinstance(bbox_raw, dict):
                    bbox = {
                        "x": float(bbox_raw.get("x", 0)),
                        "y": float(bbox_raw.get("y", 0)),
                        "width": float(bbox_raw.get("width", 0)),
                        "height": float(bbox_raw.get("height", 0)),
                    }
                else:
                    bbox = {"x": 0, "y": 0, "width": 0, "height": 0}

                image = ExtractedImage(
                    id=str(uuid4()),
                    element_id=elem.get("id", str(uuid4())),
                    name=elem.get("name") or elem.get("text") or elem.get("elementType", "unknown"),
                    text=elem.get("text"),
                    element_type=elem.get("element_type", elem.get("elementType", "unknown")),
                    bbox=bbox,
                    screenshot_id=screenshot_id,
                    source_url=source_url,
                    selector=elem.get("selector"),
                    extraction_category=elem.get(
                        "extraction_category", elem.get("extractionCategory", "")
                    ),
                )

                self.all_images.append(image)
                screen.images.append(image)
                screen.image_signatures.add(image.signature)
                self.signature_to_images[image.signature].append(image)

        logger.debug(
            "Extracted %d images with %d unique signatures from %d screens",
            len(self.all_images),
            len(self.signature_to_images),
            len(self.screens),
        )

    def _build_cooccurrence_map(self) -> None:
        """Build mapping from signature to screens where it appears."""
        for screen_id, screen in self.screens.items():
            for sig in screen.image_signatures:
                self.signature_to_screens[sig].add(screen_id)

        logger.debug(
            "Built co-occurrence map with %d signatures",
            len(self.signature_to_screens),
        )

    def _cluster_into_states(self) -> None:
        """
        Cluster signatures into states based on co-occurrence.

        Signatures that appear on the exact same set of screens belong to the same state.
        """
        # Group signatures by their screen set
        screen_set_to_signatures: dict[frozenset[str], set[str]] = defaultdict(set)

        for sig, screens in self.signature_to_screens.items():
            screen_set = frozenset(screens)
            screen_set_to_signatures[screen_set].add(sig)

        # Create states from clusters
        state_index = 1
        for screen_set, signatures in screen_set_to_signatures.items():
            if len(signatures) == 0:
                continue

            # Get representative images for this state and collect all instances
            representative_images = []
            all_image_instances: dict[str, list[ExtractedImage]] = {}

            for sig in signatures:
                if sig in self.signature_to_images:
                    # Take the first image as representative
                    representative_images.append(self.signature_to_images[sig][0])
                    # Store all instances for position analysis
                    all_image_instances[sig] = self.signature_to_images[sig]

            # Generate state name based on screen URLs
            screen_urls = set()
            for screen_id in screen_set:
                if screen_id in self.screens:
                    url = self.screens[screen_id].source_url
                    # Extract path from URL for naming
                    path = url.split("/")[-1] or "root"
                    screen_urls.add(path)

            state_name = f"State {state_index}: {', '.join(list(screen_urls)[:3])}"
            if len(screen_urls) > 3:
                state_name += f" (+{len(screen_urls) - 3} more)"

            state = StateMachineState(
                id=str(uuid4()),
                name=state_name,
                description=f"State containing {len(signatures)} elements appearing on {len(screen_set)} screens",
                image_signatures=frozenset(signatures),
                screen_ids=set(screen_set),
                images=representative_images,
                all_image_instances=all_image_instances,
            )
            self.states.append(state)
            state_index += 1

        logger.debug(
            "Clustered into %d states from %d screen sets",
            len(self.states),
            len(screen_set_to_signatures),
        )

    def _derive_transitions(self) -> None:
        """
        Derive transitions from navigation actions.

        For each InferredTransition, determine which states change:
        - Compare states visible on source screen vs target screen
        - The state containing the trigger element is the from_state
        - The state that becomes newly visible is the to_state
        """
        if not self.raw_transitions:
            logger.debug("No transitions to derive")
            return

        # Build screen→states mapping
        screen_to_states: dict[str, list[StateMachineState]] = defaultdict(list)
        for state in self.states:
            for screen_id in state.screen_ids:
                screen_to_states[screen_id].append(state)

        for trans in self.raw_transitions:
            source_url = trans.get("source_url", trans.get("sourceUrl", ""))
            target_url = trans.get("target_url", trans.get("targetUrl", ""))
            trigger_selector = trans.get("trigger_selector", trans.get("triggerSelector"))
            trigger_text = trans.get("trigger_text", trans.get("triggerText"))

            # Find source and target screens
            source_screen_id = None
            target_screen_id = None
            for screen_id, screen in self.screens.items():
                if screen.source_url == source_url:
                    source_screen_id = screen_id
                if screen.source_url == target_url:
                    target_screen_id = screen_id

            if not source_screen_id or not target_screen_id:
                continue

            # Get states visible on each screen
            source_states = {s.id for s in screen_to_states.get(source_screen_id, [])}
            target_states = {s.id for s in screen_to_states.get(target_screen_id, [])}

            # States that disappear = only on source, not on target
            disappearing = source_states - target_states
            # States that appear = only on target, not on source
            appearing = target_states - source_states

            if not disappearing or not appearing:
                continue

            # Find the trigger element
            trigger_element = None
            for img in self.all_images:
                if img.screenshot_id == source_screen_id:
                    if trigger_text and img.text and trigger_text in img.text:
                        trigger_element = img
                        break
                    if trigger_selector and img.selector == trigger_selector:
                        trigger_element = img
                        break

            # Find the state that the trigger element belongs to
            from_state_id = None
            if trigger_element:
                for state in self.states:
                    if trigger_element.signature in state.image_signatures:
                        from_state_id = state.id
                        break

            # If we couldn't find from_state from trigger, use any disappearing state
            if not from_state_id and disappearing:
                from_state_id = list(disappearing)[0]

            # To state is any appearing state
            to_state_id = list(appearing)[0] if appearing else None

            if from_state_id and to_state_id:
                transition = StateMachineTransition(
                    id=str(uuid4()),
                    from_state_id=from_state_id,
                    to_state_id=to_state_id,
                    trigger_element=trigger_element,
                    source_url=source_url,
                    target_url=target_url,
                    trigger_type=trans.get("trigger_type", trans.get("triggerType", "click")),
                )
                self.transitions.append(transition)

        logger.debug(
            "Derived %d transitions from %d raw transitions",
            len(self.transitions),
            len(self.raw_transitions),
        )

    def _generate_states_config(self) -> list[dict[str, Any]]:
        """Generate state configurations with graph layout."""
        configs = []
        # Simple grid layout for graph visualization
        cols = max(3, int(len(self.states) ** 0.5))
        for i, state in enumerate(self.states):
            row = i // cols
            col = i % cols
            x = col * 250 + 100
            y = row * 200 + 100
            configs.append(state.to_config(graph_position=(x, y)))
        return configs

    def _generate_transitions_config(self) -> list[dict[str, Any]]:
        """Generate transition configurations."""
        return [t.to_config() for t in self.transitions]


def build_state_machine_from_extraction(
    annotations: list[dict[str, Any]],
    transitions: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Build a state machine from web extraction results.

    This is the main entry point for creating a state machine from extraction data.
    It implements the co-occurrence clustering algorithm to identify states and
    derives transitions from navigation actions.

    The algorithm groups images by co-occurrence across screens:
    - Images that appear together on the same set of screens → same state
    - Navigation actions that change visible states → transitions

    Args:
        annotations: List of ExtractionAnnotation dicts from the extraction session.
            Each annotation should contain:
            - screenshot_id (or screenshotId): Unique ID for the screenshot
            - source_url (or sourceUrl): URL of the page
            - elements: List of element dicts with id, elementType, text, bbox

        transitions: List of InferredTransition dicts (optional).
            Each transition should contain:
            - source_url (or sourceUrl): URL of the source page
            - target_url (or targetUrl): URL of the target page
            - trigger_type (or triggerType): Type of trigger (click, etc.)
            - trigger_text (or triggerText): Text of the trigger element

    Returns:
        Tuple of (states_config, transitions_config) ready for project configuration.
        - states_config: List of State dicts with stateImages, empty regions/locations/strings
        - transitions_config: List of OutgoingTransition dicts

    Example:
        >>> from qontinui.state_management.builders import build_state_machine_from_extraction
        >>>
        >>> annotations = [
        ...     {
        ...         "screenshot_id": "screen1",
        ...         "source_url": "http://example.com/page1",
        ...         "elements": [
        ...             {"id": "a", "elementType": "button", "text": "Click", "bbox": {...}},
        ...         ]
        ...     }
        ... ]
        >>> states, transitions = build_state_machine_from_extraction(annotations)
        >>> project_config["states"] = states
        >>> project_config["transitions"] = transitions
    """
    builder = StateMachineBuilder(annotations, transitions)
    return builder.build()
