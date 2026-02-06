"""
State Machine Builder for Web Extraction.

This module implements the algorithm to create a State Machine from web extraction results.
The key insight is that states are defined by image co-occurrence across screens:

- States = clusters of images that appear together across screens
- Transitions = actions that change which states are visible

The algorithm:
1. Extract elements from each screenshot with their bounding boxes
2. For each element, crop its image region from the source screenshot
3. Search for that image on ALL other screenshots using template matching
4. Record which screenshots contain each image
5. Group images by co-occurrence (same set of screenshots = same state)

Example:
    Screen 1: images a, b, c, d
    Screen 2: images a, b, c, d
    Screen 3: images a, b, e

    After cross-screenshot matching:
    Image    Screens
    a        1,2,3
    b        1,2,3
    c        1,2
    d        1,2
    e        3

    States (grouped by screen set):
    State 1: {a, b} - appears on screens 1,2,3
    State 2: {c, d} - appears on screens 1,2
    State 3: {e}    - appears on screen 3

Usage:
    from qontinui.state_management.builders import build_state_machine_from_extraction_result

    states, transitions = build_state_machine_from_extraction_result(
        extraction_result,  # ExtractionResult from orchestrator
        screenshots_dir     # Path to screenshots directory
    )
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

# Note: cv2 and numpy are imported lazily inside ImageMatchingStateMachineBuilder
# to avoid import errors when cv2 DLLs are not available

if TYPE_CHECKING:
    from qontinui.extraction.models import ExtractionResult

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
            "screensFound": list(self.screen_ids),  # Which screenshots this state appears on
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


@dataclass
class ImageMatch:
    """Result of searching for an image on a screenshot."""

    screenshot_id: str
    found: bool
    bbox: dict[str, float] | None = None  # Where it was found (may differ from original)
    confidence: float = 0.0


@dataclass
class TrackedImage:
    """An image tracked across multiple screenshots."""

    id: str
    name: str
    source_screenshot_id: str  # Screenshot where this image was first found
    source_bbox: dict[str, float]  # Original bounding box
    image_data: Any = None  # Cropped image data (np.ndarray, lazy import)
    screens_found: set[str] = field(default_factory=set)  # Screenshots where this image appears
    matches: dict[str, ImageMatch] = field(default_factory=dict)  # screenshot_id -> match result
    element_type: str = "unknown"
    text: str | None = None
    selector: str | None = None
    extraction_category: str = ""


class ImageMatchingStateMachineBuilder:
    """
    Builds a State Machine using actual image matching across screenshots.

    Unlike the signature-based approach, this:
    1. Crops actual image regions from screenshots
    2. Uses template matching to find images on other screenshots
    3. Only records presence when visually confirmed
    4. Groups by actual visual co-occurrence
    """

    def __init__(
        self,
        screenshots_dir: Path,
        similarity_threshold: float = 0.7,
    ):
        """
        Initialize the builder.

        Args:
            screenshots_dir: Directory containing screenshot images
            similarity_threshold: Minimum similarity score for template matching (0-1)
        """
        # Lazy import cv2 and numpy to avoid import errors when DLLs are not available
        import cv2
        import numpy as np

        self._cv2 = cv2
        self._np = np

        self.screenshots_dir = Path(screenshots_dir)
        self.similarity_threshold = similarity_threshold
        self.screenshots: dict[str, Any] = {}  # screenshot_id -> image array (np.ndarray)
        self.candidate_elements: dict[str, list[dict[str, Any]]] = (
            {}
        )  # screenshot_id -> list of elements
        self.tracked_images: list[TrackedImage] = []
        self.states: list[dict[str, Any]] = []

    def load_screenshots(self, screenshot_ids: list[str]) -> None:
        """Load screenshot images from disk."""
        for screenshot_id in screenshot_ids:
            # Try different possible filenames
            possible_paths = [
                self.screenshots_dir / f"{screenshot_id}.png",
                self.screenshots_dir / f"{screenshot_id}.jpg",
                self.screenshots_dir / screenshot_id,
            ]

            for path in possible_paths:
                if path.exists():
                    img = self._cv2.imread(str(path))
                    if img is not None:
                        self.screenshots[screenshot_id] = img
                        logger.debug(f"Loaded screenshot {screenshot_id}: {img.shape}")
                        break
            else:
                logger.warning(f"Could not load screenshot: {screenshot_id}")

        logger.info(f"Loaded {len(self.screenshots)} screenshots")

    def extract_and_track_images(
        self,
        elements_by_screenshot: dict[str, list[dict[str, Any]]],
    ) -> None:
        """
        Extract image regions and track them across screenshots.

        Args:
            elements_by_screenshot: Dict mapping screenshot_id to list of element dicts
        """
        # Store for restricted search
        self.candidate_elements = elements_by_screenshot

        # First, extract all images from their source screenshots
        for screenshot_id, elements in elements_by_screenshot.items():
            if screenshot_id not in self.screenshots:
                logger.warning(f"Screenshot {screenshot_id} not loaded, skipping elements")
                continue

            screenshot = self.screenshots[screenshot_id]
            img_height, img_width = screenshot.shape[:2]

            for elem in elements:
                bbox = elem.get("bbox", {})
                x = int(bbox.get("x", 0))
                y = int(bbox.get("y", 0))
                width = int(bbox.get("width", 0))
                height = int(bbox.get("height", 0))

                # Skip invalid bboxes (negative pos or too small)
                if x < 0 or y < 0 or width < 10 or height < 10:
                    continue

                # Clamp to image bounds
                x = max(0, min(x, img_width - 1))
                y = max(0, min(y, img_height - 1))
                x2 = min(x + width, img_width)
                y2 = min(y + height, img_height)

                if x2 - x < 10 or y2 - y < 10:
                    continue

                # Crop the image region
                cropped = screenshot[y:y2, x:x2].copy()

                # Visual Filter: Skip if image is too uniform (likely background)
                if cropped.size > 0:
                    std_dev = self._np.std(cropped)
                    if std_dev < 2.0:  # Very uniform background
                        logger.debug(
                            f"Skipping low-variance element {elem.get('id')} (std={std_dev:.2f})"
                        )
                        continue

                tracked = TrackedImage(
                    id=elem.get("id", str(uuid4())),
                    name=elem.get("name")
                    or elem.get("text")
                    or elem.get("element_type", "unknown"),
                    source_screenshot_id=screenshot_id,
                    source_bbox={"x": x, "y": y, "width": x2 - x, "height": y2 - y},
                    image_data=cropped,
                    screens_found={screenshot_id},  # Found on source screen
                    element_type=elem.get("element_type", "unknown"),
                    text=elem.get("text"),
                    selector=elem.get("selector"),
                    extraction_category=elem.get("extraction_category", ""),
                )

                # Record match on source screenshot
                tracked.matches[screenshot_id] = ImageMatch(
                    screenshot_id=screenshot_id,
                    found=True,
                    bbox=tracked.source_bbox,
                    confidence=1.0,
                )

                self.tracked_images.append(tracked)

        logger.info(f"Extracted {len(self.tracked_images)} images from screenshots")

    def deduplicate_tracked_images(self) -> None:
        """
        Deduplicate tracked images that represent the same visual element.

        Using a Pure Comparison Model: Compare all extracted images pairwise
        to find visual matches. If Image A (Screen 1) matches Image B (Screen 2),
        they are merged. co-occurrence is based solely on actual Playwright extractions.
        """
        if not self.tracked_images:
            return

        logger.info(f"Deduplicating {len(self.tracked_images)} tracked images")

        # Use Union-Find to group identical images
        parent = {tracked.id: tracked.id for tracked in self.tracked_images}

        def find(i_id):
            if parent[i_id] == i_id:
                return i_id
            parent[i_id] = find(parent[i_id])
            return parent[i_id]

        def union(i_id, j_id):
            root_i = find(i_id)
            root_j = find(j_id)
            if root_i != root_j:
                parent[root_i] = root_j

        # Pairwise comparison to find identical elements across (or within) screens
        num_images = len(self.tracked_images)
        total_comparisons = (num_images * (num_images - 1)) // 2
        logger.info(
            f"[PERF_DEBUG] Starting pairwise comparison for {num_images} images ({total_comparisons} potential comparisons)"
        )
        start_dedup = time.time()

        for i in range(len(self.tracked_images)):
            for j in range(i + 1, len(self.tracked_images)):
                tracked = self.tracked_images[i]
                other_tracked = self.tracked_images[j]

                # Skip if already in the same group
                if find(tracked.id) == find(other_tracked.id):
                    continue

                # VISUAL CHECK: Merge if they are visually similar
                try:
                    t1 = tracked.image_data
                    t2 = other_tracked.image_data
                    if t1 is not None and t2 is not None:
                        # CONTAINMENT CHECK: Check if one image is contained within the other
                        # This handles "Fixed Header" scenarios where the header width varies (e.g. scrollbar, viewport change)
                        # but the content (logo, nav) is identical. Simple resizing distorts the text.

                        h1, w1 = t1.shape[:2]
                        h2, w2 = t2.shape[:2]

                        # Determine big vs small
                        if h1 * w1 > h2 * w2:
                            big, small = t1, t2
                        else:
                            big, small = t2, t1

                        score = 0.0
                        # Ensure small fits inside big
                        if small.shape[0] <= big.shape[0] and small.shape[1] <= big.shape[1]:
                            res = self._cv2.matchTemplate(big, small, self._cv2.TM_CCOEFF_NORMED)
                            _, max_val, _, _ = self._cv2.minMaxLoc(res)
                            score = float(max_val)
                        else:
                            # Fallback to resize if containment impossible (e.g. T1 tall/thin, T2 short/wide)
                            # Resize T2 to match T1
                            if t1.shape != t2.shape:
                                h, w = t1.shape[:2]
                                t2_resized = self._cv2.resize(t2, (w, h))
                            else:
                                t2_resized = t2

                            res = self._cv2.matchTemplate(
                                t2_resized, t1, self._cv2.TM_CCOEFF_NORMED
                            )
                            # When images are the same size, res is 1x1, so [0][0] is correct
                            # Use minMaxLoc for consistency and to handle any edge cases
                            _, max_val, _, _ = self._cv2.minMaxLoc(res)
                            score = float(max_val)

                        # LOGGING: Debug why potential matches fail
                        if score < self.similarity_threshold and score > 0.4:
                            logger.debug(
                                f"[DEDUP DEBUG] Failed match: {tracked.name} vs {other_tracked.name} "
                                f"Score={score:.4f} Threshold={self.similarity_threshold} "
                                f"Shapes: {t1.shape} vs {t2.shape}"
                            )

                        if score >= self.similarity_threshold:
                            union(tracked.id, other_tracked.id)
                            # Store match info for later bbox alignment
                            if tracked.matches.get(other_tracked.source_screenshot_id) is None:
                                tracked.matches[other_tracked.source_screenshot_id] = ImageMatch(
                                    screenshot_id=other_tracked.source_screenshot_id,
                                    found=True,
                                    confidence=score,
                                    bbox=other_tracked.source_bbox,
                                )
                except Exception as e:
                    logger.warning(
                        f"Error comparing images {tracked.id} and {other_tracked.id}: {e}"
                    )
                    continue

        duration_dedup = time.time() - start_dedup
        logger.info(f"[PERF_DEBUG] Pairwise deduplication finished in {duration_dedup:.2f}s")

        # Merge groups
        groups: dict[str, list[TrackedImage]] = defaultdict(list)
        for tracked in self.tracked_images:
            root_id = find(tracked.id)
            groups[root_id].append(tracked)

        new_tracked_images = []
        for _root_id, group in groups.items():
            if len(group) == 1:
                new_tracked_images.append(group[0])
                continue

            # Merge multiple TrackedImages into one
            # Pick the best representative:
            # 1. Highest variance (most "interesting" pixels, avoids background)
            # 2. Interactive category is better
            # 3. Prefer earlier screenshots
            def sorting_key(x, _group=group):
                # Variance (std dev) is primary indicator of "realness"
                variance = self._np.std(x.image_data) if x.image_data is not None else 0
                # Interactive category is better
                category_score = 100 if x.extraction_category == "interactive" else 0
                # Lower screenshot index is slightly better (heuristic)
                screen_index_penalty = -_group.index(x)
                return (variance, category_score, screen_index_penalty)

            representative = max(group, key=sorting_key)

            for other in group:
                if other.id == representative.id:
                    continue

                # Merge screens_found
                representative.screens_found.update(other.screens_found)

                # Merge matches (keeping the one with higher confidence if duplicate)
                for sid, match in other.matches.items():
                    if sid not in representative.matches or (
                        match.found and not representative.matches[sid].found
                    ):
                        representative.matches[sid] = match
                    elif match.found and representative.matches[sid].found:
                        # Ensure we have numbers to compare
                        other_conf = float(match.confidence or 0.0)
                        repr_conf = float(representative.matches[sid].confidence or 0.0)
                        if other_conf > repr_conf:
                            representative.matches[sid] = match

            new_tracked_images.append(representative)
            logger.debug(
                f"Merged {len(group)} images into one: {representative.name} ({representative.id})"
            )

        logger.info(
            f"Deduplication complete: {len(self.tracked_images)} -> {len(new_tracked_images)}"
        )
        self.tracked_images = new_tracked_images

    def cluster_into_states(self) -> list[dict[str, Any]]:
        """
        Cluster images into states based on co-occurrence.

        Images that appear on the exact same set of screenshots belong to the same state.
        """
        # Group images by their screen set
        screen_set_to_images: dict[frozenset[str], list[TrackedImage]] = defaultdict(list)

        for tracked in self.tracked_images:
            screen_set = frozenset(tracked.screens_found)
            screen_set_to_images[screen_set].append(tracked)
            logger.debug(
                f"[CLUSTER] Image '{tracked.name}' (id={tracked.id[:8]}) "
                f"screens_found={sorted(tracked.screens_found)}"
            )

        # Log grouping summary
        logger.info(
            f"[CLUSTER] Grouping {len(self.tracked_images)} images into {len(screen_set_to_images)} screen-set clusters"
        )
        for screen_set, images in screen_set_to_images.items():
            logger.info(
                f"[CLUSTER]   Screen set {sorted(screen_set)}: {len(images)} images "
                f"({', '.join([img.name[:20] for img in images[:5]])}{'...' if len(images) > 5 else ''})"
            )

        # Create states from clusters
        states_config = []
        state_index = 1

        for screen_set, images in screen_set_to_images.items():
            if not images:
                continue

            # Generate state name based on screens
            screen_list = sorted(screen_set)
            if len(screen_list) <= 3:
                screens_str = ", ".join(screen_list)
            else:
                screens_str = f"{', '.join(screen_list[:3])} (+{len(screen_list) - 3} more)"

            state_id = str(uuid4())
            state_name = f"State {state_index}"

            # Build stateImages for this state
            state_images: list[dict[str, Any]] = []
            for img in images:
                # Create a specialized stateImage for EACH screen this image was found on.
                # This ensures that we use the specific bounding box for that screen,
                # tackling the issue where elements move (e.g. footer) or reflow.
                for sid in img.screens_found:
                    # Determine BBox for this screen
                    bbox = img.source_bbox
                    # If this is the source screen, use source_bbox.
                    # If it's a matched screen, try to find the match bbox.
                    bbox_source = "default"
                    if sid == img.source_screenshot_id:
                        bbox = img.source_bbox
                        bbox_source = "source"
                    elif sid in img.matches and img.matches[sid].found and img.matches[sid].bbox:
                        # bbox is confirmed non-None by the condition check
                        bbox = img.matches[sid].bbox  # type: ignore[assignment]
                        bbox_source = "match"

                    # LOGGING: Trace why coord might be wrong
                    logger.debug(
                        f"[COORD DEBUG] StateImg for {img.name} on {sid}: Source={bbox_source}, BBox={bbox.get('x')},{bbox.get('y')} {bbox.get('width')}x{bbox.get('height')}"
                    )

                    # Create search region for this specific instance
                    search_regions = [
                        {
                            "id": str(uuid4()),
                            "name": f"Region for {img.name} on {sid}",
                            "x": int(bbox.get("x", 0)),
                            "y": int(bbox.get("y", 0)),
                            "width": int(bbox.get("width", 1)),
                            "height": int(bbox.get("height", 1)),
                            "offsetX": 0,
                            "offsetY": 0,
                        }
                    ]

                    # Unique ID for this instance
                    img_instance_id = str(uuid4())

                    state_image = {
                        "id": img_instance_id,
                        "name": img.name or f"Image {img.id[:8]}",
                        "patterns": [
                            {
                                "id": str(uuid4()),
                                "name": "Default pattern",
                                "imageId": None,
                                "searchRegions": search_regions,
                                "fixed": False,  # Per-screen instance is fixed to its own region
                                "similarity": self.similarity_threshold,
                            }
                        ],
                        "shared": False,
                        "source": "web-extraction",
                        "searchMode": "default",
                        "searchRegions": search_regions,
                        "screenshotId": sid,  # Crucial: Associate with specific screenshot
                        "screensFound": [sid],  # Only valid for this screen
                        "extractionCategory": img.extraction_category,
                    }
                    state_images.append(state_image)

            state_config = {
                "id": state_id,
                "name": state_name,
                "description": f"{len(images)} images appearing on {len(screen_set)} screenshots: {screens_str}",
                "stateImages": state_images,
                "screensFound": list(screen_set),  # Which screenshots this state appears on
                "regions": [],
                "locations": [],
                "strings": [],
                "position": {
                    "x": (state_index - 1) % 3 * 250 + 100,
                    "y": (state_index - 1) // 3 * 200 + 100,
                },
                "initial": state_index == 1,
                "isFinal": False,
            }
            states_config.append(state_config)
            state_index += 1

        logger.info(f"Clustered {len(self.tracked_images)} images into {len(states_config)} states")
        return states_config

    def build(
        self,
        elements_by_screenshot: dict[str, list[dict[str, Any]]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Build the state machine.

        Args:
            elements_by_screenshot: Dict mapping screenshot_id to list of element dicts

        Returns:
            Tuple of (states_config, transitions_config)
        """
        print("[IMAGE_MATCHING_DEBUG] ImageMatchingStateMachineBuilder.build() called", flush=True)
        print(
            f"[IMAGE_MATCHING_DEBUG] elements_by_screenshot has {len(elements_by_screenshot)} screenshots",
            flush=True,
        )

        # Load screenshots
        screenshot_ids = list(elements_by_screenshot.keys())
        print(f"[IMAGE_MATCHING_DEBUG] Loading screenshots: {screenshot_ids}", flush=True)
        self.load_screenshots(screenshot_ids)

        if not self.screenshots:
            print("[IMAGE_MATCHING_DEBUG] No screenshots loaded!", flush=True)
            logger.error("No screenshots loaded, cannot build state machine")
            return [], []
        print(f"[IMAGE_MATCHING_DEBUG] Loaded {len(self.screenshots)} screenshots", flush=True)

        # Extract images from each screenshot
        print("[IMAGE_MATCHING_DEBUG] Extracting images from screenshots...", flush=True)
        self.extract_and_track_images(elements_by_screenshot)

        if not self.tracked_images:
            print("[IMAGE_MATCHING_DEBUG] No images extracted!", flush=True)
            logger.warning("No images extracted, returning empty state machine")
            return [], []
        print(f"[IMAGE_MATCHING_DEBUG] Extracted {len(self.tracked_images)} images", flush=True)

        # Deduplicate images that represent the same visual element
        print("[IMAGE_MATCHING_DEBUG] Deduplicating images...", flush=True)
        self.deduplicate_tracked_images()

        # Log which images were found on which screenshots
        print(
            f"[IMAGE_MATCHING_DEBUG] Image search results: {len(self.tracked_images)} tracked images",
            flush=True,
        )
        for i, tracked in enumerate(self.tracked_images[:10]):  # Limit output
            print(
                f"[IMAGE_MATCHING_DEBUG]   Image {i}: found on screens {sorted(tracked.screens_found)}",
                flush=True,
            )
        if len(self.tracked_images) > 10:
            print(
                f"[IMAGE_MATCHING_DEBUG]   ... and {len(self.tracked_images) - 10} more images",
                flush=True,
            )

        # Cluster into states based on co-occurrence
        print("[IMAGE_MATCHING_DEBUG] Clustering images into states...", flush=True)
        states_config = self.cluster_into_states()
        print(f"[IMAGE_MATCHING_DEBUG] Created {len(states_config)} states", flush=True)
        for i, state in enumerate(states_config):
            print(
                f"[IMAGE_MATCHING_DEBUG]   State {i}: {len(state.get('stateImages', []))} images, screensFound={state.get('screensFound', [])}",
                flush=True,
            )

        # TODO: Derive transitions from navigation data
        transitions_config: list[dict[str, Any]] = []

        return states_config, transitions_config


def build_state_machine_from_extraction_result(
    extraction_result: "ExtractionResult",
    screenshots_dir: Path | str,
    similarity_threshold: float = 0.7,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Build a state machine from an ExtractionResult using image matching.

    This is the recommended entry point for creating a state machine from extraction data.
    It uses actual image matching (template matching) to find elements across screenshots,
    then groups them by co-occurrence.

    The algorithm:
    1. Load screenshots from disk
    2. For each element, crop its image region from the source screenshot
    3. Search for that image on ALL other screenshots using template matching
    4. Record which screenshots contain each image
    5. Group images by co-occurrence (same set of screenshots = same state)

    Args:
        extraction_result: ExtractionResult from the extraction orchestrator
        screenshots_dir: Path to directory containing screenshot images
        similarity_threshold: Minimum similarity score for template matching (0-1)

    Returns:
        Tuple of (states_config, transitions_config) ready for project configuration.
    """
    print("[IMAGE_MATCHING_DEBUG] build_state_machine_from_extraction_result called", flush=True)
    print(f"[IMAGE_MATCHING_DEBUG] screenshots_dir: {screenshots_dir}", flush=True)
    print(f"[IMAGE_MATCHING_DEBUG] similarity_threshold: {similarity_threshold}", flush=True)

    screenshots_dir = Path(screenshots_dir)
    print(f"[IMAGE_MATCHING_DEBUG] screenshots_dir exists: {screenshots_dir.exists()}", flush=True)
    if screenshots_dir.exists():
        files = list(screenshots_dir.iterdir())
        print(
            f"[IMAGE_MATCHING_DEBUG] screenshots_dir contains {len(files)} files: {[f.name for f in files[:10]]}",
            flush=True,
        )

    # Build elements_by_screenshot from RuntimeExtractionResult
    elements_by_screenshot: dict[str, list[dict[str, Any]]] = {}

    runtime_result = extraction_result.runtime_extraction
    print(f"[IMAGE_MATCHING_DEBUG] runtime_result exists: {runtime_result is not None}", flush=True)
    if runtime_result and runtime_result.states:
        print(
            f"[IMAGE_MATCHING_DEBUG] Found {len(runtime_result.states)} ExtractedStates",
            flush=True,
        )

        # Build element lookup map from runtime_result.elements
        all_elements = getattr(runtime_result, "elements", [])
        element_map: dict[str, Any] = {elem.id: elem for elem in all_elements}
        print(
            f"[IMAGE_MATCHING_DEBUG] Built element map with {len(element_map)} elements",
            flush=True,
        )

        for i, runtime_state in enumerate(runtime_result.states):
            # Get screenshot_id - ExtractedState has screenshot_id directly (not screenshot.id)
            screenshot_id = getattr(runtime_state, "screenshot_id", None) or runtime_state.id
            print(
                f"[IMAGE_MATCHING_DEBUG] Processing state {i}: screenshot_id={screenshot_id}",
                flush=True,
            )

            # Get elements - ExtractedState has element_ids, we look them up in element_map
            element_ids = getattr(runtime_state, "element_ids", [])
            print(f"[IMAGE_MATCHING_DEBUG]   Has {len(element_ids)} element_ids", flush=True)

            elements = []
            for elem_id in element_ids:
                elem = element_map.get(elem_id)
                if elem is None:
                    print(
                        f"[IMAGE_MATCHING_DEBUG]   WARNING: Element {elem_id} not found in map",
                        flush=True,
                    )
                    continue

                elem_dict = {
                    "id": elem.id,
                    "name": getattr(elem, "name", None) or getattr(elem, "text_content", None),
                    "text": getattr(elem, "text_content", None),
                    "element_type": (
                        elem.element_type.value
                        if hasattr(elem.element_type, "value")
                        else str(elem.element_type)
                    ),
                    "bbox": (
                        elem.bbox.to_dict()
                        if hasattr(elem.bbox, "to_dict")
                        else {
                            "x": elem.bbox.x,
                            "y": elem.bbox.y,
                            "width": elem.bbox.width,
                            "height": elem.bbox.height,
                        }
                    ),
                    "selector": getattr(elem, "selector", None),
                    "extraction_category": getattr(elem, "extraction_category", ""),
                }
                elements.append(elem_dict)

            elements_by_screenshot[screenshot_id] = elements
            logger.info(f"Screenshot {screenshot_id}: {len(elements)} elements")
            print(
                f"[IMAGE_MATCHING_DEBUG] Added {len(elements)} elements for screenshot {screenshot_id}",
                flush=True,
            )
    else:
        print("[IMAGE_MATCHING_DEBUG] No ExtractedStates in extraction result!", flush=True)
        logger.warning("No ExtractedStates in extraction result")

    print(
        f"[IMAGE_MATCHING_DEBUG] Total screenshots with elements: {len(elements_by_screenshot)}",
        flush=True,
    )
    if not elements_by_screenshot:
        print("[IMAGE_MATCHING_DEBUG] No elements found, returning empty state machine", flush=True)
        logger.error("No elements found in extraction result")
        return [], []

    # Build state machine using image matching
    print("[IMAGE_MATCHING_DEBUG] Creating ImageMatchingStateMachineBuilder...", flush=True)
    try:
        builder = ImageMatchingStateMachineBuilder(
            screenshots_dir=screenshots_dir,
            similarity_threshold=similarity_threshold,
        )
        print("[IMAGE_MATCHING_DEBUG] Builder created successfully", flush=True)
    except Exception as e:
        print(f"[IMAGE_MATCHING_DEBUG] Failed to create builder: {e}", flush=True)
        logger.error(f"Failed to create ImageMatchingStateMachineBuilder: {e}")
        return [], []

    print("[IMAGE_MATCHING_DEBUG] Calling builder.build()...", flush=True)
    result = builder.build(elements_by_screenshot)
    print(
        f"[IMAGE_MATCHING_DEBUG] builder.build() returned {len(result[0])} states, {len(result[1])} transitions",
        flush=True,
    )
    return result
