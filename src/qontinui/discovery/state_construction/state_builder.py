"""State Builder - Constructs Complete State Objects

This module builds holistic State objects from screenshot sequences and transition data,
creating complete state representations with:
- StateImages: Persistent visual elements that identify the state
- StateRegions: Functional areas (grids, panels, clickable regions)
- StateLocations: Specific click points that trigger transitions
- Meaningful names generated from OCR text extraction

The StateBuilder is the core of the state construction pipeline, integrating:
- ConsistencyDetector: Finds persistent elements across screenshots
- DifferentialConsistencyDetector: Identifies state boundaries from transitions
- OCRNameGenerator: Generates semantic names from visual text
- ElementIdentifier: Classifies and categorizes detected elements

Example Usage:
    >>> from qontinui.discovery.state_construction import StateBuilder, TransitionInfo
    >>>
    >>> builder = StateBuilder()
    >>>
    >>> # Build state from screenshots
    >>> screenshots = [capture1, capture2, capture3]
    >>> state = builder.build_state_from_screenshots(screenshots)
    >>>
    >>> print(f"State: {state.name}")
    >>> print(f"Images: {len(state.state_images)}")
    >>> print(f"Regions: {len(state.state_regions)}")
    >>> print(f"Locations: {len(state.state_locations)}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

if TYPE_CHECKING:
    from qontinui.model.state.state import State
    from qontinui.model.state.state_image import StateImage
    from qontinui.model.state.state_location import StateLocation
    from qontinui.model.state.state_region import StateRegion


@dataclass
class TransitionInfo:
    """Information about a state transition.

    Captures the before/after screenshots and user input that triggered
    a transition between states. Used for differential consistency analysis
    and click point clustering.

    Attributes:
        before_screenshot: Screenshot before the transition
        after_screenshot: Screenshot after the transition
        click_point: (x, y) coordinates where user clicked (if applicable)
        input_events: List of input events (clicks, keys, etc.)
        target_state_name: Name of the target state (if known)
        timestamp: When the transition occurred
    """

    before_screenshot: np.ndarray
    after_screenshot: np.ndarray
    click_point: Optional[Tuple[int, int]] = None
    input_events: List[Dict[str, Any]] = None
    target_state_name: Optional[str] = None
    timestamp: Optional[float] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.input_events is None:
            self.input_events = []


class StateBuilder:
    """Builds complete State objects from screenshots and transitions.

    The StateBuilder is the main entry point for state construction. It orchestrates
    multiple detection and analysis algorithms to create rich State objects with
    all their components.

    Architecture:
        1. Name Generation: Extract meaningful names from OCR
        2. Image Detection: Find persistent visual elements (StateImages)
        3. Region Detection: Identify functional areas (StateRegions)
        4. Location Detection: Cluster click points (StateLocations)
        5. Boundary Detection: Determine state spatial extent (for modals/dialogs)

    The builder integrates with:
        - ConsistencyDetector: Finds elements consistent across screenshots
        - DifferentialConsistencyDetector: Analyzes transition pairs
        - OCRNameGenerator: Extracts text for naming
        - ElementIdentifier: Classifies detected elements

    Example:
        >>> builder = StateBuilder()
        >>>
        >>> # Simple case: just screenshots
        >>> state = builder.build_state_from_screenshots(screenshots)
        >>>
        >>> # Advanced: with transition data
        >>> transitions = [TransitionInfo(before, after, click_pt) for ...]
        >>> state = builder.build_state_from_screenshots(
        ...     screenshot_sequence=screenshots,
        ...     transitions_to_state=transitions,
        ...     transitions_from_state=outgoing_transitions
        ... )
    """

    def __init__(
        self,
        consistency_threshold: float = 0.9,
        min_image_area: int = 100,
        min_region_area: int = 500,
    ):
        """Initialize StateBuilder.

        Args:
            consistency_threshold: Minimum consistency for StateImages (0-1)
            min_image_area: Minimum pixel area for detected images
            min_region_area: Minimum pixel area for detected regions
        """
        self.consistency_threshold = consistency_threshold
        self.min_image_area = min_image_area
        self.min_region_area = min_region_area

        # Import dependencies lazily to avoid circular imports
        self._consistency_detector = None
        self._diff_detector = None
        self._name_generator = None
        self._element_identifier = None

    @property
    def consistency_detector(self):
        """Lazy-load ConsistencyDetector."""
        if self._consistency_detector is None:
            try:
                from qontinui_web.research_env.detectors.consistency_detector import (
                    ConsistencyDetector,
                )

                self._consistency_detector = ConsistencyDetector()
            except ImportError:
                # Fallback: use a simple implementation
                self._consistency_detector = None
        return self._consistency_detector

    @property
    def diff_detector(self):
        """Lazy-load DifferentialConsistencyDetector."""
        if self._diff_detector is None:
            # TODO: Implement DifferentialConsistencyDetector
            # For now, return None
            self._diff_detector = None
        return self._diff_detector

    @property
    def name_generator(self):
        """Lazy-load OCRNameGenerator."""
        if self._name_generator is None:
            # TODO: Implement OCRNameGenerator
            # For now, use fallback
            self._name_generator = FallbackNameGenerator()
        return self._name_generator

    @property
    def element_identifier(self):
        """Lazy-load ElementIdentifier."""
        if self._element_identifier is None:
            # TODO: Implement ElementIdentifier
            # For now, use fallback
            self._element_identifier = FallbackElementIdentifier()
        return self._element_identifier

    def build_state_from_screenshots(
        self,
        screenshot_sequence: List[np.ndarray],
        transitions_to_state: Optional[List[TransitionInfo]] = None,
        transitions_from_state: Optional[List[TransitionInfo]] = None,
        state_name: Optional[str] = None,
    ) -> State:
        """Build a complete State object from screenshot data.

        This is the main entry point for state construction. It analyzes the provided
        screenshots and transition data to construct a complete State with all components.

        Args:
            screenshot_sequence: Screenshots where this state is active
            transitions_to_state: Transitions leading TO this state (for boundary detection)
            transitions_from_state: Transitions leading FROM this state (for click points)
            state_name: Optional explicit name (otherwise generated from OCR)

        Returns:
            Fully constructed State object with StateImages, StateRegions, StateLocations

        Example:
            >>> screenshots = load_screenshots("state_captures/*.png")
            >>> state = builder.build_state_from_screenshots(screenshots)
            >>>
            >>> # Or with transitions
            >>> transitions = capture_transitions()
            >>> state = builder.build_state_from_screenshots(
            ...     screenshots,
            ...     transitions_to_state=transitions
            ... )
        """
        from qontinui.model.state.state import State

        if not screenshot_sequence:
            raise ValueError("screenshot_sequence cannot be empty")

        # 1. Generate state name
        if state_name is None:
            state_name = self._generate_state_name(screenshot_sequence, transitions_to_state)

        # 2. Identify StateImages (persistent visual elements)
        state_images = self._identify_state_images(screenshot_sequence)

        # 3. Identify StateRegions (functional areas)
        state_regions = self._identify_state_regions(screenshot_sequence)

        # 4. Identify StateLocations (click points from transitions)
        state_locations = self._identify_state_locations(transitions_from_state)

        # 5. Determine state boundary (for modal dialogs, popup windows)
        boundary_bbox = None
        if transitions_to_state and len(transitions_to_state) >= 10:
            boundary_bbox = self._determine_state_boundary(transitions_to_state)

        # 6. Build State object using the existing model
        state = State(
            name=state_name,
            description=f"Auto-generated state from {len(screenshot_sequence)} screenshots",
        )

        # Add state images
        for state_image in state_images:
            state.add_state_image(state_image)

        # Add state regions
        for state_region in state_regions:
            state.add_state_region(state_region)

        # Add state locations
        for state_location in state_locations:
            state.add_state_location(state_location)

        # Set boundary if detected (for modal states)
        if boundary_bbox is not None:
            from qontinui.model.element.region import Region

            x, y, w, h = boundary_bbox
            state.usable_area = Region(x, y, w, h)

        return state

    def _generate_state_name(
        self,
        screenshots: List[np.ndarray],
        transitions: Optional[List[TransitionInfo]],
    ) -> str:
        """Generate meaningful name for the state.

        Strategy:
            1. Use OCR to extract text from title bar / header area
            2. Find prominent text near top of screen
            3. Use transition context if available
            4. Fallback to generic name with hash

        Args:
            screenshots: Screenshots of this state
            transitions: Transitions leading to this state (for context)

        Returns:
            State name (sanitized, lowercase with underscores)
        """
        # Use first screenshot as representative
        representative = screenshots[0]

        # Try OCR-based naming
        name = self.name_generator.generate_state_name(representative)

        # If transitions provide context, enhance the name
        if transitions and transitions[0].target_state_name:
            name = transitions[0].target_state_name

        return name

    def _identify_state_images(
        self, screenshots: List[np.ndarray]
    ) -> List[StateImage]:
        """Identify StateImages - persistent visual elements that define the state.

        StateImages are visual patterns that consistently appear across all screenshots
        of this state. They serve as the primary identification mechanism.

        Examples:
            - Title bar text/logo
            - Navigation icons
            - Persistent UI chrome
            - State-specific graphics

        Args:
            screenshots: Screenshots where state is active

        Returns:
            List of StateImage objects with names and locations
        """
        from qontinui.model.element.image import Image
        from qontinui.model.element.pattern import Pattern
        from qontinui.model.state.state_image import StateImage

        if len(screenshots) < 2:
            # Need at least 2 screenshots to find consistency
            return []

        # Use consistency detector to find persistent elements
        consistent_regions = self._detect_consistent_regions(
            screenshots, self.consistency_threshold, self.min_image_area
        )

        # Convert to StateImages
        state_images = []
        for idx, region_info in enumerate(consistent_regions):
            bbox = region_info["bbox"]
            x, y, w, h = bbox

            # Extract image from first screenshot
            img_data = screenshots[0][y : y + h, x : x + w].copy()

            # Generate semantic name
            context = self._classify_image_context(bbox, screenshots[0])
            name = self.name_generator.generate_name_from_image(img_data, context)

            # Create Pattern from image data
            pattern = Pattern(name=name)
            pattern._image_data = img_data  # Store raw data

            # Create StateImage
            state_image = StateImage(image=pattern, name=name)
            state_image._similarity = region_info.get("confidence", 0.85)

            # Set position metadata
            state_image.metadata["bbox"] = bbox
            state_image.metadata["area"] = w * h
            state_image.metadata["context"] = context

            state_images.append(state_image)

        return state_images

    def _detect_consistent_regions(
        self,
        screenshots: List[np.ndarray],
        threshold: float,
        min_area: int,
    ) -> List[Dict[str, Any]]:
        """Detect regions that are consistent across all screenshots.

        Uses edge detection and template matching to find regions that appear
        in the same location with similar visual appearance across screenshots.

        Args:
            screenshots: List of screenshots
            threshold: Consistency threshold (0-1)
            min_area: Minimum region area in pixels

        Returns:
            List of region dictionaries with bbox and confidence
        """
        regions = []

        if len(screenshots) < 2:
            return regions

        # Use first screenshot as reference
        reference = screenshots[0]
        gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

        # Detect edges to find candidate regions
        edges = cv2.Canny(gray_ref, 50, 150)

        # Close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            if area < min_area:
                continue

            # Check if this region is consistent across all screenshots
            consistency_score = self._check_region_consistency(
                screenshots, (x, y, w, h)
            )

            if consistency_score >= threshold:
                regions.append(
                    {
                        "bbox": (x, y, w, h),
                        "confidence": float(consistency_score),
                        "area": area,
                    }
                )

        return regions

    def _check_region_consistency(
        self, screenshots: List[np.ndarray], bbox: Tuple[int, int, int, int]
    ) -> float:
        """Check how consistent a region is across screenshots.

        Uses template matching and structural similarity to measure consistency.

        Args:
            screenshots: List of screenshots
            bbox: Bounding box (x, y, w, h)

        Returns:
            Consistency score (0-1)
        """
        x, y, w, h = bbox

        # Extract reference template from first screenshot
        template = screenshots[0][y : y + h, x : x + w]
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        similarities = []

        for screenshot in screenshots[1:]:
            # Extract same region
            region = screenshot[y : y + h, x : x + w]
            region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

            # Compute similarity using normalized cross-correlation
            result = cv2.matchTemplate(region_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            similarities.append(max_val)

        if not similarities:
            return 1.0

        # Return mean similarity
        return float(np.mean(similarities))

    def _classify_image_context(
        self, bbox: Tuple[int, int, int, int], screenshot: np.ndarray
    ) -> str:
        """Classify what type of element this is based on position and size.

        Uses heuristics to determine element context (title_bar, icon, button, etc.)

        Args:
            bbox: Bounding box (x, y, w, h)
            screenshot: Full screenshot

        Returns:
            Context string (e.g., 'title_bar', 'icon', 'button')
        """
        x, y, w, h = bbox
        img_h, img_w = screenshot.shape[:2]

        # Title bar: top 10%, spans width
        if y < img_h * 0.1 and w > img_w * 0.3:
            return "title_bar"

        # Icon: small, roughly square
        if w < 100 and h < 100 and 0.5 < w / h < 2.0:
            return "icon"

        # Button: medium rectangular
        if 50 < w < 300 and 20 < h < 80:
            return "button"

        # Header: top third, wide
        if y < img_h * 0.33 and w > img_w * 0.5:
            return "header"

        # Logo: small to medium, top area
        if y < img_h * 0.15 and 50 < w < 200 and 30 < h < 150:
            return "logo"

        return "element"

    def _identify_state_regions(
        self, screenshots: List[np.ndarray]
    ) -> List[StateRegion]:
        """Identify StateRegions - functional areas like panels, grids, or lists.

        StateRegions represent interactive or significant areas within a state.
        They are typically larger than StateImages and may contain multiple elements.

        Examples:
            - Inventory grids
            - Skill panels
            - Menu lists
            - Input form areas

        Args:
            screenshots: Screenshots of the state

        Returns:
            List of StateRegion objects
        """
        from qontinui.model.element.region import Region
        from qontinui.model.state.state_region import StateRegion

        regions = []

        # Use element identifier to find structured regions
        detected = self.element_identifier.identify_regions(screenshots)

        for region_info in detected:
            bbox = region_info["bbox"]
            region_type = region_info.get("type", "panel")

            # Generate semantic name
            name = self._generate_region_name(region_info, screenshots[0])

            # Create Region object
            x, y, w, h = bbox
            region_obj = Region(x, y, w, h)

            # Create StateRegion
            state_region = StateRegion(region=region_obj, name=name)
            state_region._interaction_region = True
            state_region.metadata["type"] = region_type
            state_region.metadata["bbox"] = bbox

            regions.append(state_region)

        return regions

    def _generate_region_name(
        self, region_info: Dict[str, Any], screenshot: np.ndarray
    ) -> str:
        """Generate name for a region.

        Tries OCR first, falls back to type + position.

        Args:
            region_info: Region information dict
            screenshot: Screenshot containing the region

        Returns:
            Region name
        """
        region_type = region_info.get("type", "region")
        bbox = region_info["bbox"]
        x, y, w, h = bbox

        # Try OCR
        region_img = screenshot[y : y + h, x : x + w]
        ocr_name = self.name_generator.generate_name_from_image(region_img, region_type)

        # If OCR gives meaningful name, use it
        if len(ocr_name) > len(region_type) + 5:
            return ocr_name

        # Otherwise use type + position
        return f"{region_type}_{x}_{y}"

    def _identify_state_locations(
        self, transitions: Optional[List[TransitionInfo]]
    ) -> List[StateLocation]:
        """Identify StateLocations - clickable points that trigger transitions.

        Clusters click points from transitions to find stable, effective click locations.
        Each location is associated with a target state.

        Args:
            transitions: Transitions FROM this state

        Returns:
            List of StateLocation objects
        """
        from qontinui.model.element.location import Location
        from qontinui.model.state.state_location import StateLocation

        if not transitions:
            return []

        # Group transitions by target state
        by_target: Dict[str, List[TransitionInfo]] = {}
        for trans in transitions:
            target = trans.target_state_name or "unknown"
            if target not in by_target:
                by_target[target] = []
            by_target[target].append(trans)

        locations = []

        for target_state, trans_list in by_target.items():
            # Extract click points
            click_points = [
                t.click_point for t in trans_list if t.click_point is not None
            ]

            if not click_points:
                continue

            # Compute centroid (mean position)
            centroid = np.mean(click_points, axis=0).astype(int)

            # Compute consistency (inverse of standard deviation)
            std = np.std(click_points, axis=0)
            consistency = 1.0 / (1.0 + np.mean(std))  # 0-1 score

            # Create Location and StateLocation
            loc = Location(x=int(centroid[0]), y=int(centroid[1]))
            state_location = StateLocation(
                location=loc, name=f"click_to_{target_state}"
            )
            state_location.metadata["target_state"] = target_state
            state_location.metadata["confidence"] = float(consistency)
            state_location.metadata["sample_size"] = len(click_points)

            locations.append(state_location)

        return locations

    def _determine_state_boundary(
        self, transitions: List[TransitionInfo]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Determine bounding box of state using differential consistency.

        Useful for modal dialogs, popup windows, or other overlay states where
        the state occupies a specific region rather than the full screen.

        Uses differential consistency detection to find regions that consistently
        change during transitions to this state.

        Args:
            transitions: Transitions TO this state

        Returns:
            Bounding box (x, y, w, h) or None if full-screen state
        """
        if not self.diff_detector:
            # Differential detector not available
            return None

        # Create transition pairs for differential analysis
        pairs = [(t.before_screenshot, t.after_screenshot) for t in transitions]

        try:
            # Use differential consistency detector
            regions = self.diff_detector.detect_state_regions(
                pairs, consistency_threshold=0.7, min_region_area=500
            )

            if not regions:
                return None

            # Return largest region as state boundary
            largest = max(regions, key=lambda r: r.bbox[2] * r.bbox[3])
            return largest.bbox

        except Exception:
            # Detector failed, return None
            return None


class FallbackNameGenerator:
    """Fallback name generator when OCR is not available.

    Generates simple position-based names.
    """

    def generate_state_name(self, screenshot: np.ndarray) -> str:
        """Generate fallback state name."""
        img_hash = hash(screenshot.tobytes()) % 10000
        return f"state_{img_hash}"

    def generate_name_from_image(self, image: np.ndarray, context: str) -> str:
        """Generate fallback image name."""
        img_hash = hash(image.tobytes()) % 1000
        return f"{context}_{img_hash}"


class FallbackElementIdentifier:
    """Fallback element identifier when ML is not available.

    Uses basic geometric heuristics.
    """

    def identify_regions(self, screenshots: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Identify regions using basic heuristics."""
        regions = []

        if not screenshots:
            return regions

        # Simple implementation: detect large rectangular areas
        screenshot = screenshots[0]
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # Find contours
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Look for medium to large regions
            if 5000 < area < 100000:
                regions.append(
                    {
                        "bbox": (x, y, w, h),
                        "type": "panel",
                        "confidence": 0.6,
                    }
                )

        return regions
