"""StateImage Extraction Service for qontinui-runner.

This service extracts identifying images (StateImages) from detected states,
particularly around click locations. It uses edge detection and contour analysis
to find UI elements and determines whether positions are fixed or dynamic.

Key Features:
- Extracts images within state boundaries
- Focuses on regions around click locations from input events
- Uses edge detection and contour analysis to find UI elements
- Determines if positions are fixed (same across frames) or dynamic
- Supports multiple extraction methods
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from qontinui.discovery.state_analysis.models import DetectedState, Frame, InputEvent, StateImage

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ImageExtractionConfig:
    """Configuration for image extraction."""

    min_size: tuple[int, int] = (20, 20)
    max_size: tuple[int, int] = (500, 500)
    edge_detection: str = "canny"  # "canny", "sobel", "laplacian"
    contour_approximation: float = 0.02
    extract_at_click_locations: bool = True
    click_region_padding: int = 20
    position_tolerance_px: int = 5
    canny_threshold1: int = 50
    canny_threshold2: int = 150
    similarity_threshold: float = 0.85
    min_contour_area: int = 100
    max_contours: int = 50


class StateImageExtractor:
    """Extracts StateImages from detected states using various methods.

    This service analyzes frames belonging to a state and identifies visual
    elements that can be used to recognize and distinguish that state.
    """

    def __init__(self, config: ImageExtractionConfig | None = None):
        """Initialize the StateImage extractor.

        Args:
            config: Configuration for extraction. If None, uses defaults.
        """
        self.config = config or ImageExtractionConfig()
        logger.info("StateImageExtractor initialized with config: %s", self.config)

    def extract_from_state(
        self,
        state: DetectedState,
        frames: list[Frame],
        events: list[InputEvent],
    ) -> list[StateImage]:
        """Extract StateImages from a detected state.

        This is the main entry point for image extraction. It:
        1. Gets frames belonging to this state
        2. Identifies click locations within state
        3. Extracts regions around click locations
        4. Detects additional UI elements via contour analysis
        5. Assigns positions (fixed vs. dynamic)

        Args:
            state: The detected state to extract images from
            frames: All available frames
            events: All input events

        Returns:
            List of extracted StateImage objects

        Raises:
            ValueError: If state has no frames or frames are invalid
        """
        logger.info("Extracting images from state: %s", state.name)

        # Step 1: Get frames for this state
        state_frames = self._get_state_frames(state, frames)
        if not state_frames:
            logger.warning("No frames found for state %s", state.name)
            return []

        logger.debug("Found %d frames for state %s", len(state_frames), state.name)

        # Step 2: Identify click locations within state
        click_locations = self._get_click_locations(state, events)
        logger.debug("Found %d click locations in state %s", len(click_locations), state.name)

        extracted_images: list[StateImage] = []

        # Step 3: Extract regions around click locations
        if self.config.extract_at_click_locations and click_locations:
            for click_x, click_y in click_locations:
                for _i, frame in enumerate(state_frames):
                    try:
                        state_image = self.extract_at_location(
                            frame.image,
                            click_x,
                            click_y,
                            padding=self.config.click_region_padding,
                            frame_index=frame.frame_index,
                        )
                        if state_image:
                            state_image.name = f"{state.name}_click_{click_x}_{click_y}"
                            state_image.extraction_method = "click_location"  # type: ignore[attr-defined]
                            extracted_images.append(state_image)
                            logger.debug(
                                "Extracted image at click location (%d, %d) from frame %d",
                                click_x,
                                click_y,
                                frame.frame_index,
                            )
                            break  # Only extract from first frame
                    except Exception as e:
                        logger.error(
                            "Error extracting at location (%d, %d): %s",
                            click_x,
                            click_y,
                            e,
                        )

        # Step 4: Detect additional UI elements via contour analysis
        if state_frames:
            try:
                reference_frame = state_frames[0]
                contour_images = self._extract_from_contours(
                    reference_frame,
                    state,
                    click_locations,
                )
                extracted_images.extend(contour_images)
                logger.debug("Extracted %d images from contour analysis", len(contour_images))
            except Exception as e:
                logger.error("Error in contour analysis: %s", e)

        # Step 5: Determine position types (fixed vs. dynamic)
        for state_image in extracted_images:
            try:
                occurrences = self._find_occurrences(state_image, state_frames)
                is_fixed = self.determine_position_type(state_image, occurrences)
                state_image.position_type = "fixed" if is_fixed else "dynamic"  # type: ignore[attr-defined]
                state_image.metadata["occurrences_count"] = len(occurrences)
                logger.debug(
                    "Image %s classified as %s position (found in %d frames)",
                    state_image.name,
                    state_image.position_type,  # type: ignore[attr-defined]
                    len(occurrences),
                )
            except Exception as e:
                logger.error("Error determining position type for %s: %s", state_image.name, e)
                state_image.position_type = "unknown"  # type: ignore[attr-defined]

        logger.info("Extracted %d total images from state %s", len(extracted_images), state.name)
        return extracted_images

    def extract_at_location(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        padding: int = 20,
        frame_index: int | None = None,
    ) -> StateImage | None:
        """Extract image region around a specific location.

        Args:
            frame: The frame to extract from (OpenCV BGR format)
            x: X coordinate of the center point
            y: Y coordinate of the center point
            padding: Padding around the center point in pixels
            frame_index: Index of the source frame (for metadata)

        Returns:
            StateImage object if extraction succeeds, None otherwise
        """
        try:
            h, w = frame.shape[:2]

            # Calculate bounding box with padding
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + padding)
            y2 = min(h, y + padding)

            # Validate size
            bbox_w = x2 - x1
            bbox_h = y2 - y1

            if bbox_w < self.config.min_size[0] or bbox_h < self.config.min_size[1]:
                logger.debug("Region too small at (%d, %d): %dx%d", x, y, bbox_w, bbox_h)
                return None

            if bbox_w > self.config.max_size[0] or bbox_h > self.config.max_size[1]:
                logger.debug("Region too large at (%d, %d): %dx%d", x, y, bbox_w, bbox_h)
                return None

            # Extract image
            extracted = frame[y1:y2, x1:x2].copy()

            # Extract larger context (2x padding)
            context_padding = padding * 2
            cx1 = max(0, x - context_padding)
            cy1 = max(0, y - context_padding)
            cx2 = min(w, x + context_padding)
            cy2 = min(h, y + context_padding)
            context = frame[cy1:cy2, cx1:cx2].copy()

            # Create StateImage
            state_image = StateImage(  # type: ignore[call-arg]
                name=f"image_{x}_{y}",
                image=extracted,  # type: ignore[arg-type]
                bbox=(x1, y1, bbox_w, bbox_h),
                position_type="unknown",  # Will be determined later
                position=(x1, y1),
                similarity_threshold=self.config.similarity_threshold,
                context_image=context,
                source_frame_index=frame_index,
                extraction_method="location",
                metadata={
                    "center_point": (x, y),
                    "padding": padding,
                },
            )

            return state_image

        except Exception as e:
            logger.error("Error extracting at location (%d, %d): %s", x, y, e)
            return None

    def detect_contours(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect UI element contours using edge detection.

        Args:
            frame: The frame to analyze (OpenCV BGR format)

        Returns:
            List of bounding boxes as (x, y, width, height) tuples
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply edge detection based on config
            if self.config.edge_detection == "canny":
                edges = cv2.Canny(
                    gray,
                    self.config.canny_threshold1,
                    self.config.canny_threshold2,
                )
            elif self.config.edge_detection == "sobel":
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                edges = cv2.magnitude(sobelx, sobely)
                edges = np.uint8(edges)  # type: ignore[assignment]
            elif self.config.edge_detection == "laplacian":
                edges = cv2.Laplacian(gray, cv2.CV_64F)
                edges = np.uint8(np.absolute(edges))  # type: ignore[assignment]
            else:
                logger.warning("Unknown edge detection method: %s", self.config.edge_detection)
                edges = cv2.Canny(gray, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(
                edges,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            # Convert to bounding boxes and filter
            bboxes = []
            for contour in contours:
                # Calculate area
                area = cv2.contourArea(contour)
                if area < self.config.min_contour_area:
                    continue

                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Filter by size
                if (
                    w < self.config.min_size[0]
                    or h < self.config.min_size[1]
                    or w > self.config.max_size[0]
                    or h > self.config.max_size[1]
                ):
                    continue

                bboxes.append((x, y, w, h))

                # Limit number of contours
                if len(bboxes) >= self.config.max_contours:
                    break

            logger.debug("Detected %d contours", len(bboxes))
            return bboxes

        except Exception as e:
            logger.error("Error detecting contours: %s", e)
            return []

    def determine_position_type(
        self,
        image: StateImage,
        occurrences: list[tuple[int, int]],
    ) -> bool:
        """Determine if an image position is fixed or dynamic.

        Analyzes multiple occurrences to determine if the image appears at
        the same location across frames (fixed) or at varying locations (dynamic).

        Args:
            image: The StateImage to analyze
            occurrences: List of (x, y) positions where image was found

        Returns:
            True if position is fixed (same location across frames), False otherwise
        """
        if len(occurrences) < 2:
            # Not enough data, assume fixed if found only once
            return True

        # Calculate variance in positions
        positions = np.array(occurrences)
        mean_pos = positions.mean(axis=0)
        distances = np.sqrt(((positions - mean_pos) ** 2).sum(axis=1))
        max_distance = distances.max()

        # If all occurrences are within tolerance, consider it fixed
        is_fixed = max_distance <= self.config.position_tolerance_px

        logger.debug(
            "Position analysis for %s: %d occurrences, max distance: %.2f, fixed: %s",
            image.name,
            len(occurrences),
            max_distance,
            is_fixed,
        )

        return is_fixed  # type: ignore[no-any-return]

    def find_best_crop(
        self,
        frame: np.ndarray,
        region: tuple[int, int, int, int],
    ) -> StateImage:
        """Find optimal crop boundaries using edge detection.

        Refines a region by detecting edges and finding the tightest
        bounding box around significant content.

        Args:
            frame: The frame containing the region
            region: Initial region as (x, y, width, height)

        Returns:
            StateImage with optimized crop boundaries
        """
        try:
            x, y, w, h = region

            # Extract region
            roi = frame[y : y + h, x : x + w]

            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Apply edge detection
            edges = cv2.Canny(
                gray,
                self.config.canny_threshold1,
                self.config.canny_threshold2,
            )

            # Find contours in the ROI
            contours, _ = cv2.findContours(
                edges,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            if contours:
                # Get bounding box of all contours combined
                all_points = np.vstack(contours)
                rx, ry, rw, rh = cv2.boundingRect(all_points)

                # Add small padding
                padding = 5
                rx = max(0, rx - padding)
                ry = max(0, ry - padding)
                rw = min(w - rx, rw + 2 * padding)
                rh = min(h - ry, rh + 2 * padding)

                # Extract refined region
                refined = roi[ry : ry + rh, rx : rx + rw].copy()

                # Create StateImage with refined boundaries
                state_image = StateImage(  # type: ignore[call-arg]
                    name=f"refined_{x+rx}_{y+ry}",
                    image=refined,  # type: ignore[arg-type]
                    bbox=(x + rx, y + ry, rw, rh),
                    position_type="unknown",
                    position=(x + rx, y + ry),
                    similarity_threshold=self.config.similarity_threshold,
                    context_image=roi.copy(),
                    source_frame_index=None,
                    extraction_method="best_crop",
                    metadata={
                        "original_region": region,
                        "refined": True,
                    },
                )

                return state_image

            # If no contours found, return original region
            state_image = StateImage(  # type: ignore[call-arg]
                name=f"original_{x}_{y}",
                image=roi.copy(),  # type: ignore[arg-type]
                bbox=(x, y, w, h),
                position_type="unknown",
                position=(x, y),
                similarity_threshold=self.config.similarity_threshold,
                context_image=roi.copy(),
                source_frame_index=None,
                extraction_method="best_crop",
                metadata={
                    "original_region": region,
                    "refined": False,
                },
            )

            return state_image

        except Exception as e:
            logger.error("Error finding best crop for region %s: %s", region, e)
            # Return original region as fallback
            x, y, w, h = region
            roi = frame[y : y + h, x : x + w].copy()
            return StateImage(  # type: ignore[call-arg]
                name=f"fallback_{x}_{y}",
                image=roi,  # type: ignore[arg-type]
                bbox=(x, y, w, h),
                position_type="unknown",
                position=(x, y),
                similarity_threshold=self.config.similarity_threshold,
                context_image=roi,
                source_frame_index=None,
                extraction_method="best_crop_fallback",
                metadata={"error": str(e)},
            )

    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    def _get_state_frames(
        self,
        state: DetectedState,
        frames: list[Frame],
    ) -> list[Frame]:
        """Get frames belonging to a specific state.

        Args:
            state: The state to get frames for
            frames: All available frames

        Returns:
            List of frames belonging to this state
        """
        if state.frame_indices:  # type: ignore[attr-defined]
            # Use explicit frame indices if provided
            frame_map = {f.frame_index: f for f in frames}
            return [frame_map[idx] for idx in state.frame_indices if idx in frame_map]  # type: ignore[attr-defined]
        else:
            # Use start/end range
            return [
                f
                for f in frames
                if state.start_frame_index <= f.frame_index <= state.end_frame_index  # type: ignore[attr-defined]
            ]

    def _get_click_locations(
        self,
        state: DetectedState,
        events: list[InputEvent],
    ) -> list[tuple[int, int]]:
        """Get click locations within a state's timeframe.

        Args:
            state: The state to get clicks for
            events: All input events

        Returns:
            List of (x, y) click coordinates
        """
        # If state has explicit click locations, use those
        if state.click_locations:  # type: ignore[attr-defined]
            return state.click_locations  # type: ignore[no-any-return,attr-defined]

        # Otherwise, extract from events
        # Note: This is a simplified implementation. In practice, you'd need
        # to match events to state timeframes based on timestamps.
        clicks = []
        for event in events:
            if event.event_type == "click" and event.x is not None and event.y is not None:
                clicks.append((event.x, event.y))

        return clicks

    def _extract_from_contours(
        self,
        frame: Frame,
        state: DetectedState,
        exclude_locations: list[tuple[int, int]],
    ) -> list[StateImage]:
        """Extract StateImages from contours, excluding click locations.

        Args:
            frame: Frame to analyze
            state: The state being processed
            exclude_locations: Click locations to exclude

        Returns:
            List of extracted StateImage objects
        """
        images = []

        try:
            # Detect contours
            bboxes = self.detect_contours(frame.image)

            # Filter out regions near click locations
            filtered_bboxes = []
            for bbox in bboxes:
                x, y, w, h = bbox
                center_x = x + w // 2
                center_y = y + h // 2

                # Check if too close to any click location
                too_close = False
                for click_x, click_y in exclude_locations:
                    distance = np.sqrt((center_x - click_x) ** 2 + (center_y - click_y) ** 2)
                    if distance < self.config.click_region_padding * 2:
                        too_close = True
                        break

                if not too_close:
                    filtered_bboxes.append(bbox)

            # Extract images from filtered bboxes
            for i, bbox in enumerate(filtered_bboxes[: self.config.max_contours]):
                x, y, w, h = bbox

                try:
                    # Use best_crop to refine the region
                    state_image = self.find_best_crop(frame.image, bbox)
                    state_image.name = f"{state.name}_contour_{i}"
                    state_image.source_frame_index = frame.frame_index  # type: ignore[attr-defined]
                    state_image.extraction_method = "contour"  # type: ignore[attr-defined]
                    images.append(state_image)
                except Exception as e:
                    logger.error("Error extracting contour %d: %s", i, e)

        except Exception as e:
            logger.error("Error in contour extraction: %s", e)

        return images

    def _find_occurrences(
        self,
        state_image: StateImage,
        frames: list[Frame],
    ) -> list[tuple[int, int]]:
        """Find all occurrences of an image across frames.

        Args:
            state_image: The image to search for
            frames: Frames to search in

        Returns:
            List of (x, y) positions where the image was found
        """
        occurrences = []

        try:
            # Template matching
            template = state_image.image

            # Convert template to grayscale for matching
            if len(template.shape) == 3:  # type: ignore[union-attr,call-overload]
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)  # type: ignore[call-overload]
            else:
                template_gray = template

            for frame in frames:
                try:
                    # Convert frame to grayscale
                    if len(frame.image.shape) == 3:
                        frame_gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
                    else:
                        frame_gray = frame.image

                    # Perform template matching
                    result = cv2.matchTemplate(
                        frame_gray,
                        template_gray,
                        cv2.TM_CCOEFF_NORMED,
                    )

                    # Find locations above threshold
                    locations = np.where(result >= state_image.similarity_threshold)

                    # Add locations
                    for pt in zip(*locations[::-1], strict=False):
                        occurrences.append(pt)

                except Exception as e:
                    logger.debug("Error matching in frame %d: %s", frame.frame_index, e)

        except Exception as e:
            logger.error("Error finding occurrences: %s", e)

        return occurrences


# ============================================================================
# Utility Functions
# ============================================================================


def save_state_image(state_image: StateImage, output_dir: Path) -> Path:
    """Save a StateImage to disk.

    Args:
        state_image: The StateImage to save
        output_dir: Directory to save to

    Returns:
        Path to the saved image file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename
    safe_name = state_image.name.replace("/", "_").replace("\\", "_")
    file_path = output_dir / f"{safe_name}.png"

    # Save image
    cv2.imwrite(str(file_path), state_image.image)  # type: ignore[call-overload]

    # Save context if available
    if state_image.context_image is not None:  # type: ignore[attr-defined]
        context_path = output_dir / f"{safe_name}_context.png"
        cv2.imwrite(str(context_path), state_image.context_image)  # type: ignore[attr-defined]

    logger.info("Saved StateImage to %s", file_path)
    return file_path


def load_state_image(file_path: Path, metadata_path: Path | None = None) -> StateImage:
    """Load a StateImage from disk.

    Args:
        file_path: Path to the image file
        metadata_path: Optional path to JSON metadata file

    Returns:
        StateImage object
    """
    import json

    # Load image
    image = cv2.imread(str(file_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {file_path}")

    # Load metadata if available
    metadata = {}
    if metadata_path and metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    # Extract bbox from image dimensions
    h, w = image.shape[:2]

    # Create StateImage
    state_image = StateImage(  # type: ignore[call-arg]
        name=file_path.stem,
        image=image,  # type: ignore[arg-type]
        bbox=metadata.get("bbox", (0, 0, w, h)),
        position_type=metadata.get("position_type", "unknown"),
        position=metadata.get("position", (0, 0)),
        similarity_threshold=metadata.get("similarity_threshold", 0.85),
        context_image=None,
        source_frame_index=metadata.get("source_frame_index"),
        extraction_method=metadata.get("extraction_method", "loaded"),
        metadata=metadata.get("metadata", {}),
    )

    return state_image
