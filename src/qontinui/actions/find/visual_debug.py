"""Visual debug image generator for image matching.

Creates annotated screenshots showing:
- Found matches (green boxes)
- Top non-matches (red boxes)
- Confidence scores
- Template image for reference

Internally uses composable annotators from ``qontinui.find.annotators``
for the ``Detections``-based code path while preserving the legacy
``Match``/``debug_data`` API for backwards compatibility.
"""

import base64
import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VisualDebugGenerator:
    """Generates visual debug images for template matching.

    Single Responsibility: Create annotated images showing match results.
    """

    def __init__(self):
        """Initialize generator."""
        # Color constants (BGR format for OpenCV)
        self.COLOR_MATCH = (0, 255, 0)  # Green for matches
        self.COLOR_NON_MATCH = (0, 0, 255)  # Red for non-matches
        self.COLOR_TEXT_BG = (0, 0, 0)  # Black text background
        self.COLOR_TEXT = (255, 255, 255)  # White text

        # Visual settings
        self.BOX_THICKNESS = 2
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE = 0.5
        self.FONT_THICKNESS = 1
        self.TEXT_PADDING = 5

    def _safe_int(self, value: Any, name: str = "value") -> int:
        """Safely convert a value to int, with detailed error logging."""
        try:
            result = int(round(float(value)))
            return result
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to convert {name}={value} (type={type(value)}) to int: {e}")
            return 0

    # ------------------------------------------------------------------
    # Detections-based API (uses composable annotators)
    # ------------------------------------------------------------------

    def generate_from_detections(
        self,
        screenshot: np.ndarray,
        detections: Any,
        *,
        confidence_threshold: float = 0.8,
    ) -> str | None:
        """Generate annotated debug image from a ``Detections`` container.

        Uses the composable annotator pipeline: bounding boxes, labels,
        and confidence bars are drawn in sequence.

        Args:
            screenshot: BGR image (numpy array).
            detections: A ``Detections`` container from ``qontinui.find.detections``.
            confidence_threshold: Confidence dividing match/non-match colours.

        Returns:
            Base64-encoded PNG, or ``None`` on error.
        """
        try:
            from ...find.annotators import (
                BoundingBoxAnnotator,
                ConfidenceBarAnnotator,
                LabelAnnotator,
            )

            if screenshot is None or detections is None or detections.is_empty():
                return None

            scene = self._prepare_scene(screenshot)
            if scene is None:
                return None

            box_ann = BoundingBoxAnnotator(
                color_match=self.COLOR_MATCH,
                color_non_match=self.COLOR_NON_MATCH,
                confidence_threshold=confidence_threshold,
                thickness=self.BOX_THICKNESS,
            )
            label_ann = LabelAnnotator(
                color_text=self.COLOR_TEXT,
                color_background=self.COLOR_TEXT_BG,
                font_scale=self.FONT_SCALE,
                thickness=self.FONT_THICKNESS,
                padding=self.TEXT_PADDING,
            )
            bar_ann = ConfidenceBarAnnotator()

            scene = box_ann.annotate(scene, detections)
            scene = label_ann.annotate(scene, detections)
            scene = bar_ann.annotate(scene, detections)

            self._draw_threshold_info(scene, confidence_threshold, len(detections))
            return self._encode_image(scene)
        except Exception as e:
            logger.error(f"Failed to generate visual debug image: {e}", exc_info=True)
            return None

    @staticmethod
    def annotate_detections(
        screenshot: np.ndarray,
        detections: Any,
        annotators: list | None = None,
    ) -> np.ndarray:
        """Apply a list of annotators to a screenshot.

        Convenience method for ad-hoc annotator composition without
        encoding to base64.

        Args:
            screenshot: BGR image (numpy array).
            detections: A ``Detections`` container.
            annotators: Ordered list of ``Annotator`` instances.
                If ``None``, uses ``[BoundingBoxAnnotator(), LabelAnnotator()]``.

        Returns:
            The annotated image.
        """
        if annotators is None:
            from ...find.annotators import BoundingBoxAnnotator, LabelAnnotator

            annotators = [BoundingBoxAnnotator(), LabelAnnotator()]

        scene = screenshot.copy()
        for ann in annotators:
            scene = ann.annotate(scene, detections)
        return scene

    # ------------------------------------------------------------------
    # Legacy API (preserved for backwards compatibility)
    # ------------------------------------------------------------------

    def generate_debug_image(
        self,
        screenshot: np.ndarray,
        matches: list[Any],
        debug_data: dict[str, Any],
        threshold: float,
    ) -> str | None:
        """Generate annotated debug image.

        Args:
            screenshot: The screenshot that was searched (BGR format)
            matches: List of Match objects that passed threshold
            debug_data: Debug data with top_matches info
            threshold: Similarity threshold used

        Returns:
            Base64-encoded PNG image, or None on error
        """
        try:
            logger.debug("[VISUAL_DEBUG] Starting visual debug image generation")

            scene = self._prepare_scene(screenshot)
            if scene is None:
                return None

            try:
                num_matches = len(matches)
                logger.debug(f"[VISUAL_DEBUG] Number of matches: {num_matches}")
            except Exception as e:
                logger.debug(f"[VISUAL_DEBUG] ERROR getting len(matches): {e}")
                return None

            annotated = scene

            # IMPORTANT: Draw in order from worst to best so best matches appear on top
            # Step 1: Draw debug matches (cyan and red) from debug data FIRST
            # Step 2: Draw actual returned matches (green) LAST so they appear on top
            if debug_data and "top_matches" in debug_data:
                top_matches = debug_data["top_matches"]
                logger.info(
                    f"[VISUAL_DEBUG] Drawing non-match boxes from {len(top_matches)} top matches"
                )
                logger.info(f"[VISUAL_DEBUG] Threshold for red boxes: {threshold}")
                logger.info(f"[VISUAL_DEBUG] Number of green boxes drawn: {len(matches)}")

                # Build a set of match coordinates to identify which top_matches were returned
                returned_match_coords = set()
                for match in matches:
                    x = int(match.target.region.x)
                    y = int(match.target.region.y)
                    returned_match_coords.add((x, y))

                # Sort top_matches by confidence (ascending) so we draw worst first, best last
                sorted_top_matches = sorted(top_matches, key=lambda tm: tm.get("confidence", 0.0))

                for idx, top_match in enumerate(sorted_top_matches):
                    try:
                        confidence = top_match.get("confidence", 0.0)
                        location = top_match.get("location", {})
                        x = self._safe_int(location.get("x", 0), f"top_match[{idx}].x")
                        y = self._safe_int(location.get("y", 0), f"top_match[{idx}].y")

                        logger.info(
                            f"[VISUAL_DEBUG] Top match #{idx + 1}: confidence={confidence}, threshold={threshold}, coords=({x},{y})"
                        )

                        # Skip if this match was already drawn as a green box
                        if (x, y) in returned_match_coords:
                            logger.info(
                                f"[VISUAL_DEBUG] Skipping top match #{idx + 1} - already drawn as green box"
                            )
                            continue

                        # Determine color: cyan if above threshold, red if below
                        if confidence >= threshold:
                            box_color = (
                                255,
                                255,
                                0,
                            )  # Cyan for "above threshold but not returned"
                            label_text = f"NOT RETURNED {confidence:.1%}"
                        else:
                            box_color = self.COLOR_NON_MATCH  # Red for below threshold
                            label_text = f"NO MATCH {confidence:.1%}"

                        # Get template size from debug data
                        template_size = debug_data.get("template_size", {"width": 0, "height": 0})
                        w = self._safe_int(template_size.get("width", 0), "template.width")
                        h = self._safe_int(template_size.get("height", 0), "template.height")

                        logger.debug(
                            f"[VISUAL_DEBUG] Drawing {label_text} at ({x}, {y}): w={w}, h={h}, conf={confidence:.3f}"
                        )

                        # Validate coordinates
                        if w <= 0 or h <= 0:
                            logger.warning(
                                f"[VISUAL_DEBUG] Skipping match {idx} - invalid dimensions"
                            )
                            continue

                        # Create points as pure Python tuples
                        pt1 = (x, y)
                        pt2 = (x + w, y + h)

                        # Draw box
                        cv2.rectangle(
                            annotated,
                            pt1,
                            pt2,
                            box_color,
                            self.BOX_THICKNESS,
                        )

                        # Draw confidence label
                        self._draw_label(
                            annotated,
                            x,
                            y - 5,
                            label_text,
                            box_color,
                        )
                    except (ValueError, AttributeError, TypeError, KeyError) as e:
                        logger.warning(f"Skipping non-match box due to error: {e}")

            # Step 2: Draw actual returned matches (green) LAST so they appear on top
            logger.debug(f"[VISUAL_DEBUG] Drawing {num_matches} green match boxes on top")
            for idx, match in enumerate(matches):
                try:
                    raw_x = match.target.region.x
                    raw_y = match.target.region.y
                    raw_w = match.target.region.w
                    raw_h = match.target.region.h

                    # Safely convert all coordinates to integers
                    x = self._safe_int(raw_x, f"match[{idx}].x")
                    y = self._safe_int(raw_y, f"match[{idx}].y")
                    w = self._safe_int(raw_w, f"match[{idx}].w")
                    h = self._safe_int(raw_h, f"match[{idx}].h")
                    confidence = match.similarity

                    logger.debug(f"[VISUAL_DEBUG] Match {idx}: x={x}, y={y}, w={w}, h={h}")

                    # Validate coordinates
                    if w <= 0 or h <= 0:
                        logger.debug(
                            f"[VISUAL_DEBUG] Skipping match {idx} - invalid dimensions w={w}, h={h}"
                        )
                        continue

                    # Create points as pure Python tuples of ints
                    pt1 = (x, y)
                    pt2 = (x + w, y + h)

                    # Draw green box
                    cv2.rectangle(annotated, pt1, pt2, self.COLOR_MATCH, self.BOX_THICKNESS)

                    # Draw confidence label
                    self._draw_label(
                        annotated,
                        x,
                        y - 5,
                        f"MATCH {confidence:.1%}",
                        self.COLOR_MATCH,
                    )
                except (ValueError, AttributeError, TypeError) as e:
                    logger.warning(f"Skipping match box due to error: {e}")

            # Add threshold line at top
            logger.info("[VISUAL_DEBUG] Drawing threshold info")
            self._draw_threshold_info(annotated, threshold, len(matches))

            # Encode to base64 PNG
            logger.info("[VISUAL_DEBUG] Encoding image to base64")
            result = self._encode_image(annotated)
            logger.info("[VISUAL_DEBUG] Visual debug image generated successfully")
            return result

        except Exception as e:
            logger.error(f"Failed to generate visual debug image: {e}", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _prepare_scene(self, screenshot: np.ndarray) -> np.ndarray | None:
        """Validate and copy the screenshot, converting from PIL if needed."""
        if screenshot is None:
            logger.debug("[VISUAL_DEBUG] screenshot is None")
            return None

        from PIL import Image

        if isinstance(screenshot, Image.Image):
            logger.debug("[VISUAL_DEBUG] Converting PIL Image to NumPy array")
            screenshot = np.array(screenshot)
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

        try:
            shape = screenshot.shape
            logger.debug(f"[VISUAL_DEBUG] Screenshot shape: {shape}")
        except Exception as e:
            logger.debug(f"[VISUAL_DEBUG] ERROR accessing screenshot.shape: {e}")
            return None

        try:
            logger.debug(f"[VISUAL_DEBUG] Screenshot dtype: {screenshot.dtype}")
        except Exception as e:
            logger.debug(f"[VISUAL_DEBUG] ERROR accessing screenshot.dtype: {e}")
            return None

        try:
            annotated = screenshot.copy()
        except Exception as e:
            logger.debug(f"[VISUAL_DEBUG] ERROR copying screenshot: {e}")
            return None

        if annotated is None or annotated.size == 0:
            logger.debug("[VISUAL_DEBUG] Invalid screenshot - returning None")
            return None

        return annotated

    def _draw_label(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        text: str,
        color: tuple[int, int, int],
    ) -> None:
        """Draw text label with background.

        Args:
            image: Image to draw on
            x: X position
            y: Y position
            text: Text to draw
            color: Text color (BGR)
        """
        # Ensure coordinates are integers
        x = int(round(x))
        y = int(round(y))

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.FONT, self.FONT_SCALE, self.FONT_THICKNESS
        )

        # Draw black background (all coordinates must be int)
        cv2.rectangle(
            image,
            (x, y - text_height - self.TEXT_PADDING),
            (x + text_width + self.TEXT_PADDING, y),
            self.COLOR_TEXT_BG,
            -1,  # Filled
        )

        # Draw white text
        cv2.putText(
            image,
            text,
            (x + 2, y - 2),
            self.FONT,
            self.FONT_SCALE,
            self.COLOR_TEXT,
            self.FONT_THICKNESS,
            cv2.LINE_AA,
        )

    def _draw_threshold_info(self, image: np.ndarray, threshold: float, match_count: int) -> None:
        """Draw threshold and match count info at top of image.

        Args:
            image: Image to draw on
            threshold: Threshold value
            match_count: Number of matches found
        """
        try:
            info_text = f"Threshold: {threshold:.1%} | Matches: {match_count}"

            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                info_text, self.FONT, self.FONT_SCALE, self.FONT_THICKNESS
            )

            # Draw at top center - ensure integers
            x = int((image.shape[1] - text_width) // 2)
            y = int(text_height + self.TEXT_PADDING)

            # Draw background
            cv2.rectangle(
                image,
                (x - self.TEXT_PADDING, 0),
                (x + text_width + self.TEXT_PADDING, y + self.TEXT_PADDING),
                self.COLOR_TEXT_BG,
                -1,
            )

            # Draw text
            cv2.putText(
                image,
                info_text,
                (x, y),
                self.FONT,
                self.FONT_SCALE,
                self.COLOR_TEXT,
                self.FONT_THICKNESS,
                cv2.LINE_AA,
            )
        except Exception as e:
            logger.warning(f"Failed to draw threshold info: {e}")

    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image as base64 PNG.

        Args:
            image: Image to encode (BGR format)

        Returns:
            Base64-encoded PNG string
        """
        # Encode as PNG
        success, buffer = cv2.imencode(".png", image)

        if not success:
            raise RuntimeError("Failed to encode image as PNG")

        # Convert to base64
        png_bytes = buffer.tobytes()
        base64_str = base64.b64encode(png_bytes).decode("utf-8")

        return base64_str
