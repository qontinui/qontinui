"""Visual debug image generator for image matching.

Creates annotated screenshots showing:
- Found matches (green boxes)
- Top non-matches (red boxes)
- Confidence scores
- Template image for reference
"""

import base64
from typing import Any

import cv2
import numpy as np


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
        import logging

        logger = logging.getLogger(__name__)

        try:
            result = int(round(float(value)))
            return result
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to convert {name}={value} (type={type(value)}) to int: {e}")
            return 0

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
        import logging
        import os
        import tempfile

        logger = logging.getLogger(__name__)

        # Create a debug log file
        debug_log_path = os.path.join(tempfile.gettempdir(), "qontinui_visual_debug.log")

        def log_to_file(msg: str):
            """Write to both logger and debug file."""
            logger.info(msg)
            try:
                with open(debug_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{msg}\n")
            except Exception:
                pass

        try:
            log_to_file(
                f"[VISUAL_DEBUG] Starting visual debug image generation (log: {debug_log_path})"
            )

            # Check screenshot validity before any operations
            log_to_file("[VISUAL_DEBUG] Step 1: Checking if screenshot is None")
            if screenshot is None:
                log_to_file("[VISUAL_DEBUG] ERROR: screenshot is None")
                return None

            log_to_file(f"[VISUAL_DEBUG] Step 2: screenshot type = {type(screenshot)}")

            # Convert PIL Image to NumPy array if needed
            from PIL import Image

            if isinstance(screenshot, Image.Image):
                log_to_file("[VISUAL_DEBUG] Step 2.5: Converting PIL Image to NumPy array")
                screenshot = np.array(screenshot)
                # PIL uses RGB, OpenCV uses BGR - convert
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
                log_to_file(f"[VISUAL_DEBUG] Converted to NumPy array, shape: {screenshot.shape}")

            try:
                log_to_file("[VISUAL_DEBUG] Step 3: Accessing screenshot.shape")
                shape = screenshot.shape
                log_to_file(f"[VISUAL_DEBUG] Screenshot shape: {shape}")
            except Exception as e:
                log_to_file(f"[VISUAL_DEBUG] ERROR accessing screenshot.shape: {e}")
                return None

            try:
                log_to_file("[VISUAL_DEBUG] Step 4: Accessing screenshot.dtype")
                dtype = screenshot.dtype
                log_to_file(f"[VISUAL_DEBUG] Screenshot dtype: {dtype}")
            except Exception as e:
                log_to_file(f"[VISUAL_DEBUG] ERROR accessing screenshot.dtype: {e}")
                return None

            try:
                log_to_file("[VISUAL_DEBUG] Step 5: Getting number of matches")
                num_matches = len(matches)
                log_to_file(f"[VISUAL_DEBUG] Number of matches: {num_matches}")
            except Exception as e:
                log_to_file(f"[VISUAL_DEBUG] ERROR getting len(matches): {e}")
                return None

            try:
                log_to_file("[VISUAL_DEBUG] Step 6: Copying screenshot")
                annotated = screenshot.copy()
                log_to_file(
                    f"[VISUAL_DEBUG] Screenshot copied successfully, annotated type: {type(annotated)}"
                )
            except Exception as e:
                log_to_file(f"[VISUAL_DEBUG] ERROR copying screenshot: {e}")
                return None

            # Validate screenshot
            if annotated is None or annotated.size == 0:
                log_to_file("[VISUAL_DEBUG] Invalid screenshot - returning None")
                return None

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
                            f"[VISUAL_DEBUG] Top match #{idx+1}: confidence={confidence}, threshold={threshold}, coords=({x},{y})"
                        )

                        # Skip if this match was already drawn as a green box
                        if (x, y) in returned_match_coords:
                            logger.info(
                                f"[VISUAL_DEBUG] Skipping top match #{idx+1} - already drawn as green box"
                            )
                            continue

                        # Determine color: cyan if above threshold, red if below
                        if confidence >= threshold:
                            box_color = (255, 255, 0)  # Cyan for "above threshold but not returned"
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
                        import logging

                        logger = logging.getLogger(__name__)
                        logger.warning(f"Skipping non-match box due to error: {e}")

            # Step 2: Draw actual returned matches (green) LAST so they appear on top
            log_to_file(f"[VISUAL_DEBUG] Drawing {num_matches} green match boxes on top")
            for idx, match in enumerate(matches):
                try:
                    # Get raw values first for logging
                    raw_x = match.target.region.x
                    raw_y = match.target.region.y
                    raw_w = match.target.region.w
                    raw_h = match.target.region.h

                    log_to_file(
                        f"[VISUAL_DEBUG] Match {idx} RAW: x={raw_x} (type={type(raw_x).__name__}), y={raw_y} (type={type(raw_y).__name__}), w={raw_w} (type={type(raw_w).__name__}), h={raw_h} (type={type(raw_h).__name__})"
                    )

                    # Safely convert all coordinates to integers
                    x = self._safe_int(raw_x, f"match[{idx}].x")
                    y = self._safe_int(raw_y, f"match[{idx}].y")
                    w = self._safe_int(raw_w, f"match[{idx}].w")
                    h = self._safe_int(raw_h, f"match[{idx}].h")
                    confidence = match.similarity

                    log_to_file(
                        f"[VISUAL_DEBUG] Match {idx} CONVERTED: x={x} (type={type(x).__name__}), y={y}, w={w}, h={h}"
                    )

                    # Validate coordinates
                    if w <= 0 or h <= 0:
                        log_to_file(
                            f"[VISUAL_DEBUG] Skipping match {idx} - invalid dimensions w={w}, h={h}"
                        )
                        continue

                    # Create points as pure Python tuples of ints
                    pt1 = (x, y)
                    pt2 = (x + w, y + h)

                    log_to_file(
                        f"[VISUAL_DEBUG] About to call cv2.rectangle with: pt1={pt1} (types={type(pt1[0]).__name__},{type(pt1[1]).__name__}), pt2={pt2} (types={type(pt2[0]).__name__},{type(pt2[1]).__name__})"
                    )

                    # Draw green box
                    cv2.rectangle(annotated, pt1, pt2, self.COLOR_MATCH, self.BOX_THICKNESS)

                    log_to_file(f"[VISUAL_DEBUG] Successfully drew rectangle for match {idx}")

                    # Draw confidence label
                    self._draw_label(
                        annotated,
                        x,
                        y - 5,
                        f"MATCH {confidence:.1%}",
                        self.COLOR_MATCH,
                    )
                except (ValueError, AttributeError, TypeError) as e:
                    import logging

                    logger = logging.getLogger(__name__)
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
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Failed to generate visual debug image: {e}", exc_info=True)
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

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
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to draw threshold info: {e}")

    def _draw_template_overlay(self, image: np.ndarray, template: np.ndarray, log_to_file) -> None:
        """Draw template image overlay in top-left corner.

        Args:
            image: Image to draw on
            template: Template image to display
            log_to_file: Logging function for file-based logging
        """
        try:
            import logging

            logger = logging.getLogger(__name__)

            log_to_file("[VISUAL_DEBUG] _draw_template_overlay called")
            log_to_file(f"[VISUAL_DEBUG] Template type: {type(template)}")
            log_to_file(f"[VISUAL_DEBUG] Template is None: {template is None}")

            if template is None:
                log_to_file("[VISUAL_DEBUG] Template is None, skipping overlay")
                return

            if template.size == 0:
                log_to_file("[VISUAL_DEBUG] Template size is 0, skipping overlay")
                return

            # Get template dimensions
            t_h, t_w = template.shape[:2]
            log_to_file(f"[VISUAL_DEBUG] Template dimensions: {t_w}x{t_h}")

            # Scale template for visibility - minimum 100px on the larger dimension
            min_display_size = 100
            max_display_size = 200

            scale_factor = 1.0
            larger_dimension = max(t_w, t_h)

            if larger_dimension < min_display_size:
                # Scale up small templates for visibility
                scale_factor = min_display_size / larger_dimension
                log_to_file(f"[VISUAL_DEBUG] Template too small, scaling up by {scale_factor:.2f}x")
            elif larger_dimension > max_display_size:
                # Scale down large templates
                scale_factor = max_display_size / larger_dimension
                log_to_file(
                    f"[VISUAL_DEBUG] Template too large, scaling down by {scale_factor:.2f}x"
                )

            if scale_factor != 1.0:
                new_w = int(t_w * scale_factor)
                new_h = int(t_h * scale_factor)
                interpolation = cv2.INTER_CUBIC if scale_factor > 1.0 else cv2.INTER_AREA
                template = cv2.resize(template, (new_w, new_h), interpolation=interpolation)
                t_h, t_w = new_h, new_w
                log_to_file(f"[VISUAL_DEBUG] Scaled template to {t_w}x{t_h}")

            # Position in top-left with padding
            padding = 10
            border = 3
            x_pos = padding
            y_pos = padding + 40  # Below threshold info

            log_to_file(f"[VISUAL_DEBUG] Template position: x={x_pos}, y={y_pos}")

            # Ensure template fits in image
            img_h, img_w = image.shape[:2]
            log_to_file(f"[VISUAL_DEBUG] Image dimensions: {img_w}x{img_h}")

            # Draw white border background
            log_to_file("[VISUAL_DEBUG] Drawing white border background")
            cv2.rectangle(
                image,
                (x_pos - border, y_pos - border),
                (x_pos + t_w + border, y_pos + t_h + border),
                self.COLOR_TEXT,  # White
                -1,  # Filled
            )

            # Draw black inner border
            log_to_file("[VISUAL_DEBUG] Drawing black inner border")
            cv2.rectangle(
                image,
                (x_pos - 1, y_pos - 1),
                (x_pos + t_w + 1, y_pos + t_h + 1),
                self.COLOR_TEXT_BG,  # Black
                1,
            )

            # Overlay template image
            log_to_file(
                f"[VISUAL_DEBUG] Overlaying template at position [{y_pos}:{y_pos + t_h}, {x_pos}:{x_pos + t_w}]"
            )
            image[y_pos : y_pos + t_h, x_pos : x_pos + t_w] = template
            log_to_file("[VISUAL_DEBUG] Template overlaid successfully")

            # Add label above template
            label = "Template"
            label_y = y_pos - border - 5
            log_to_file(f"[VISUAL_DEBUG] Adding label '{label}' at position ({x_pos}, {label_y})")
            self._draw_label(image, x_pos, label_y, label, (255, 255, 0))  # Yellow

            log_to_file("[VISUAL_DEBUG] Template overlay drawn successfully")

        except Exception as e:
            import logging
            import traceback

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to draw template overlay: {e}")
            log_to_file(f"[VISUAL_DEBUG] ERROR in template overlay: {e}")
            log_to_file(f"[VISUAL_DEBUG] Traceback: {traceback.format_exc()}")

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
