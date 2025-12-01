"""RealFindImplementation - Performs actual template matching.

This is the real execution implementation in model-based GUI automation.
It performs actual image finding using OpenCV template matching.
"""

import logging
import time
from typing import Any

from ...config.framework_settings import FrameworkSettings
from ...find.matchers.template_matcher import TemplateMatcher
from ...find.screenshot.pure_actions_provider import PureActionsScreenshotProvider
from ...model.element import Pattern
from ...reporting.events import EventType, emit_event
from .find_options import FindOptions
from .find_result import FindResult
from .visual_debug import VisualDebugGenerator

logger = logging.getLogger(__name__)


class RealFindImplementation:
    """Performs actual template matching for find operations.

    Single Responsibility: Execute real image finding with template matching.

    Components:
    - PatternLoader: Loads image + mask
    - ScreenshotProvider: Captures screen
    - TemplateMatcher: Performs OpenCV matching
    - Event emission: Emits IMAGE_RECOGNITION events
    """

    def __init__(self):
        """Initialize real implementation with components."""
        self.screenshot_provider = PureActionsScreenshotProvider()
        self.template_matcher = TemplateMatcher()
        self.visual_debug = VisualDebugGenerator()

    def execute(
        self,
        pattern: Pattern,
        options: FindOptions,
    ) -> FindResult:
        """Execute real find operation.

        Args:
            pattern: Pattern to find
            options: Find configuration

        Returns:
            FindResult with actual matches from screen
        """
        # File-based debug logging at ENTRY point
        import os
        import tempfile
        from datetime import datetime

        debug_log = os.path.join(tempfile.gettempdir(), "qontinui_event_emission.log")
        try:
            with open(debug_log, "a", encoding="utf-8") as f:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                f.write(f"[{ts}] RealFindImplementation.execute() ENTRY\n")
                f.write(f"[{ts}]   pattern.id={pattern.id}, pattern.name={pattern.name}\n")
                f.write(f"[{ts}]   pattern.pixel_data is None: {pattern.pixel_data is None}\n")
                if pattern.pixel_data is not None:
                    f.write(f"[{ts}]   pattern.pixel_data.shape={pattern.pixel_data.shape}\n")
                f.write(
                    f"[{ts}]   FindOptions: similarity={options.similarity}, find_all={options.find_all}\n"
                )
        except Exception:
            pass

        logger.debug(
            f"[FIND_DEBUG] RealFindImplementation.execute() ENTRY - pattern.id={pattern.id}, pattern.name={pattern.name}"
        )
        logger.debug(f"[FIND_DEBUG] Pattern pixel_data is None: {pattern.pixel_data is None}")
        if pattern.pixel_data is not None:
            logger.debug(f"[FIND_DEBUG] Pattern pixel_data shape: {pattern.pixel_data.shape}")
        logger.debug(
            f"[FIND_DEBUG] FindOptions: similarity={options.similarity}, find_all={options.find_all}"
        )

        start_time = time.time()

        # Get debug setting from FrameworkSettings
        settings = FrameworkSettings.get_instance()
        collect_debug = options.collect_debug or settings.image_debug.emit_match_details

        logger.debug(f"[DEBUG_FEATURE] emit_match_details={collect_debug} for image {pattern.id}")

        try:
            # Capture screenshot and record timestamp
            logger.debug("[FIND_DEBUG] Calling screenshot_provider.capture()")
            screenshot_start_time = time.time()
            screenshot = self.screenshot_provider.capture(options.search_region)
            screenshot_timestamp = screenshot_start_time  # Save for event emission
            screenshot_end_time = time.time()
            screenshot_duration = screenshot_end_time - screenshot_start_time

            logger.debug(
                f"[FIND_DEBUG] Screenshot captured - shape: {screenshot.shape if hasattr(screenshot, 'shape') else 'unknown'}"
            )
            # Log screenshot timing
            logger.debug(
                f"[TIMING] Screenshot capture took {screenshot_duration:.3f}s ({screenshot_duration*1000:.1f}ms) for pattern {pattern.name}"
            )

            # Perform template matching with debug support
            logger.debug("[FIND_DEBUG] Calling template_matcher.find_matches_with_debug()")
            matching_start_time = time.time()
            matches, debug_data = self.template_matcher.find_matches_with_debug(
                screenshot=screenshot,
                pattern=pattern,
                find_all=options.find_all,
                similarity=options.similarity,
                search_region=options.search_region,
                collect_debug=collect_debug,
            )
            matching_end_time = time.time()
            matching_duration = matching_end_time - matching_start_time

            logger.debug(
                f"[FIND_DEBUG] Template matching returned {len(matches) if matches else 0} matches"
            )
            # Log matching timing
            logger.debug(
                f"[TIMING] Template matching took {matching_duration:.3f}s ({matching_duration*1000:.1f}ms) for pattern {pattern.name}"
            )

            duration_ms = (time.time() - start_time) * 1000

            # Log total find timing
            logger.debug(
                f"[TIMING] Total FIND operation took {duration_ms:.1f}ms for pattern {pattern.name}"
            )

            # Generate visual debug image if debug mode is enabled
            visual_debug_image = None
            if collect_debug and debug_data:
                logger.info("[REAL_FIND] Generating visual debug image")
                visual_debug_image = self.visual_debug.generate_debug_image(
                    screenshot=screenshot,
                    matches=matches,
                    debug_data=debug_data,
                    threshold=options.similarity,
                )
                if visual_debug_image:
                    logger.info("[REAL_FIND] Visual debug image generated successfully")
                else:
                    logger.warning("[REAL_FIND] Visual debug image generation returned None")

            # Emit IMAGE_RECOGNITION event (this makes debug work everywhere!)
            logger.debug("[FIND_DEBUG] Calling _emit_image_recognition_event()")
            self._emit_image_recognition_event(
                pattern=pattern,
                matches=matches,
                options=options,
                duration_ms=duration_ms,
                debug_data=debug_data if collect_debug else None,
                visual_debug_image=visual_debug_image,
                screenshot_timestamp=screenshot_timestamp,
                screenshot=screenshot,
            )
            logger.debug("[FIND_DEBUG] _emit_image_recognition_event() completed")

            return FindResult(
                matches=matches,
                found=len(matches) > 0,
                pattern_name=pattern.name,
                duration_ms=duration_ms,
                debug_data=debug_data if collect_debug else None,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Error finding pattern {pattern.name}: {e}", exc_info=True)

            # Emit error event
            self._emit_image_recognition_event(
                pattern=pattern,
                matches=[],
                options=options,
                duration_ms=duration_ms,
                debug_data={"error": str(e)},
            )

            return FindResult(
                matches=[],
                found=False,
                pattern_name=pattern.name,
                duration_ms=duration_ms,
                debug_data={"error": str(e)},
            )

    def _emit_image_recognition_event(
        self,
        pattern: Pattern,
        matches: list,
        options: FindOptions,
        duration_ms: float,
        debug_data: dict[str, Any] | None,
        visual_debug_image: str | None = None,
        screenshot_timestamp: float | None = None,
        screenshot: Any | None = None,
    ) -> None:
        """Emit IMAGE_RECOGNITION event for runner to display.

        This event is what the runner uses to populate the Images tab.
        By emitting it for ALL find operations, debug info works everywhere.

        Args:
            pattern: Pattern that was searched for
            matches: List of Match objects found
            options: Find options used
            duration_ms: Time taken in milliseconds
            debug_data: Debug information dictionary
            visual_debug_image: Base64-encoded PNG image with annotated matches
            screenshot_timestamp: Unix timestamp when screenshot was captured
            screenshot: Screenshot image data (NumPy array or PIL Image) for encoding
        """
        # File-based debug logging (works even when console logging is disabled)
        import os
        import tempfile
        from datetime import datetime

        debug_log = os.path.join(tempfile.gettempdir(), "qontinui_event_emission.log")
        try:
            with open(debug_log, "a", encoding="utf-8") as f:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                f.write(f"[{ts}] _emit_image_recognition_event CALLED\n")
                f.write(f"[{ts}]   pattern.id={pattern.id}, pattern.name={pattern.name}\n")
                f.write(f"[{ts}]   matches count={len(matches)}\n")
                f.write(f"[{ts}]   found={len(matches) > 0}\n")
        except Exception:
            pass
        # Get template size from pattern
        template_size: tuple[int, int] = (0, 0)
        if hasattr(pattern, "pixel_data") and pattern.pixel_data is not None:
            template_size = pattern.pixel_data.shape[:2][::-1]  # type: ignore[assignment]

        # Get pattern name with fallback - use pattern.id as it's the filename
        # pattern.name might be empty, but pattern.id contains the actual filename
        pattern_name = pattern.name if pattern.name and pattern.name.strip() else pattern.id

        logger.debug(f"Pattern name='{pattern.name}', id='{pattern.id}', using='{pattern_name}'")

        # Build location with full region info (x, y, width, height)
        location = None
        if matches:
            region = matches[0].target.region
            location = {
                "x": region.x,
                "y": region.y,
                "width": region.width,
                "height": region.height,
            }

        # Build event data
        event_data = {
            "image_id": pattern.id,
            "pattern_name": pattern_name,
            "image_path": pattern_name,  # Frontend expects image_path for display name
            "template_size": template_size,
            "threshold": options.similarity,
            "found": len(matches) > 0,
            "confidence": matches[0].similarity if matches else 0.0,
            "location": location,
            "duration_ms": duration_ms,
            "screenshot_timestamp": screenshot_timestamp,  # Unix timestamp when screenshot was captured
        }

        # Debug: Log that we're including screenshot_timestamp
        logger.debug(f"[EVENT] Including screenshot_timestamp: {screenshot_timestamp}")

        # Add screenshot as base64 PNG
        if screenshot is not None:
            logger.info(f"[EVENT] Encoding screenshot for pattern {pattern.id}")
            screenshot_image = self._encode_template_image(screenshot)
            if screenshot_image:
                logger.info(
                    f"[EVENT] Screenshot encoded successfully, length={len(screenshot_image)}"
                )
                event_data["screenshot_base64"] = screenshot_image
            else:
                logger.warning(
                    f"[EVENT] Screenshot encoding returned None for pattern {pattern.id}"
                )

        # Add template image as separate base64 PNG
        if hasattr(pattern, "pixel_data") and pattern.pixel_data is not None:
            logger.info(f"[EVENT] Encoding template image for pattern {pattern.id}")
            template_image = self._encode_template_image(pattern.pixel_data)
            if template_image:
                logger.info(
                    f"[EVENT] Template image encoded successfully, length={len(template_image)}"
                )
                event_data["template_image"] = template_image
                event_data["image_data"] = (
                    template_image  # Frontend expects image_data for template display
                )
            else:
                logger.warning(
                    f"[EVENT] Template image encoding returned None for pattern {pattern.id}"
                )

        # Add debug data if available
        if debug_data:
            event_data["debug"] = debug_data

        # Add visual debug image if available
        if visual_debug_image:
            event_data["visual_debug_image"] = visual_debug_image
            event_data["debug_visual_base64"] = visual_debug_image  # Alias for spec compliance

        # Add timestamp
        event_data["timestamp"] = time.time()

        # Emit MATCH_ATTEMPTED event (EventTranslator listens for this)
        logger.debug(f"[FIND_DEBUG] Emitting MATCH_ATTEMPTED event for pattern {pattern.id}")

        # File-based debug logging
        try:
            with open(debug_log, "a", encoding="utf-8") as f:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                f.write(f"[{ts}] About to call emit_event(EventType.MATCH_ATTEMPTED, ...)\n")
                f.write(f"[{ts}]   EventType.MATCH_ATTEMPTED={EventType.MATCH_ATTEMPTED}\n")
                f.write(f"[{ts}]   event_data keys={list(event_data.keys())}\n")
        except Exception:
            pass

        emit_event(EventType.MATCH_ATTEMPTED, event_data)

        # File-based debug logging
        try:
            with open(debug_log, "a", encoding="utf-8") as f:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                f.write(f"[{ts}] emit_event() call completed successfully\n")
        except Exception:
            pass

        logger.info(
            f"[EventTranslator] MATCH_ATTEMPTED event emitted successfully for pattern {pattern.id}"
        )

    def _encode_template_image(self, template: Any) -> str | None:
        """Encode template as base64 PNG for display.

        Args:
            template: Template image data (PIL Image or NumPy array)

        Returns:
            Base64-encoded PNG string, or None on error
        """
        try:
            import base64

            import cv2
            import numpy as np
            from PIL import Image

            # Convert PIL Image to NumPy if needed
            if isinstance(template, Image.Image):
                template = np.array(template)
                template = cv2.cvtColor(template, cv2.COLOR_RGB2BGR)

            # Validate
            if template is None or template.size == 0:
                return None

            # Scale template for visibility (minimum 100px on larger dimension)
            t_h, t_w = template.shape[:2]
            min_display_size = 100
            max_display_size = 200

            scale_factor = 1.0
            larger_dimension = max(t_w, t_h)

            if larger_dimension < min_display_size:
                scale_factor = min_display_size / larger_dimension
            elif larger_dimension > max_display_size:
                scale_factor = max_display_size / larger_dimension

            if scale_factor != 1.0:
                new_w = int(t_w * scale_factor)
                new_h = int(t_h * scale_factor)
                interpolation = cv2.INTER_CUBIC if scale_factor > 1.0 else cv2.INTER_AREA
                template = cv2.resize(template, (new_w, new_h), interpolation=interpolation)

            # Encode as PNG
            success, buffer = cv2.imencode(".png", template)
            if not success:
                return None

            # Convert to base64
            png_bytes = buffer.tobytes()
            base64_str = base64.b64encode(png_bytes).decode("utf-8")

            return base64_str

        except Exception as e:
            logger.error(f"Failed to encode template image: {e}")
            return None

    async def execute_async(
        self,
        patterns: list[Pattern],
        options: FindOptions,
        max_concurrent: int = 15,
    ) -> list[FindResult]:
        """Execute async find operations for multiple patterns.

        Args:
            patterns: List of patterns to find
            options: Find configuration
            max_concurrent: Maximum concurrent pattern searches

        Returns:
            List of FindResults for each pattern
        """
        import asyncio

        logger.info(f"Executing async find for {len(patterns)} patterns")

        # Create tasks for each pattern
        tasks = [asyncio.to_thread(self.execute, pattern, options) for pattern in patterns]

        # Execute all searches concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to empty FindResults
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error finding pattern {patterns[i].name}: {result}")
                final_results.append(
                    FindResult(
                        matches=[],
                        found=False,
                        pattern_name=patterns[i].name,
                        duration_ms=0.0,
                        debug_data={"error": str(result)},
                    )
                )
            else:
                final_results.append(result)  # type: ignore[arg-type]

        return final_results
