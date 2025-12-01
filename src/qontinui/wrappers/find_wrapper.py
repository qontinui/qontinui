"""FindWrapper - Routes pattern finding to mock or real implementations (Brobot pattern).

This wrapper provides the routing layer for pattern finding operations,
delegating to either MockFind (historical playback) or HAL implementations
(real screen capture + OpenCV matching) based on ExecutionMode.

Architecture:
    FindImage/StateDetector (high-level)
      ↓
    FindWrapper (this layer) ← Routes based on ExecutionMode
      ↓
    ├─ if mock → MockFind → ActionHistory → Returns List[Match]
    └─ if real → HAL Layer → MSSScreenCapture + OpenCVMatcher → Returns List[Match]
"""

import logging
from typing import TYPE_CHECKING, Optional

from ..model.element.region import Region
from .base import BaseWrapper

if TYPE_CHECKING:
    from ..actions.action_result import ActionResult
    from ..model.element.pattern import Pattern
    from ..model.match.match import Match

logger = logging.getLogger(__name__)


class FindWrapper(BaseWrapper):
    """Wrapper for pattern finding operations.

    Routes find operations to either mock or real implementations based on
    ExecutionMode. This follows the Brobot pattern where high-level code
    is agnostic to whether it's running in mock or real mode.

    Example:
        # Initialize wrapper
        wrapper = FindWrapper()

        # Find pattern (automatically routed to mock or real)
        matches = wrapper.find_all(pattern, search_region)

        # High-level code doesn't know or care whether this used:
        # - MockFind.get_matches(pattern) → pattern.match_history → List[Match]
        # - HAL screen capture + OpenCV template matching → List[Match]

    Attributes:
        mock_find: MockFind instance for historical playback
        hal_matcher: OpenCV matcher for real pattern matching
        hal_capture: Screen capture for real mode
    """

    def __init__(self) -> None:
        """Initialize FindWrapper.

        Sets up both mock and real implementations. The actual implementation
        used is determined at runtime based on ExecutionMode.
        """
        super().__init__()

        # Lazy initialization to avoid circular imports
        self._mock_find = None
        self._hal_matcher = None
        self._hal_capture = None

        logger.debug("FindWrapper initialized")

    @property
    def mock_find(self):
        """Get MockFind instance (lazy initialization).

        Returns:
            MockFind instance
        """
        if self._mock_find is None:
            from ..mock.mock_find import MockFind

            self._mock_find = MockFind()
            logger.debug("MockFind initialized")
        return self._mock_find

    @property
    def hal_matcher(self):
        """Get HAL pattern matcher (lazy initialization).

        Returns:
            IPatternMatcher implementation (OpenCVMatcher)
        """
        if self._hal_matcher is None:
            from ..hal.factory import HALFactory

            self._hal_matcher = HALFactory.get_pattern_matcher()
            logger.debug("HAL pattern matcher initialized")
        return self._hal_matcher

    @property
    def hal_capture(self):
        """Get HAL screen capture (lazy initialization).

        Returns:
            IScreenCapture implementation (MSSScreenCapture)
        """
        if self._hal_capture is None:
            from ..hal.factory import HALFactory

            self._hal_capture = HALFactory.get_screen_capture()
            logger.debug("HAL screen capture initialized")
        return self._hal_capture

    def find(
        self,
        pattern: "Pattern",
        search_region: Region | None = None,
    ) -> "ActionResult":
        """Find a pattern (single match).

        Routes to MockFind or HAL based on ExecutionMode.

        Args:
            pattern: Pattern to find
            search_region: Optional region to search in

        Returns:
            ActionResult with match data

        Example:
            wrapper = FindWrapper()
            result = wrapper.find(login_button_pattern)
            if result.success:
                print(f"Found at: {result.match_list[0].get_region()}")
        """
        if self.is_mock_mode():
            logger.debug(f"FindWrapper.find (MOCK): {pattern.name}")
            return self.mock_find.find(pattern, search_region)  # type: ignore[no-any-return]
        else:
            logger.debug(f"FindWrapper.find (REAL): {pattern.name}")
            return self._find_real(pattern, search_region)

    def find_all(
        self,
        pattern: "Pattern",
        search_region: Region | None = None,
    ) -> list["Match"]:
        """Find all occurrences of a pattern.

        Routes to MockFind or HAL based on ExecutionMode.

        Args:
            pattern: Pattern to find
            search_region: Optional region to search in

        Returns:
            List of Match objects (empty if none found)

        Example:
            wrapper = FindWrapper()
            matches = wrapper.find_all(button_pattern)
            print(f"Found {len(matches)} matches")
        """
        if self.is_mock_mode():
            logger.debug(f"FindWrapper.find_all (MOCK): {pattern.name}")
            return self.mock_find.find_all(pattern, search_region)  # type: ignore[no-any-return]
        else:
            logger.debug(f"FindWrapper.find_all (REAL): {pattern.name}")
            return self._find_all_real(pattern, search_region)

    def wait_for(
        self,
        pattern: "Pattern",
        timeout: float = 5.0,
        search_region: Region | None = None,
    ) -> Optional["Match"]:
        """Wait for a pattern to appear.

        Routes to MockFind or HAL based on ExecutionMode.

        Args:
            pattern: Pattern to wait for
            timeout: Maximum wait time in seconds
            search_region: Optional region to search in

        Returns:
            Match if found within timeout, None otherwise

        Example:
            wrapper = FindWrapper()
            match = wrapper.wait_for(dialog_pattern, timeout=10.0)
            if match:
                print("Dialog appeared!")
        """
        if self.is_mock_mode():
            logger.debug(
                f"FindWrapper.wait_for (MOCK): {pattern.name}, timeout={timeout}"
            )
            return self.mock_find.wait_for(pattern, timeout)  # type: ignore[no-any-return]
        else:
            logger.debug(
                f"FindWrapper.wait_for (REAL): {pattern.name}, timeout={timeout}"
            )
            return self._wait_for_real(pattern, timeout, search_region)

    def _find_real(
        self,
        pattern: "Pattern",
        search_region: Region | None = None,
    ) -> "ActionResult":
        """Find pattern using real HAL implementations.

        Args:
            pattern: Pattern to find
            search_region: Optional search region

        Returns:
            ActionResult with match data
        """
        import time

        from ..actions.action_result import ActionResult
        from ..model.element.location import Location
        from ..model.match.match import Match

        start_time = time.time()
        result = ActionResult()  # type: ignore[call-arg]
        screenshot_pil = None

        try:
            # Capture screen
            if search_region and search_region.is_defined():
                screenshot = self.hal_capture.capture_region(search_region)
            else:
                screenshot = self.hal_capture.capture()

            # Get pattern image
            pattern_image = pattern.image if hasattr(pattern, "image") else None
            if not pattern_image:
                logger.warning(f"Pattern {pattern.name} has no image data")
                object.__setattr__(result, "success", False)
                return result

            # Get effective similarity threshold
            similarity = (
                pattern.get_effective_similarity()
                if hasattr(pattern, "get_effective_similarity")
                else 0.7
            )

            # Find pattern using OpenCV matcher
            # Note: IPatternMatcher.find_pattern expects PIL Images
            import numpy as np
            from PIL import Image as PILImage

            # Convert to PIL if needed
            if isinstance(screenshot, np.ndarray):
                screenshot_pil = PILImage.fromarray(screenshot)
            else:
                screenshot_pil = screenshot

            if hasattr(pattern_image, "get_pil"):
                pattern_pil = pattern_image.get_pil()
            elif isinstance(pattern_image, np.ndarray):
                pattern_pil = PILImage.fromarray(pattern_image)
            else:
                pattern_pil = pattern_image

            # Use HAL matcher
            hal_match = self.hal_matcher.find_pattern(
                haystack=screenshot_pil,
                needle=pattern_pil,
                confidence=similarity,
                grayscale=(
                    not pattern.use_color if hasattr(pattern, "use_color") else False
                ),
            )

            if hal_match:
                # Convert HAL Match to qontinui Match
                # HAL Match has: x, y, width, height, confidence, center
                match = Match(
                    target=Location(
                        region=Region(
                            x=hal_match.x,
                            y=hal_match.y,
                            width=hal_match.width,
                            height=hal_match.height,
                        )
                    ),
                    score=hal_match.confidence,
                    name=pattern.name,
                )

                object.__setattr__(result, "success", True)
                result.add_match(match)  # type: ignore[attr-defined]
                logger.debug(
                    f"Found pattern {pattern.name} at ({hal_match.x}, {hal_match.y}) with score {hal_match.confidence:.3f}"
                )
            else:
                object.__setattr__(result, "success", False)
                logger.debug(f"Pattern {pattern.name} not found")

        except Exception as e:
            logger.error(f"Error finding pattern {pattern.name}: {e}", exc_info=True)
            object.__setattr__(result, "success", False)

        # Set duration
        from datetime import timedelta

        elapsed_seconds = time.time() - start_time
        object.__setattr__(result, "duration", timedelta(seconds=elapsed_seconds))

        # Record action if recording is enabled
        duration_ms = elapsed_seconds * 1000
        matches = result.match_list if result.match_list else []  # type: ignore[attr-defined]
        self._record_find_action(pattern, matches, screenshot_pil, duration_ms)

        return result

    def _find_all_real(
        self,
        pattern: "Pattern",
        search_region: Region | None = None,
    ) -> list["Match"]:
        """Find all pattern occurrences using real HAL implementations.

        Args:
            pattern: Pattern to find
            search_region: Optional search region

        Returns:
            List of Match objects
        """
        import time

        from ..model.element.location import Location
        from ..model.match.match import Match

        start_time = time.time()
        screenshot_pil = None

        try:
            # Capture screen
            if search_region and search_region.is_defined():
                screenshot = self.hal_capture.capture_region(search_region)
            else:
                screenshot = self.hal_capture.capture()

            # Get pattern image
            pattern_image = pattern.image if hasattr(pattern, "image") else None
            if not pattern_image:
                logger.warning(f"Pattern {pattern.name} has no image data")
                return []

            # Get effective similarity threshold
            similarity = (
                pattern.get_effective_similarity()
                if hasattr(pattern, "get_effective_similarity")
                else 0.7
            )

            # Convert to PIL if needed
            import numpy as np
            from PIL import Image as PILImage

            if isinstance(screenshot, np.ndarray):
                screenshot_pil = PILImage.fromarray(screenshot)
            else:
                screenshot_pil = screenshot

            if hasattr(pattern_image, "get_pil"):
                pattern_pil = pattern_image.get_pil()
            elif isinstance(pattern_image, np.ndarray):
                pattern_pil = PILImage.fromarray(pattern_image)
            else:
                pattern_pil = pattern_image

            # Use HAL matcher to find all
            hal_matches = self.hal_matcher.find_all_patterns(
                haystack=screenshot_pil,
                needle=pattern_pil,
                confidence=similarity,
                grayscale=(
                    not pattern.use_color if hasattr(pattern, "use_color") else False
                ),
            )

            # Convert HAL matches to qontinui Matches
            matches = []
            for hal_match in hal_matches:
                match = Match(
                    target=Location(
                        region=Region(
                            x=hal_match.x,
                            y=hal_match.y,
                            width=hal_match.width,
                            height=hal_match.height,
                        )
                    ),
                    score=hal_match.confidence,
                    name=pattern.name,
                )
                matches.append(match)

            logger.debug(f"Found {len(matches)} occurrences of pattern {pattern.name}")

            # Record action if recording is enabled
            duration_ms = (time.time() - start_time) * 1000
            self._record_find_action(pattern, matches, screenshot_pil, duration_ms)

            return matches

        except Exception as e:
            logger.error(
                f"Error finding all patterns {pattern.name}: {e}", exc_info=True
            )
            return []

    def _wait_for_real(
        self,
        pattern: "Pattern",
        timeout: float = 5.0,
        search_region: Region | None = None,
    ) -> Optional["Match"]:
        """Wait for pattern using real HAL implementations.

        Args:
            pattern: Pattern to wait for
            timeout: Maximum wait time in seconds
            search_region: Optional search region

        Returns:
            Match if found, None if timeout
        """
        import time

        start_time = time.time()
        poll_interval = 0.5  # Check every 500ms

        while time.time() - start_time < timeout:
            result = self._find_real(pattern, search_region)
            if result.success and result.match_list:  # type: ignore[attr-defined]
                return result.match_list[0]  # type: ignore[attr-defined,no-any-return]

            # Wait before next attempt
            time.sleep(poll_interval)

        logger.debug(f"Timeout waiting for pattern {pattern.name} after {timeout}s")
        return None

    def _record_find_action(
        self,
        pattern: "Pattern",
        matches: list["Match"],
        screenshot,
        duration_ms: float,
    ):
        """Record a find action if recording is enabled.

        Args:
            pattern: Pattern that was searched
            matches: List of matches found
            screenshot: Screenshot image (PIL Image)
            duration_ms: Duration of search in milliseconds
        """
        # Get controller and check if recording
        from .controller import get_controller

        controller = get_controller()

        if not controller.is_recording():
            return

        # Get active states
        from ..state_management.state_memory import get_state_memory

        state_memory = get_state_memory()
        active_state_names = state_memory.get_active_state_names()

        # Get pattern ID and name
        pattern_id = pattern.id if hasattr(pattern, "id") else pattern.name
        pattern_name = pattern.name

        # Record action
        try:
            if controller.recorder is not None:
                controller.recorder.record_find_action(
                    pattern_id=pattern_id,
                    pattern_name=pattern_name,
                    matches=matches,
                    screenshot=screenshot,
                    active_states=set(active_state_names),
                    duration_ms=duration_ms,
                )
                logger.debug(f"Recorded find action for pattern {pattern_name}")
        except Exception as e:
            logger.error(f"Failed to record find action: {e}", exc_info=True)
