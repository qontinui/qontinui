"""Base wrapper protocol for action routing (Brobot pattern).

Defines the common interface that all action wrappers must implement.
Wrappers route calls to either mock or real implementations based on
ExecutionMode configuration.

Architecture:
    High-Level Actions (FindImage, StateDetector, etc.)
      ↓
    Wrappers (FindWrapper, CaptureWrapper, etc.) ← This layer
      ↓
    ├─ if mock → Mock Implementations (MockFind, MockCapture, etc.)
    └─ if real → HAL Layer → Platform Implementations
"""

from typing import Protocol, runtime_checkable

from ..config.execution_mode import get_execution_mode


@runtime_checkable
class ActionWrapper(Protocol):
    """Base protocol for all action wrappers.

    All wrappers (FindWrapper, CaptureWrapper, MouseWrapper, etc.) must
    implement this protocol to ensure consistent behavior across the system.

    The wrapper's primary responsibility is routing: check ExecutionMode
    and delegate to either mock or real implementations.

    Example:
        class FindWrapper:
            def find_pattern(self, pattern: Pattern, ...) -> List[Match]:
                if self.is_mock_mode():
                    return self.mock_find.get_matches(pattern)
                else:
                    return self.hal_matcher.find_all_patterns(...)

    Note:
        This is a Protocol (structural typing), not a base class.
        Wrappers don't need to explicitly inherit from this, they just
        need to implement the required methods.
    """

    def is_mock_mode(self) -> bool:
        """Check if currently running in mock mode.

        Returns:
            True if in mock mode (use historical data), False otherwise

        Example:
            wrapper = FindWrapper()
            if wrapper.is_mock_mode():
                print("Using mock implementations")
            else:
                print("Using real HAL implementations")
        """
        ...


class BaseWrapper:
    """Base implementation providing common wrapper functionality.

    Concrete wrappers can inherit from this to get standard implementations
    of common methods like is_mock_mode().

    This is optional - wrappers can implement ActionWrapper protocol directly
    without inheriting from this class.

    Example:
        class FindWrapper(BaseWrapper):
            def __init__(self):
                super().__init__()
                self.mock_find = MockFind()
                self.hal_matcher = OpenCVMatcher()

            def find_pattern(self, pattern: Pattern) -> List[Match]:
                if self.is_mock_mode():
                    return self.mock_find.get_matches(pattern)
                else:
                    return self.hal_matcher.find_all_patterns(...)
    """

    def __init__(self):
        """Initialize base wrapper.

        Sets up access to global ExecutionMode configuration.
        """
        pass  # No state needed - uses global config

    def is_mock_mode(self) -> bool:
        """Check if currently running in mock mode.

        Returns:
            True if in mock mode, False if in real mode

        Implementation:
            Delegates to global ExecutionModeConfig.is_mock()
        """
        return get_execution_mode().is_mock()

    def is_screenshot_mode(self) -> bool:
        """Check if currently running in screenshot-based testing mode.

        Returns:
            True if in screenshot mode with valid directory

        Note:
            Screenshot mode is considered "real" mode (not mock),
            but uses pre-captured screenshots instead of live screen capture.
        """
        return get_execution_mode().is_screenshot_mode()

    def is_real_mode(self) -> bool:
        """Check if currently running in real automation mode.

        Returns:
            True if using real HAL implementations (includes screenshot mode)

        Note:
            Real mode includes both:
            - True real mode (live screen capture + automation)
            - Screenshot mode (pre-captured screens + real CV/automation)
        """
        return get_execution_mode().is_real()
