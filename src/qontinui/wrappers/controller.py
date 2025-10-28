"""ExecutionModeController - Central coordinator for all wrappers (Brobot pattern).

Provides a single point of access to all wrapper instances, following the
Brobot pattern where high-level code uses a central controller rather than
instantiating wrappers directly.

This controller manages the lifecycle of all wrappers and provides convenient
access to find, capture, mouse, keyboard, and time operations.

Architecture:
    High-Level Actions
      ↓
    ExecutionModeController (this class) ← Single point of contact
      ↓
    ├─ FindWrapper
    ├─ CaptureWrapper
    ├─ MouseWrapper
    ├─ KeyboardWrapper
    └─ TimeWrapper
      ↓
    Each wrapper routes to mock or real based on ExecutionMode
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..config.execution_mode import ExecutionModeConfig, get_execution_mode, set_execution_mode
from .capture_wrapper import CaptureWrapper
from .find_wrapper import FindWrapper
from .input_wrapper import KeyboardWrapper, MouseWrapper
from .time_wrapper import TimeWrapper

if TYPE_CHECKING:
    from ..mock.recorder import SnapshotRecorder

logger = logging.getLogger(__name__)


class ExecutionModeController:
    """Central controller for all action wrappers.

    Provides unified access to all wrappers and execution mode management.
    This is the main entry point for high-level actions to access automation
    functionality in a mock/real-agnostic way.

    Example:
        # Get controller (singleton)
        controller = ExecutionModeController.get_instance()

        # Access wrappers
        screenshot = controller.capture.capture()
        matches = controller.find.find_all(pattern)
        controller.mouse.click(100, 200)
        controller.keyboard.type_text("Hello")
        controller.time.wait(1.0)

        # Switch modes
        controller.set_mock_mode()
        # All subsequent operations use mock implementations

    Attributes:
        find: FindWrapper instance
        capture: CaptureWrapper instance
        mouse: MouseWrapper instance
        keyboard: KeyboardWrapper instance
        time: TimeWrapper instance
    """

    # Singleton instance
    _instance: Optional["ExecutionModeController"] = None

    def __init__(self) -> None:
        """Initialize controller with all wrappers.

        Note: Use get_instance() instead of direct instantiation.
        """
        # Lazy initialization of wrappers
        self._find: FindWrapper | None = None
        self._capture: CaptureWrapper | None = None
        self._mouse: MouseWrapper | None = None
        self._keyboard: KeyboardWrapper | None = None
        self._time: TimeWrapper | None = None

        # Recording state
        self.recorder: SnapshotRecorder | None = (
            None  # Type hint as string to avoid circular import
        )
        self.recording_enabled: bool = False

        logger.debug("ExecutionModeController initialized")

    @classmethod
    def get_instance(cls) -> "ExecutionModeController":
        """Get singleton controller instance.

        Returns:
            ExecutionModeController singleton

        Example:
            controller = ExecutionModeController.get_instance()
            controller.find.find_all(pattern)
        """
        if cls._instance is None:
            cls._instance = ExecutionModeController()
            logger.info("ExecutionModeController singleton created")
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance.

        Useful for testing or when you need to reinitialize all wrappers.

        Example:
            # Reset controller
            ExecutionModeController.reset_instance()

            # Get fresh instance
            controller = ExecutionModeController.get_instance()
        """
        cls._instance = None
        logger.debug("ExecutionModeController singleton reset")

    @property
    def find(self) -> FindWrapper:
        """Get FindWrapper instance.

        Returns:
            FindWrapper for pattern finding operations

        Example:
            controller = ExecutionModeController.get_instance()
            matches = controller.find.find_all(button_pattern)
        """
        if self._find is None:
            self._find = FindWrapper()
            logger.debug("FindWrapper instantiated")
        return self._find

    @property
    def capture(self) -> CaptureWrapper:
        """Get CaptureWrapper instance.

        Returns:
            CaptureWrapper for screen capture operations

        Example:
            controller = ExecutionModeController.get_instance()
            screenshot = controller.capture.capture()
        """
        if self._capture is None:
            self._capture = CaptureWrapper()
            logger.debug("CaptureWrapper instantiated")
        return self._capture

    @property
    def mouse(self) -> MouseWrapper:
        """Get MouseWrapper instance.

        Returns:
            MouseWrapper for mouse operations

        Example:
            controller = ExecutionModeController.get_instance()
            controller.mouse.click(100, 200)
        """
        if self._mouse is None:
            self._mouse = MouseWrapper()
            logger.debug("MouseWrapper instantiated")
        return self._mouse

    @property
    def keyboard(self) -> KeyboardWrapper:
        """Get KeyboardWrapper instance.

        Returns:
            KeyboardWrapper for keyboard operations

        Example:
            controller = ExecutionModeController.get_instance()
            controller.keyboard.type_text("Hello World")
        """
        if self._keyboard is None:
            self._keyboard = KeyboardWrapper()
            logger.debug("KeyboardWrapper instantiated")
        return self._keyboard

    @property
    def time(self) -> TimeWrapper:
        """Get TimeWrapper instance.

        Returns:
            TimeWrapper for time operations

        Example:
            controller = ExecutionModeController.get_instance()
            controller.time.wait(2.5)
        """
        if self._time is None:
            self._time = TimeWrapper()
            logger.debug("TimeWrapper instantiated")
        return self._time

    # Execution mode management

    def is_mock_mode(self) -> bool:
        """Check if currently in mock mode.

        Returns:
            True if in mock mode

        Example:
            controller = ExecutionModeController.get_instance()
            if controller.is_mock_mode():
                print("Running in mock mode")
        """
        return get_execution_mode().is_mock()

    def is_real_mode(self) -> bool:
        """Check if currently in real mode.

        Returns:
            True if in real mode

        Example:
            controller = ExecutionModeController.get_instance()
            if controller.is_real_mode():
                print("Running in real mode")
        """
        return get_execution_mode().is_real()

    def set_mock_mode(self) -> None:
        """Switch to mock mode.

        All subsequent operations will use mock implementations.

        Example:
            controller = ExecutionModeController.get_instance()
            controller.set_mock_mode()
            # Now all operations use mock
        """
        from ..config.execution_mode import MockMode

        set_execution_mode(ExecutionModeConfig(mode=MockMode.MOCK))
        logger.info("Switched to MOCK mode")

    def set_real_mode(self) -> None:
        """Switch to real mode.

        All subsequent operations will use real HAL implementations.

        Example:
            controller = ExecutionModeController.get_instance()
            controller.set_real_mode()
            # Now all operations use real HAL
        """
        from ..config.execution_mode import MockMode

        set_execution_mode(ExecutionModeConfig(mode=MockMode.REAL))
        logger.info("Switched to REAL mode")

    def set_screenshot_mode(self, screenshot_dir: str) -> None:
        """Switch to screenshot-based testing mode.

        Args:
            screenshot_dir: Directory containing pre-captured screenshots

        Example:
            controller = ExecutionModeController.get_instance()
            controller.set_screenshot_mode("./test_screenshots")
            # Now operations use pre-captured screenshots
        """
        from ..config.execution_mode import MockMode

        set_execution_mode(
            ExecutionModeConfig(
                mode=MockMode.SCREENSHOT,
                screenshot_dir=screenshot_dir,
            )
        )
        logger.info(f"Switched to SCREENSHOT mode: {screenshot_dir}")

    def get_execution_mode(self) -> ExecutionModeConfig:
        """Get current execution mode configuration.

        Returns:
            Current ExecutionModeConfig

        Example:
            controller = ExecutionModeController.get_instance()
            mode = controller.get_execution_mode()
            print(f"Current mode: {mode.mode.value}")
        """
        return get_execution_mode()

    # Recording management

    def start_recording(self, base_dir: str) -> str:
        """Start recording snapshots to directory.

        Args:
            base_dir: Base directory for snapshot storage

        Returns:
            Full path to the run directory where snapshots are being saved

        Raises:
            RuntimeError: If recording is already in progress

        Example:
            controller = ExecutionModeController.get_instance()
            snapshot_dir = controller.start_recording("/tmp/snapshots")
            print(f"Recording to: {snapshot_dir}")

            # Run automation...

            controller.stop_recording()
        """
        if self.recording_enabled:
            raise RuntimeError("Recording already in progress")

        # Import here to avoid circular dependency
        from ..mock.recorder import RecorderConfig, SnapshotRecorder

        config = RecorderConfig(base_dir=Path(base_dir))
        self.recorder = SnapshotRecorder(config)
        self.recording_enabled = True

        snapshot_dir = str(self.recorder.get_snapshot_directory())
        logger.info(f"Started recording to: {snapshot_dir}")

        return snapshot_dir

    def stop_recording(self) -> str | None:
        """Stop recording and finalize snapshots.

        Returns:
            Path to snapshot directory, or None if not recording

        Example:
            controller = ExecutionModeController.get_instance()
            controller.start_recording("/tmp/snapshots")

            # Run automation...

            snapshot_dir = controller.stop_recording()
            print(f"Recording saved to: {snapshot_dir}")
        """
        if not self.recording_enabled or not self.recorder:
            logger.warning("stop_recording() called but no recording in progress")
            return None

        self.recorder.finalize()
        snapshot_dir = str(self.recorder.get_snapshot_directory())

        self.recording_enabled = False
        self.recorder = None

        logger.info(f"Stopped recording: {snapshot_dir}")

        return snapshot_dir

    def is_recording(self) -> bool:
        """Check if recording is currently enabled.

        Returns:
            True if recording is active

        Example:
            controller = ExecutionModeController.get_instance()
            if controller.is_recording():
                print("Recording in progress")
        """
        return self.recording_enabled

    def get_recording_stats(self) -> dict | None:
        """Get current recording statistics.

        Returns:
            Dictionary with statistics, or None if not recording

        Example:
            controller = ExecutionModeController.get_instance()
            stats = controller.get_recording_stats()
            if stats:
                print(f"Actions recorded: {stats['actions_recorded']}")
        """
        if self.recorder:
            return self.recorder.get_statistics()
        return None


# Convenience function for getting controller
def get_controller() -> ExecutionModeController:
    """Get ExecutionModeController singleton.

    Convenience function as alternative to ExecutionModeController.get_instance().

    Returns:
        ExecutionModeController singleton

    Example:
        from qontinui.wrappers.controller import get_controller

        controller = get_controller()
        controller.find.find_all(pattern)
    """
    return ExecutionModeController.get_instance()
