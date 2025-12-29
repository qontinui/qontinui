"""Checkpoint service for capturing automation state with screenshots and OCR."""

import tempfile
from pathlib import Path
from typing import Any

from PIL import Image
from qontinui_schemas.common import utc_now

from ..exceptions import ScreenCaptureException
from ..hal.interfaces.ocr_engine import IOCREngine
from ..hal.interfaces.screen_capture import IScreenCapture
from ..logging import get_logger
from .checkpoint_result import CheckpointData, CheckpointTrigger, TextRegionData

logger = get_logger(__name__)


class CheckpointService:
    """Service for capturing and managing automation checkpoints.

    Provides functionality to capture screenshots with optional OCR analysis
    at key points during automation execution. Useful for debugging, verification,
    and creating audit trails.

    Example:
        >>> from qontinui.hal.implementations import MSSScreenCapture, EasyOCREngine
        >>> screen_capture = MSSScreenCapture()
        >>> ocr_engine = EasyOCREngine()
        >>> service = CheckpointService(screen_capture, ocr_engine)
        >>>
        >>> # Capture checkpoint after action
        >>> checkpoint = service.capture_checkpoint(
        ...     name="after_login",
        ...     trigger=CheckpointTrigger.TRANSITION_COMPLETE,
        ...     action_context="login_action"
        ... )
        >>> print(f"Captured {checkpoint.region_count} text regions")
        >>> print(f"Screenshot: {checkpoint.screenshot_path}")
    """

    def __init__(
        self,
        screen_capture: IScreenCapture,
        ocr_engine: IOCREngine | None = None,
        output_dir: Path | None = None,
    ):
        """Initialize checkpoint service.

        Args:
            screen_capture: Screen capture implementation (e.g., MSSScreenCapture)
            ocr_engine: Optional OCR engine for text extraction (e.g., EasyOCREngine)
            output_dir: Directory to save screenshots (defaults to temp directory)
        """
        self._screen_capture = screen_capture
        self._ocr_engine = ocr_engine

        # Set up output directory
        if output_dir is None:
            self._output_dir = Path(tempfile.gettempdir()) / "qontinui_checkpoints"
        else:
            self._output_dir = output_dir

        self._output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "checkpoint_service_initialized",
            output_dir=str(self._output_dir),
            ocr_enabled=self._ocr_engine is not None,
        )

    def capture_checkpoint(
        self,
        name: str,
        trigger: CheckpointTrigger,
        action_context: str | None = None,
        region: tuple[int, int, int, int] | None = None,
        run_ocr: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> CheckpointData:
        """Capture a checkpoint with screenshot and optional OCR.

        Args:
            name: Descriptive name for this checkpoint
            trigger: What triggered this checkpoint
            action_context: Optional context (e.g., action name that triggered it)
            region: Optional region to capture (x, y, width, height), None for full screen
            run_ocr: Whether to run OCR on the screenshot
            metadata: Additional metadata to store with checkpoint

        Returns:
            CheckpointData containing all captured information

        Raises:
            ScreenCaptureException: If screenshot capture fails
        """
        timestamp = utc_now()

        logger.debug(
            "capturing_checkpoint",
            name=name,
            trigger=trigger.value,
            action_context=action_context,
            region=region,
            run_ocr=run_ocr,
        )

        try:
            # Capture screenshot
            screenshot_path = self.save_screenshot(name, region=region)

            # Extract OCR data if enabled and available
            ocr_text = ""
            text_regions: tuple[TextRegionData, ...] = ()

            if run_ocr and self._ocr_engine is not None:
                try:
                    ocr_text, text_regions = self._extract_ocr_data(screenshot_path)
                    logger.debug(
                        "ocr_extraction_complete",
                        name=name,
                        char_count=len(ocr_text),
                        region_count=len(text_regions),
                    )
                except Exception as e:
                    # Log OCR failure but don't fail the checkpoint
                    logger.warning(
                        "ocr_extraction_failed",
                        name=name,
                        error=str(e),
                        error_type=type(e).__name__,
                    )

            # Create checkpoint data
            checkpoint = CheckpointData(
                name=name,
                timestamp=timestamp,
                screenshot_path=screenshot_path,
                ocr_text=ocr_text,
                text_regions=text_regions,
                trigger=trigger,
                action_context=action_context,
                metadata=metadata,
            )

            logger.info(
                "checkpoint_captured",
                name=name,
                trigger=trigger.value,
                screenshot=screenshot_path,
                has_ocr=bool(ocr_text),
            )

            return checkpoint

        except ScreenCaptureException:
            raise
        except Exception as e:
            raise ScreenCaptureException(f"Failed to capture checkpoint '{name}': {e}") from e

    def save_screenshot(
        self,
        name: str,
        region: tuple[int, int, int, int] | None = None,
    ) -> str:
        """Save screenshot to output directory.

        Args:
            name: Base name for the screenshot file
            region: Optional region to capture (x, y, width, height)

        Returns:
            Absolute path to saved screenshot

        Raises:
            ScreenCaptureException: If screenshot save fails
        """
        try:
            # Generate filename with timestamp
            timestamp = utc_now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{name}_{timestamp}.png"
            filepath = self._output_dir / filename

            # Use screen capture's save method
            if region:
                saved_path = self._screen_capture.save_screenshot(
                    filepath=str(filepath),
                    region=region,
                )
            else:
                saved_path = self._screen_capture.save_screenshot(
                    filepath=str(filepath),
                )

            logger.debug("screenshot_saved", name=name, path=saved_path)

            return saved_path

        except Exception as e:
            raise ScreenCaptureException(f"Failed to save screenshot '{name}': {e}") from e

    def _extract_ocr_data(
        self,
        screenshot_path: str,
    ) -> tuple[str, tuple[TextRegionData, ...]]:
        """Extract OCR text and regions from screenshot.

        Args:
            screenshot_path: Path to screenshot file

        Returns:
            Tuple of (full_text, text_regions)

        Raises:
            Exception: If OCR extraction fails
        """
        if self._ocr_engine is None:
            return "", ()

        # Load image
        image = Image.open(screenshot_path)

        # Extract full text
        full_text = self._ocr_engine.extract_text(image)

        # Get text regions with bounding boxes
        ocr_regions = self._ocr_engine.get_text_regions(image, min_confidence=0.5)

        # Convert to immutable TextRegionData tuples
        text_regions = tuple(
            TextRegionData(
                text=region.text,
                x=region.x,
                y=region.y,
                width=region.width,
                height=region.height,
                confidence=region.confidence,
            )
            for region in ocr_regions
        )

        return full_text, text_regions

    @property
    def output_dir(self) -> Path:
        """Get output directory for screenshots."""
        return self._output_dir

    @property
    def has_ocr(self) -> bool:
        """Check if OCR engine is available."""
        return self._ocr_engine is not None

    def clear_checkpoints(self, older_than_hours: int | None = None) -> int:
        """Clear checkpoint screenshots from output directory.

        Args:
            older_than_hours: Only delete checkpoints older than this many hours.
                            None to delete all.

        Returns:
            Number of files deleted
        """
        import time

        deleted_count = 0

        try:
            for filepath in self._output_dir.glob("*.png"):
                should_delete = True

                if older_than_hours is not None:
                    # Check file age
                    file_age_seconds = time.time() - filepath.stat().st_mtime
                    file_age_hours = file_age_seconds / 3600
                    should_delete = file_age_hours > older_than_hours

                if should_delete:
                    filepath.unlink()
                    deleted_count += 1

            logger.info(
                "checkpoints_cleared",
                deleted_count=deleted_count,
                older_than_hours=older_than_hours,
            )

            return deleted_count

        except Exception as e:
            logger.warning("checkpoint_cleanup_failed", error=str(e))
            return deleted_count
