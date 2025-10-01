"""Base interfaces for semantic processors."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from ...model.element.region import Region
from ..core.semantic_scene import SemanticScene


@dataclass
class ProcessorConfig:
    """Configuration for semantic processors.

    Provides common configuration options that can be used across
    different processor implementations.
    """

    min_confidence: float = 0.5
    """Minimum confidence threshold for detections."""

    enable_ocr: bool = True
    """Whether to enable OCR text extraction."""

    detect_colors: bool = True
    """Whether to detect dominant colors."""

    detect_shapes: bool = True
    """Whether to detect geometric shapes."""

    model_name: str | None = None
    """Name of the model to use (processor-specific)."""

    max_objects: int = 1000
    """Maximum number of objects to detect."""

    custom_params: dict[str, Any] = field(default_factory=dict)
    """Processor-specific custom parameters."""

    @classmethod
    def builder(cls) -> ProcessorConfigBuilder:
        """Create a builder for fluent configuration.

        Returns:
            ProcessorConfigBuilder instance
        """
        return ProcessorConfigBuilder()


class ProcessorConfigBuilder:
    """Builder for ProcessorConfig using fluent interface."""

    def __init__(self):
        self._config = ProcessorConfig()

    def with_min_confidence(self, confidence: float) -> ProcessorConfigBuilder:
        """Set minimum confidence threshold.

        Args:
            confidence: Minimum confidence (0.0 to 1.0)

        Returns:
            Self for chaining
        """
        self._config.min_confidence = confidence
        return self

    def with_ocr(self, enable: bool) -> ProcessorConfigBuilder:
        """Enable or disable OCR.

        Args:
            enable: Whether to enable OCR

        Returns:
            Self for chaining
        """
        self._config.enable_ocr = enable
        return self

    def with_color_detection(self, enable: bool) -> ProcessorConfigBuilder:
        """Enable or disable color detection.

        Args:
            enable: Whether to detect colors

        Returns:
            Self for chaining
        """
        self._config.detect_colors = enable
        return self

    def with_shape_detection(self, enable: bool) -> ProcessorConfigBuilder:
        """Enable or disable shape detection.

        Args:
            enable: Whether to detect shapes

        Returns:
            Self for chaining
        """
        self._config.detect_shapes = enable
        return self

    def with_model(self, model_name: str) -> ProcessorConfigBuilder:
        """Set the model to use.

        Args:
            model_name: Name of the model

        Returns:
            Self for chaining
        """
        self._config.model_name = model_name
        return self

    def with_max_objects(self, max_objects: int) -> ProcessorConfigBuilder:
        """Set maximum number of objects to detect.

        Args:
            max_objects: Maximum object count

        Returns:
            Self for chaining
        """
        self._config.max_objects = max_objects
        return self

    def with_custom_param(self, key: str, value: Any) -> ProcessorConfigBuilder:
        """Add a custom parameter.

        Args:
            key: Parameter name
            value: Parameter value

        Returns:
            Self for chaining
        """
        self._config.custom_params[key] = value
        return self

    def build(self) -> ProcessorConfig:
        """Build the configuration.

        Returns:
            ProcessorConfig instance
        """
        return cast(ProcessorConfig, self._config)


@dataclass
class ProcessingHints:
    """Hints to optimize processing.

    Provides contextual information that can help processors
    optimize their analysis strategies.
    """

    expected_object_types: list[str] = field(default_factory=list)
    """Types of objects expected in the scene."""

    focus_regions: list[Region] = field(default_factory=list)
    """Regions to focus processing on."""

    previous_scene: SemanticScene | None = None
    """Previous scene for incremental processing."""

    quick_mode: bool = False
    """Trade accuracy for speed when True."""

    context: str = ""
    """Context description (e.g., 'game_inventory', 'web_page')."""

    @classmethod
    def for_game_inventory(cls) -> ProcessingHints:
        """Create hints optimized for game inventory screens.

        Returns:
            ProcessingHints configured for inventory
        """
        return cls(
            expected_object_types=["icon", "text", "button", "list_item"], context="game_inventory"
        )

    @classmethod
    def for_web_page(cls) -> ProcessingHints:
        """Create hints optimized for web pages.

        Returns:
            ProcessingHints configured for web content
        """
        return cls(
            expected_object_types=["text", "link", "button", "image", "heading"], context="web_page"
        )

    @classmethod
    def for_desktop_app(cls) -> ProcessingHints:
        """Create hints optimized for desktop applications.

        Returns:
            ProcessingHints configured for desktop apps
        """
        return cls(
            expected_object_types=["window", "button", "menu", "text_field", "toolbar"],
            context="desktop_app",
        )

    @classmethod
    def for_dialog(cls) -> ProcessingHints:
        """Create hints optimized for dialog boxes.

        Returns:
            ProcessingHints configured for dialogs
        """
        return cls(
            expected_object_types=["button", "text", "checkbox", "radio_button"], context="dialog"
        )


class SemanticProcessor(ABC):
    """Base interface that all semantic processors must implement.

    Defines the contract for processors that analyze screenshots
    and extract semantic information.
    """

    def __init__(self):
        """Initialize processor."""
        self._config = ProcessorConfig()
        self._processing_times: list[float] = []
        self._max_processing_time: float | None = None

    @abstractmethod
    def process(self, screenshot: np.ndarray[Any, Any]) -> SemanticScene:
        """Process a screenshot and extract semantic information.

        Args:
            screenshot: Screenshot as numpy array (BGR format)

        Returns:
            SemanticScene containing detected objects
        """
        pass

    def configure(self, config: ProcessorConfig) -> None:
        """Configure the processor.

        Args:
            config: ProcessorConfig with settings
        """
        self._config = config

    def get_configuration(self) -> ProcessorConfig:
        """Get current configuration.

        Returns:
            Current ProcessorConfig
        """
        return cast(ProcessorConfig, self._config)

    @abstractmethod
    def get_supported_object_types(self) -> set[str]:
        """Get object types this processor can detect.

        Returns:
            Set of supported object type names
        """
        pass

    def supports_incremental_processing(self) -> bool:
        """Check if processor supports incremental processing.

        Returns:
            True if incremental processing is supported
        """
        return False

    def get_average_processing_time(self) -> float:
        """Get average processing time in milliseconds.

        Returns:
            Average time or 0 if no data
        """
        if not self._processing_times:
            return 0.0
        return sum(self._processing_times) / len(self._processing_times)

    def set_max_processing_time(self, milliseconds: float) -> None:
        """Set maximum allowed processing time.

        Args:
            milliseconds: Maximum time in milliseconds
        """
        self._max_processing_time = milliseconds

    def process_region(self, screenshot: np.ndarray[Any, Any], roi: Region) -> SemanticScene:
        """Process specific region of screenshot.

        Default implementation crops and processes the region.

        Args:
            screenshot: Full screenshot
            roi: Region of interest

        Returns:
            SemanticScene for the region
        """
        # Crop the image to the region
        cropped = screenshot[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]

        # Process the cropped region
        scene = self.process(cropped)

        # Adjust object locations to account for crop offset
        for obj in scene.objects:
            obj.location = obj.location.translate(roi.x, roi.y)

        return scene

    def process_with_hints(
        self, screenshot: np.ndarray[Any, Any], hints: ProcessingHints
    ) -> SemanticScene:
        """Process with optimization hints.

        Default implementation ignores hints and calls regular process.

        Args:
            screenshot: Screenshot to process
            hints: Processing hints

        Returns:
            SemanticScene with detected objects
        """
        # Store current config
        original_config = self._config

        try:
            # Apply hints to configuration if relevant
            if hints.quick_mode:
                self._config = ProcessorConfig(
                    min_confidence=self._config.min_confidence * 1.2,  # Higher threshold
                    max_objects=min(self._config.max_objects, 50),  # Fewer objects
                )

            # Process with modified config
            return self.process(screenshot)

        finally:
            # Restore original config
            self._config = original_config

    def _record_processing_time(self, start_time: float) -> None:
        """Record processing time for statistics.

        Args:
            start_time: Start time from time.time()
        """
        elapsed = (time.time() - start_time) * 1000  # Convert to ms
        self._processing_times.append(elapsed)

        # Keep only last 100 times
        if len(self._processing_times) > 100:
            self._processing_times = self._processing_times[-100:]

    def _check_timeout(self, start_time: float) -> bool:
        """Check if processing has exceeded max time.

        Args:
            start_time: Start time from time.time()

        Returns:
            True if timeout exceeded
        """
        if self._max_processing_time is None:
            return False

        elapsed = (time.time() - start_time) * 1000
        return elapsed > self._max_processing_time
