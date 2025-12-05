"""Unit tests for checkpointing module."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from PIL import Image

from qontinui.checkpointing import (
    CheckpointData,
    CheckpointService,
    CheckpointTrigger,
    TextRegionData,
)
from qontinui.hal.interfaces.ocr_engine import TextRegion


class TestTextRegionData:
    """Test TextRegionData model."""

    def test_creation(self):
        """Test creating a TextRegionData instance."""
        region = TextRegionData(
            text="Hello",
            x=10,
            y=20,
            width=100,
            height=50,
            confidence=0.95,
        )

        assert region.text == "Hello"
        assert region.x == 10
        assert region.y == 20
        assert region.width == 100
        assert region.height == 50
        assert region.confidence == 0.95

    def test_bounds(self):
        """Test bounds property."""
        region = TextRegionData(
            text="Test",
            x=10,
            y=20,
            width=100,
            height=50,
            confidence=0.9,
        )

        assert region.bounds == (10, 20, 100, 50)

    def test_center(self):
        """Test center property."""
        region = TextRegionData(
            text="Test",
            x=0,
            y=0,
            width=100,
            height=50,
            confidence=0.9,
        )

        assert region.center == (50, 25)

    def test_immutability(self):
        """Test that TextRegionData is immutable."""
        region = TextRegionData(
            text="Test",
            x=10,
            y=20,
            width=100,
            height=50,
            confidence=0.9,
        )

        with pytest.raises(AttributeError):
            region.text = "Modified"  # type: ignore[misc]


class TestCheckpointTrigger:
    """Test CheckpointTrigger enum."""

    def test_enum_values(self):
        """Test enum has expected values."""
        assert CheckpointTrigger.MANUAL.value == "manual"
        assert CheckpointTrigger.TRANSITION_COMPLETE.value == "transition"
        assert CheckpointTrigger.TRANSITION_FAILURE.value == "transition_failure"
        assert CheckpointTrigger.TERMINAL_FAILURE.value == "terminal_failure"


class TestCheckpointData:
    """Test CheckpointData model."""

    def test_creation(self):
        """Test creating a CheckpointData instance."""
        region = TextRegionData(
            text="Test",
            x=10,
            y=20,
            width=100,
            height=50,
            confidence=0.9,
        )

        checkpoint = CheckpointData(
            name="test_checkpoint",
            timestamp=datetime.now(),
            screenshot_path="/tmp/test.png",
            ocr_text="Hello World",
            text_regions=(region,),
            trigger=CheckpointTrigger.MANUAL,
            action_context="test_action",
            metadata={"key": "value"},
        )

        assert checkpoint.name == "test_checkpoint"
        assert checkpoint.screenshot_path == "/tmp/test.png"
        assert checkpoint.ocr_text == "Hello World"
        assert len(checkpoint.text_regions) == 1
        assert checkpoint.trigger == CheckpointTrigger.MANUAL
        assert checkpoint.action_context == "test_action"
        assert checkpoint.metadata == {"key": "value"}

    def test_has_screenshot(self):
        """Test has_screenshot property."""
        checkpoint_with = CheckpointData(
            name="test",
            timestamp=datetime.now(),
            screenshot_path="/tmp/test.png",
            ocr_text="",
            text_regions=(),
            trigger=CheckpointTrigger.MANUAL,
            action_context=None,
            metadata=None,
        )

        checkpoint_without = CheckpointData(
            name="test",
            timestamp=datetime.now(),
            screenshot_path=None,
            ocr_text="",
            text_regions=(),
            trigger=CheckpointTrigger.MANUAL,
            action_context=None,
            metadata=None,
        )

        assert checkpoint_with.has_screenshot is True
        assert checkpoint_without.has_screenshot is False

    def test_region_count(self):
        """Test region_count property."""
        region1 = TextRegionData("Test1", 0, 0, 100, 50, 0.9)
        region2 = TextRegionData("Test2", 100, 0, 100, 50, 0.9)

        checkpoint = CheckpointData(
            name="test",
            timestamp=datetime.now(),
            screenshot_path=None,
            ocr_text="Test1 Test2",
            text_regions=(region1, region2),
            trigger=CheckpointTrigger.MANUAL,
            action_context=None,
            metadata=None,
        )

        assert checkpoint.region_count == 2

    def test_get_regions_containing(self):
        """Test searching for text in regions."""
        region1 = TextRegionData("Hello World", 0, 0, 100, 50, 0.9)
        region2 = TextRegionData("Goodbye", 100, 0, 100, 50, 0.9)
        region3 = TextRegionData("hello there", 200, 0, 100, 50, 0.9)

        checkpoint = CheckpointData(
            name="test",
            timestamp=datetime.now(),
            screenshot_path=None,
            ocr_text="",
            text_regions=(region1, region2, region3),
            trigger=CheckpointTrigger.MANUAL,
            action_context=None,
            metadata=None,
        )

        # Case-insensitive search
        matches = checkpoint.get_regions_containing("hello", case_sensitive=False)
        assert len(matches) == 2
        assert matches[0].text == "Hello World"
        assert matches[1].text == "hello there"

        # Case-sensitive search
        matches = checkpoint.get_regions_containing("Hello", case_sensitive=True)
        assert len(matches) == 1
        assert matches[0].text == "Hello World"


class TestCheckpointService:
    """Test CheckpointService."""

    @pytest.fixture
    def mock_screen_capture(self):
        """Create a mock screen capture."""
        mock = MagicMock()
        mock.save_screenshot.return_value = "/tmp/test.png"
        return mock

    @pytest.fixture
    def mock_ocr_engine(self):
        """Create a mock OCR engine."""
        mock = MagicMock()
        mock.extract_text.return_value = "Test OCR Text"
        mock.get_text_regions.return_value = [
            TextRegion(
                text="Test",
                x=10,
                y=20,
                width=100,
                height=50,
                confidence=0.95,
            )
        ]
        return mock

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_initialization(self, mock_screen_capture, temp_output_dir):
        """Test CheckpointService initialization."""
        service = CheckpointService(
            screen_capture=mock_screen_capture,
            output_dir=temp_output_dir,
        )

        assert service._screen_capture == mock_screen_capture
        assert service._ocr_engine is None
        assert service.output_dir == temp_output_dir
        assert service.has_ocr is False

    def test_initialization_with_ocr(self, mock_screen_capture, mock_ocr_engine, temp_output_dir):
        """Test CheckpointService initialization with OCR."""
        service = CheckpointService(
            screen_capture=mock_screen_capture,
            ocr_engine=mock_ocr_engine,
            output_dir=temp_output_dir,
        )

        assert service._ocr_engine == mock_ocr_engine
        assert service.has_ocr is True

    def test_save_screenshot(self, mock_screen_capture, temp_output_dir):
        """Test saving a screenshot."""
        service = CheckpointService(
            screen_capture=mock_screen_capture,
            output_dir=temp_output_dir,
        )

        path = service.save_screenshot("test_checkpoint")

        assert path == "/tmp/test.png"
        mock_screen_capture.save_screenshot.assert_called_once()

    def test_save_screenshot_with_region(self, mock_screen_capture, temp_output_dir):
        """Test saving a screenshot with region."""
        service = CheckpointService(
            screen_capture=mock_screen_capture,
            output_dir=temp_output_dir,
        )

        region = (10, 20, 100, 50)
        path = service.save_screenshot("test_checkpoint", region=region)

        assert path == "/tmp/test.png"
        mock_screen_capture.save_screenshot.assert_called_once()
        call_kwargs = mock_screen_capture.save_screenshot.call_args[1]
        assert call_kwargs["region"] == region

    @patch("PIL.Image.open")
    def test_capture_checkpoint(
        self, mock_image_open, mock_screen_capture, mock_ocr_engine, temp_output_dir
    ):
        """Test capturing a checkpoint."""
        # Mock PIL Image
        mock_image = Mock(spec=Image.Image)
        mock_image_open.return_value = mock_image

        service = CheckpointService(
            screen_capture=mock_screen_capture,
            ocr_engine=mock_ocr_engine,
            output_dir=temp_output_dir,
        )

        checkpoint = service.capture_checkpoint(
            name="test_checkpoint",
            trigger=CheckpointTrigger.MANUAL,
            action_context="test_action",
            metadata={"key": "value"},
        )

        assert checkpoint.name == "test_checkpoint"
        assert checkpoint.trigger == CheckpointTrigger.MANUAL
        assert checkpoint.action_context == "test_action"
        assert checkpoint.metadata == {"key": "value"}
        assert checkpoint.screenshot_path == "/tmp/test.png"
        assert checkpoint.ocr_text == "Test OCR Text"
        assert len(checkpoint.text_regions) == 1
        assert checkpoint.text_regions[0].text == "Test"

    def test_capture_checkpoint_without_ocr(self, mock_screen_capture, temp_output_dir):
        """Test capturing a checkpoint without OCR."""
        service = CheckpointService(
            screen_capture=mock_screen_capture,
            output_dir=temp_output_dir,
        )

        checkpoint = service.capture_checkpoint(
            name="test_checkpoint",
            trigger=CheckpointTrigger.MANUAL,
        )

        assert checkpoint.ocr_text == ""
        assert len(checkpoint.text_regions) == 0

    @patch("PIL.Image.open")
    def test_capture_checkpoint_ocr_disabled(
        self, mock_image_open, mock_screen_capture, mock_ocr_engine, temp_output_dir
    ):
        """Test capturing a checkpoint with OCR disabled."""
        mock_image = Mock(spec=Image.Image)
        mock_image_open.return_value = mock_image

        service = CheckpointService(
            screen_capture=mock_screen_capture,
            ocr_engine=mock_ocr_engine,
            output_dir=temp_output_dir,
        )

        checkpoint = service.capture_checkpoint(
            name="test_checkpoint",
            trigger=CheckpointTrigger.MANUAL,
            run_ocr=False,
        )

        assert checkpoint.ocr_text == ""
        assert len(checkpoint.text_regions) == 0
        mock_ocr_engine.extract_text.assert_not_called()

    def test_clear_checkpoints(self, mock_screen_capture, temp_output_dir):
        """Test clearing checkpoint files."""
        service = CheckpointService(
            screen_capture=mock_screen_capture,
            output_dir=temp_output_dir,
        )

        # Create some dummy checkpoint files
        (temp_output_dir / "checkpoint1.png").touch()
        (temp_output_dir / "checkpoint2.png").touch()
        (temp_output_dir / "checkpoint3.png").touch()

        # Clear all checkpoints
        deleted = service.clear_checkpoints()

        assert deleted == 3
        assert len(list(temp_output_dir.glob("*.png"))) == 0
