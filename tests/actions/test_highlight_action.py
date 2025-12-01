"""Unit tests for HIGHLIGHT action executor."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from src.qontinui.action_executors.base import ExecutionContext
from src.qontinui.action_executors.mouse import MouseActionExecutor
from src.qontinui.config.schema import (
    Action,
    Coordinates,
    CoordinatesTarget,
    HighlightActionConfig,
)


@pytest.fixture
def mock_context():
    """Create a mock execution context for testing."""
    context = Mock(spec=ExecutionContext)
    context.mouse = Mock()
    context.keyboard = Mock()
    context.screen = Mock()
    context.time = Mock()
    context.config = Mock()
    context.defaults = Mock()
    context.last_action_result = None
    context.variable_context = Mock()
    context.state_executor = None
    context.control_flow_executor = Mock()
    context.data_operations_executor = Mock()
    context.workflow_executor = None
    context.emit_event = Mock()
    context.emit_action_event = Mock()
    context.emit_image_recognition_event = Mock()

    # Mock config.image_map
    context.config.image_map = {}

    return context


@pytest.fixture
def mouse_executor(mock_context):
    """Create a MouseActionExecutor with mocked context."""
    return MouseActionExecutor(mock_context)


class TestHighlightAction:
    """Test suite for HIGHLIGHT action."""

    def test_highlight_action_registered(self, mouse_executor):
        """Test that HIGHLIGHT action is in supported action types."""
        supported = mouse_executor.get_supported_action_types()
        assert "HIGHLIGHT" in supported

    def test_highlight_with_coordinates(self, mouse_executor):
        """Test HIGHLIGHT action with coordinate target."""
        # Arrange
        action = Action(
            id="test-highlight-1",
            type="HIGHLIGHT",
        )

        config = HighlightActionConfig(
            target=CoordinatesTarget(
                type="coordinates",
                coordinates=Coordinates(x=500, y=300),
            ),
            duration=2000,
            color="#FF0000",
            thickness=5,
            style="box",
        )

        # Mock the overlay
        with patch(
            "src.qontinui.action_executors.mouse.HighlightOverlay"
        ) as mock_overlay_class:
            mock_overlay = MagicMock()
            mock_overlay_class.return_value = mock_overlay

            # Act
            result = mouse_executor.execute(action, config)

            # Assert
            assert result is True
            mock_overlay_class.assert_called_once_with(
                x=500,
                y=300,
                duration_ms=2000,
                color="#FF0000",
                thickness=5,
                style="box",
            )
            mock_overlay.show.assert_called_once()

    def test_highlight_with_default_values(self, mouse_executor):
        """Test HIGHLIGHT action uses default values when not specified."""
        # Arrange
        action = Action(
            id="test-highlight-2",
            type="HIGHLIGHT",
        )

        config = HighlightActionConfig(
            target=CoordinatesTarget(
                type="coordinates",
                coordinates=Coordinates(x=100, y=200),
            ),
            # No duration, color, thickness, or style specified
        )

        # Mock the overlay
        with patch(
            "src.qontinui.action_executors.mouse.HighlightOverlay"
        ) as mock_overlay_class:
            mock_overlay = MagicMock()
            mock_overlay_class.return_value = mock_overlay

            # Act
            result = mouse_executor.execute(action, config)

            # Assert
            assert result is True
            # Check default values are used
            call_args = mock_overlay_class.call_args
            assert call_args.kwargs["duration_ms"] == 2000  # Default
            assert call_args.kwargs["color"] == "#FF0000"  # Default red
            assert call_args.kwargs["thickness"] == 3  # Default
            assert call_args.kwargs["style"] == "box"  # Default

    def test_highlight_with_circle_style(self, mouse_executor):
        """Test HIGHLIGHT action with circle style."""
        # Arrange
        action = Action(
            id="test-highlight-3",
            type="HIGHLIGHT",
        )

        config = HighlightActionConfig(
            target=CoordinatesTarget(
                type="coordinates",
                coordinates=Coordinates(x=700, y=400),
            ),
            duration=3000,
            color="#00FF00",
            thickness=4,
            style="circle",
        )

        # Mock the overlay
        with patch(
            "src.qontinui.action_executors.mouse.HighlightOverlay"
        ) as mock_overlay_class:
            mock_overlay = MagicMock()
            mock_overlay_class.return_value = mock_overlay

            # Act
            result = mouse_executor.execute(action, config)

            # Assert
            assert result is True
            call_args = mock_overlay_class.call_args
            assert call_args.kwargs["style"] == "circle"
            assert call_args.kwargs["color"] == "#00FF00"

    def test_highlight_with_arrow_style(self, mouse_executor):
        """Test HIGHLIGHT action with arrow style."""
        # Arrange
        action = Action(
            id="test-highlight-4",
            type="HIGHLIGHT",
        )

        config = HighlightActionConfig(
            target=CoordinatesTarget(
                type="coordinates",
                coordinates=Coordinates(x=900, y=600),
            ),
            duration=1500,
            color="#0000FF",
            thickness=6,
            style="arrow",
        )

        # Mock the overlay
        with patch(
            "src.qontinui.action_executors.mouse.HighlightOverlay"
        ) as mock_overlay_class:
            mock_overlay = MagicMock()
            mock_overlay_class.return_value = mock_overlay

            # Act
            result = mouse_executor.execute(action, config)

            # Assert
            assert result is True
            call_args = mock_overlay_class.call_args
            assert call_args.kwargs["style"] == "arrow"
            assert call_args.kwargs["color"] == "#0000FF"

    def test_highlight_without_target_location(self, mouse_executor):
        """Test HIGHLIGHT action fails gracefully when target not found."""
        # Arrange
        action = Action(
            id="test-highlight-5",
            type="HIGHLIGHT",
        )

        # Create a config with no valid target (will return None from _get_target_location)
        config = HighlightActionConfig(
            target=CoordinatesTarget(
                type="coordinates",
                coordinates=Coordinates(x=0, y=0),
            ),
        )

        # Mock _get_target_location to return None
        with patch.object(mouse_executor, "_get_target_location", return_value=None):
            # Act
            result = mouse_executor.execute(action, config)

            # Assert
            assert result is False

    def test_highlight_overlay_exception_handling(self, mouse_executor):
        """Test HIGHLIGHT action handles overlay exceptions gracefully."""
        # Arrange
        action = Action(
            id="test-highlight-6",
            type="HIGHLIGHT",
        )

        config = HighlightActionConfig(
            target=CoordinatesTarget(
                type="coordinates",
                coordinates=Coordinates(x=500, y=300),
            ),
        )

        # Mock the overlay to raise an exception
        with patch(
            "src.qontinui.action_executors.mouse.HighlightOverlay"
        ) as mock_overlay_class:
            mock_overlay_class.side_effect = RuntimeError("Overlay creation failed")

            # Act
            result = mouse_executor.execute(action, config)

            # Assert
            assert result is False


class TestHighlightOverlay:
    """Test suite for HighlightOverlay class."""

    def test_overlay_initialization(self):
        """Test that HighlightOverlay initializes correctly."""
        from src.qontinui.action_executors.highlight_overlay import HighlightOverlay

        overlay = HighlightOverlay(
            x=500,
            y=300,
            duration_ms=2000,
            color="#FF0000",
            thickness=5,
            style="box",
            size=100,
        )

        assert overlay.x == 500
        assert overlay.y == 300
        assert overlay.duration_ms == 2000
        assert overlay.color == "#FF0000"
        assert overlay.thickness == 5
        assert overlay.style == "box"
        assert overlay.size == 100

    def test_overlay_default_values(self):
        """Test that HighlightOverlay uses default values correctly."""
        from src.qontinui.action_executors.highlight_overlay import HighlightOverlay

        overlay = HighlightOverlay(x=100, y=200)

        assert overlay.x == 100
        assert overlay.y == 200
        assert overlay.duration_ms == 2000
        assert overlay.color == "#FF0000"
        assert overlay.thickness == 3
        assert overlay.style == "box"
        assert overlay.size == 100

    def test_overlay_show_spawns_thread(self):
        """Test that show() method spawns a thread (non-blocking)."""

        from src.qontinui.action_executors.highlight_overlay import HighlightOverlay

        overlay = HighlightOverlay(x=100, y=200, duration_ms=100)

        # Mock threading to avoid actually creating GUI
        with patch(
            "src.qontinui.action_executors.highlight_overlay.threading.Thread"
        ) as mock_thread:
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance

            # Act
            overlay.show()

            # Assert
            mock_thread.assert_called_once()
            mock_thread_instance.start.assert_called_once()
