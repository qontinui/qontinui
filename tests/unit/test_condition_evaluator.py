"""Unit tests for ConditionEvaluator._evaluate_image_exists_condition().

This module tests the image existence checking functionality in control flow conditions.
Tests cover success cases, error handling, and proper integration with the registry
and FindAction components.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch

# Mock cv2 before importing qontinui modules to avoid DLL issues in tests
if "cv2" not in sys.modules:
    sys.modules["cv2"] = MagicMock()

import pytest

from qontinui.actions.control_flow.condition_evaluator import ConditionEvaluator
from qontinui.actions.find.find_result import FindResult
from qontinui.actions.find.matches import Matches
from qontinui.config import ConditionConfig
from qontinui.orchestration.execution_context import ExecutionContext


class TestEvaluateImageExistsCondition:
    """Test _evaluate_image_exists_condition method."""

    @pytest.mark.asyncio
    async def test_image_exists_with_valid_image_id_found(self):
        """Test image_exists condition when image is found on screen.

        Tests the happy path where:
        1. Image ID exists in registry
        2. FindAction.find() returns found=True
        """
        # Arrange
        context = ExecutionContext({})
        evaluator = ConditionEvaluator(context)
        condition = ConditionConfig(type="image_exists", image_id="test_button")

        # Create mock image and metadata
        mock_image = Mock()
        mock_image.name = "test_button"
        mock_metadata = {"file_path": "/path/to/test_button.png", "name": "test_button"}

        # Create mock find result
        mock_find_result = FindResult(
            matches=Matches(),
            found=True,
            pattern_name="test_button",
            duration_ms=50.0,
        )

        with (
            patch("qontinui.registry.get_image") as mock_get_image,
            patch("qontinui.registry.get_image_metadata") as mock_get_metadata,
            patch("qontinui.actions.find.FindAction") as mock_find_action_class,
            patch("qontinui.model.element.Pattern") as mock_pattern_class,
        ):
            # Configure mocks
            mock_get_image.return_value = mock_image
            mock_get_metadata.return_value = mock_metadata
            mock_pattern_class.from_file.return_value = Mock()
            mock_find_action = Mock()
            mock_find_action.find = AsyncMock(return_value=mock_find_result)
            mock_find_action_class.return_value = mock_find_action

            # Act
            result = await evaluator._evaluate_image_exists_condition(condition)

            # Assert
            assert result is True
            mock_get_image.assert_called_once_with("test_button")
            mock_find_action.find.assert_called_once()

    @pytest.mark.asyncio
    async def test_image_exists_with_valid_image_id_not_found(self):
        """Test image_exists condition when image is not found on screen.

        Tests the case where:
        1. Image ID exists in registry
        2. FindAction.find() returns found=False
        """
        # Arrange
        context = ExecutionContext({})
        evaluator = ConditionEvaluator(context)
        condition = ConditionConfig(type="image_exists", image_id="missing_button")

        # Create mock image and metadata
        mock_image = Mock()
        mock_metadata = {"file_path": "/path/to/missing_button.png", "name": "missing_button"}

        # Create mock find result - NOT found
        mock_find_result = FindResult(
            matches=Matches(),
            found=False,
            pattern_name="missing_button",
            duration_ms=50.0,
        )

        with (
            patch("qontinui.registry.get_image") as mock_get_image,
            patch("qontinui.registry.get_image_metadata") as mock_get_metadata,
            patch("qontinui.actions.find.FindAction") as mock_find_action_class,
            patch("qontinui.model.element.Pattern") as mock_pattern_class,
        ):
            # Configure mocks
            mock_get_image.return_value = mock_image
            mock_get_metadata.return_value = mock_metadata
            mock_pattern_class.from_file.return_value = Mock()
            mock_find_action = Mock()
            mock_find_action.find = AsyncMock(return_value=mock_find_result)
            mock_find_action_class.return_value = mock_find_action

            # Act
            result = await evaluator._evaluate_image_exists_condition(condition)

            # Assert
            assert result is False
            mock_get_image.assert_called_once_with("missing_button")
            mock_find_action.find.assert_called_once()

    @pytest.mark.asyncio
    async def test_image_exists_with_missing_image_id_raises_error(self):
        """Test that missing image_id raises ValueError with helpful message.

        Tests error handling when the condition configuration doesn't include
        an image_id field.
        """
        # Arrange
        context = ExecutionContext({})
        evaluator = ConditionEvaluator(context)
        condition = ConditionConfig(type="image_exists", image_id=None)

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            await evaluator._evaluate_image_exists_condition(condition)

        # Verify error message is helpful
        assert "Image condition requires 'image_id'" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_image_exists_with_image_not_in_registry_raises_error(self):
        """Test that image not found in registry raises ValueError with helpful message.

        Tests error handling when:
        1. Image ID is provided
        2. But registry.get_image() returns None (image not registered)
        """
        # Arrange
        context = ExecutionContext({})
        evaluator = ConditionEvaluator(context)
        condition = ConditionConfig(type="image_exists", image_id="unregistered_image")

        with patch("qontinui.registry.get_image") as mock_get_image:
            # Configure registry to return None (image not found)
            mock_get_image.return_value = None

            # Act & Assert
            with pytest.raises(ValueError) as exc_info:
                await evaluator._evaluate_image_exists_condition(condition)

            # Verify error message is helpful and includes image ID
            error_message = str(exc_info.value)
            assert "not found in registry" in error_message
            assert "unregistered_image" in error_message
            mock_get_image.assert_called_once_with("unregistered_image")

    @pytest.mark.asyncio
    async def test_image_vanished_calls_image_exists_and_negates_result(self):
        """Test that image_vanished condition properly negates image_exists result.

        Tests that the evaluate_condition method correctly handles image_vanished
        by calling _evaluate_image_exists_condition and negating the result.
        """
        # Arrange
        context = ExecutionContext({})
        evaluator = ConditionEvaluator(context)

        # Create mock image and metadata
        mock_image = Mock()
        mock_metadata = {"file_path": "/path/to/test_image.png", "name": "test_image"}

        # Test when image exists (vanished should return False)
        condition_exists = ConditionConfig(type="image_vanished", image_id="test_image")

        # Create mock find result - image IS found
        mock_find_result_found = FindResult(
            matches=Matches(),
            found=True,
            pattern_name="test_image",
            duration_ms=50.0,
        )

        with (
            patch("qontinui.registry.get_image") as mock_get_image,
            patch("qontinui.registry.get_image_metadata") as mock_get_metadata,
            patch("qontinui.actions.find.FindAction") as mock_find_action_class,
            patch("qontinui.model.element.Pattern") as mock_pattern_class,
        ):
            mock_get_image.return_value = mock_image
            mock_get_metadata.return_value = mock_metadata
            mock_pattern_class.from_file.return_value = Mock()
            mock_find_action = Mock()
            mock_find_action.find = AsyncMock(return_value=mock_find_result_found)
            mock_find_action_class.return_value = mock_find_action

            # Act
            result = await evaluator.evaluate_condition(condition_exists)

            # Assert - image exists, so vanished should be False
            assert result is False

        # Test when image doesn't exist (vanished should return True)
        mock_find_result_not_found = FindResult(
            matches=Matches(),
            found=False,
            pattern_name="test_image",
            duration_ms=50.0,
        )

        with (
            patch("qontinui.registry.get_image") as mock_get_image,
            patch("qontinui.registry.get_image_metadata") as mock_get_metadata,
            patch("qontinui.actions.find.FindAction") as mock_find_action_class,
            patch("qontinui.model.element.Pattern") as mock_pattern_class,
        ):
            mock_get_image.return_value = mock_image
            mock_get_metadata.return_value = mock_metadata
            mock_pattern_class.from_file.return_value = Mock()
            mock_find_action = Mock()
            mock_find_action.find = AsyncMock(return_value=mock_find_result_not_found)
            mock_find_action_class.return_value = mock_find_action

            # Act
            result = await evaluator.evaluate_condition(condition_exists)

            # Assert - image doesn't exist, so vanished should be True
            assert result is True

    @pytest.mark.asyncio
    async def test_image_exists_with_empty_string_image_id(self):
        """Test that empty string image_id is treated as missing.

        Verifies that an empty string for image_id triggers the same
        validation error as None.
        """
        # Arrange
        context = ExecutionContext({})
        evaluator = ConditionEvaluator(context)
        # Empty string should be falsy in Python
        condition = ConditionConfig(type="image_exists", image_id="")

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            await evaluator._evaluate_image_exists_condition(condition)

        assert "Image condition requires 'image_id'" in str(exc_info.value)


class TestImageExistsIntegrationWithEvaluateCondition:
    """Test image_exists through the public evaluate_condition method."""

    @pytest.mark.asyncio
    async def test_evaluate_condition_routes_to_image_exists(self):
        """Test that evaluate_condition correctly routes image_exists type.

        Verifies the public API properly delegates to the private method.
        """
        # Arrange
        context = ExecutionContext({})
        evaluator = ConditionEvaluator(context)
        condition = ConditionConfig(type="image_exists", image_id="public_test")

        mock_image = Mock()
        mock_metadata = {"file_path": "/path/to/public_test.png", "name": "public_test"}

        mock_find_result = FindResult(
            matches=Matches(),
            found=True,
            pattern_name="public_test",
            duration_ms=50.0,
        )

        with (
            patch("qontinui.registry.get_image") as mock_get_image,
            patch("qontinui.registry.get_image_metadata") as mock_get_metadata,
            patch("qontinui.actions.find.FindAction") as mock_find_action_class,
            patch("qontinui.model.element.Pattern") as mock_pattern_class,
        ):
            mock_get_image.return_value = mock_image
            mock_get_metadata.return_value = mock_metadata
            mock_pattern_class.from_file.return_value = Mock()
            mock_find_action = Mock()
            mock_find_action.find = AsyncMock(return_value=mock_find_result)
            mock_find_action_class.return_value = mock_find_action

            # Act
            result = await evaluator.evaluate_condition(condition)

            # Assert
            assert result is True
            mock_get_image.assert_called_once_with("public_test")

    @pytest.mark.asyncio
    async def test_evaluate_condition_with_image_exists_error_propagates(self):
        """Test that errors from _evaluate_image_exists_condition propagate correctly.

        Ensures error handling works through the public API.
        """
        # Arrange
        context = ExecutionContext({})
        evaluator = ConditionEvaluator(context)
        condition = ConditionConfig(type="image_exists", image_id=None)

        # Act & Assert
        with pytest.raises(ValueError, match="Image condition requires 'image_id'"):
            await evaluator.evaluate_condition(condition)


class TestErrorMessages:
    """Test that error messages are helpful and actionable."""

    @pytest.mark.asyncio
    async def test_missing_image_id_error_message_is_helpful(self):
        """Test that missing image_id error message guides the user."""
        context = ExecutionContext({})
        evaluator = ConditionEvaluator(context)
        condition = ConditionConfig(type="image_exists", image_id=None)

        with pytest.raises(ValueError) as exc_info:
            await evaluator._evaluate_image_exists_condition(condition)

        error_msg = str(exc_info.value)
        # Error should mention what's required
        assert "image_id" in error_msg.lower()
        assert "requires" in error_msg.lower()

    @pytest.mark.asyncio
    async def test_image_not_in_registry_error_message_includes_image_id(self):
        """Test that registry lookup failure includes the image ID in error."""
        context = ExecutionContext({})
        evaluator = ConditionEvaluator(context)
        image_id = "my_specific_button"
        condition = ConditionConfig(type="image_exists", image_id=image_id)

        with patch("qontinui.registry.get_image") as mock_get_image:
            mock_get_image.return_value = None

            with pytest.raises(ValueError) as exc_info:
                await evaluator._evaluate_image_exists_condition(condition)

            error_msg = str(exc_info.value)
            # Error should include the specific image ID that failed
            assert image_id in error_msg
            # Error should indicate it's a registry issue
            assert "registry" in error_msg.lower()

    @pytest.mark.asyncio
    async def test_error_message_format_for_troubleshooting(self):
        """Test that error messages follow a consistent, parseable format."""
        context = ExecutionContext({})
        evaluator = ConditionEvaluator(context)
        condition = ConditionConfig(type="image_exists", image_id="debug_button")

        with patch("qontinui.registry.get_image") as mock_get_image:
            mock_get_image.return_value = None

            with pytest.raises(ValueError) as exc_info:
                await evaluator._evaluate_image_exists_condition(condition)

            error_msg = str(exc_info.value)
            # Error should be in format: "Image 'id' not found in registry"
            # This makes it easy to parse and troubleshoot
            assert "Image" in error_msg
            assert "debug_button" in error_msg
            assert "not found in registry" in error_msg
