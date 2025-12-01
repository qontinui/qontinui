"""Unit tests for ConditionEvaluator._evaluate_image_exists_condition().

This module tests the image existence checking functionality in control flow conditions.
Tests cover success cases, error handling, and proper integration with the registry
and StateImage components.
"""

from unittest.mock import Mock, patch

import pytest

from qontinui.actions.control_flow.condition_evaluator import ConditionEvaluator
from qontinui.config import ConditionConfig
from qontinui.orchestration.execution_context import ExecutionContext


class TestEvaluateImageExistsCondition:
    """Test _evaluate_image_exists_condition method."""

    def test_image_exists_with_valid_image_id_found(self):
        """Test image_exists condition when image is found on screen.

        Tests the happy path where:
        1. Image ID exists in registry
        2. StateImage.exists() returns True
        """
        # Arrange
        context = ExecutionContext({})
        evaluator = ConditionEvaluator(context)
        condition = ConditionConfig(type="image_exists", image_id="test_button")

        # Create mock image
        mock_image = Mock()
        mock_image.name = "test_button"

        # Mock StateImage to return True for exists()
        mock_state_image = Mock()
        mock_state_image.exists.return_value = True

        with (
            patch("qontinui.registry.get_image") as mock_get_image,
            patch("qontinui.model.state.StateImage") as mock_state_image_class,
        ):

            # Configure mocks
            mock_get_image.return_value = mock_image
            mock_state_image_class.return_value = mock_state_image

            # Act
            result = evaluator._evaluate_image_exists_condition(condition)

            # Assert
            assert result is True
            mock_get_image.assert_called_once_with("test_button")
            mock_state_image_class.assert_called_once_with(image=mock_image, name="test_button")
            mock_state_image.exists.assert_called_once()

    def test_image_exists_with_valid_image_id_not_found(self):
        """Test image_exists condition when image is not found on screen.

        Tests the case where:
        1. Image ID exists in registry
        2. StateImage.exists() returns False
        """
        # Arrange
        context = ExecutionContext({})
        evaluator = ConditionEvaluator(context)
        condition = ConditionConfig(type="image_exists", image_id="missing_button")

        # Create mock image
        mock_image = Mock()
        mock_image.name = "missing_button"

        # Mock StateImage to return False for exists()
        mock_state_image = Mock()
        mock_state_image.exists.return_value = False

        with (
            patch("qontinui.registry.get_image") as mock_get_image,
            patch("qontinui.model.state.StateImage") as mock_state_image_class,
        ):

            # Configure mocks
            mock_get_image.return_value = mock_image
            mock_state_image_class.return_value = mock_state_image

            # Act
            result = evaluator._evaluate_image_exists_condition(condition)

            # Assert
            assert result is False
            mock_get_image.assert_called_once_with("missing_button")
            mock_state_image_class.assert_called_once_with(image=mock_image, name="missing_button")
            mock_state_image.exists.assert_called_once()

    def test_image_exists_with_missing_image_id_raises_error(self):
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
            evaluator._evaluate_image_exists_condition(condition)

        # Verify error message is helpful
        assert "Image condition requires 'image_id'" in str(exc_info.value)

    def test_image_exists_with_image_not_in_registry_raises_error(self):
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
                evaluator._evaluate_image_exists_condition(condition)

            # Verify error message is helpful and includes image ID
            error_message = str(exc_info.value)
            assert "not found in registry" in error_message
            assert "unregistered_image" in error_message
            mock_get_image.assert_called_once_with("unregistered_image")

    def test_image_vanished_calls_image_exists_and_negates_result(self):
        """Test that image_vanished condition properly negates image_exists result.

        Tests that the evaluate_condition method correctly handles image_vanished
        by calling _evaluate_image_exists_condition and negating the result.
        """
        # Arrange
        context = ExecutionContext({})
        evaluator = ConditionEvaluator(context)

        # Create mock image
        mock_image = Mock()
        mock_image.name = "test_image"

        # Test when image exists (vanished should return False)
        condition_exists = ConditionConfig(type="image_vanished", image_id="test_image")

        with (
            patch("qontinui.registry.get_image") as mock_get_image,
            patch("qontinui.model.state.StateImage") as mock_state_image_class,
        ):

            mock_state_image = Mock()
            mock_state_image.exists.return_value = True
            mock_get_image.return_value = mock_image
            mock_state_image_class.return_value = mock_state_image

            # Act
            result = evaluator.evaluate_condition(condition_exists)

            # Assert - image exists, so vanished should be False
            assert result is False

        # Test when image doesn't exist (vanished should return True)
        with (
            patch("qontinui.registry.get_image") as mock_get_image,
            patch("qontinui.model.state.StateImage") as mock_state_image_class,
        ):

            mock_state_image = Mock()
            mock_state_image.exists.return_value = False
            mock_get_image.return_value = mock_image
            mock_state_image_class.return_value = mock_state_image

            # Act
            result = evaluator.evaluate_condition(condition_exists)

            # Assert - image doesn't exist, so vanished should be True
            assert result is True

    def test_image_exists_with_complex_image_object(self):
        """Test image_exists with a more complex Image object.

        Ensures the method works correctly with realistic Image objects
        that have multiple attributes.
        """
        # Arrange
        context = ExecutionContext({})
        evaluator = ConditionEvaluator(context)
        condition = ConditionConfig(type="image_exists", image_id="complex_button")

        # Create more realistic mock image with attributes
        mock_image = Mock()
        mock_image.name = "complex_button"
        mock_image.path = "/path/to/button.png"
        mock_image.size = (100, 50)

        # Mock StateImage
        mock_state_image = Mock()
        mock_state_image.exists.return_value = True

        with (
            patch("qontinui.registry.get_image") as mock_get_image,
            patch("qontinui.model.state.StateImage") as mock_state_image_class,
        ):

            mock_get_image.return_value = mock_image
            mock_state_image_class.return_value = mock_state_image

            # Act
            result = evaluator._evaluate_image_exists_condition(condition)

            # Assert
            assert result is True
            # Verify StateImage was created with correct parameters
            mock_state_image_class.assert_called_once_with(image=mock_image, name="complex_button")

    def test_image_exists_state_image_creation_parameters(self):
        """Test that StateImage is created with correct parameters.

        Verifies the exact parameters passed to StateImage constructor
        to ensure proper integration.
        """
        # Arrange
        context = ExecutionContext({})
        evaluator = ConditionEvaluator(context)
        image_id = "verification_image"
        condition = ConditionConfig(type="image_exists", image_id=image_id)

        mock_image = Mock()

        with (
            patch("qontinui.registry.get_image") as mock_get_image,
            patch("qontinui.model.state.StateImage") as mock_state_image_class,
        ):

            mock_get_image.return_value = mock_image
            mock_state_image_instance = Mock()
            mock_state_image_instance.exists.return_value = True
            mock_state_image_class.return_value = mock_state_image_instance

            # Act
            evaluator._evaluate_image_exists_condition(condition)

            # Assert - verify exact constructor call
            mock_state_image_class.assert_called_once()
            call_kwargs = mock_state_image_class.call_args[1]
            assert call_kwargs["image"] is mock_image
            assert call_kwargs["name"] == image_id

    def test_image_exists_logging_on_success(self, caplog):
        """Test that successful image existence check logs appropriately.

        Verifies that debug logging includes relevant information about
        the check and its result.
        """
        # Arrange
        import logging

        context = ExecutionContext({})
        evaluator = ConditionEvaluator(context)
        condition = ConditionConfig(type="image_exists", image_id="logged_button")

        mock_image = Mock()
        mock_state_image = Mock()
        mock_state_image.exists.return_value = True

        with (
            patch("qontinui.registry.get_image") as mock_get_image,
            patch("qontinui.model.state.StateImage") as mock_state_image_class,
        ):

            mock_get_image.return_value = mock_image
            mock_state_image_class.return_value = mock_state_image

            # Act
            with caplog.at_level(logging.DEBUG):
                result = evaluator._evaluate_image_exists_condition(condition)

            # Assert
            assert result is True
            # Check that logging includes the image_id
            log_messages = [record.message for record in caplog.records]
            assert any("logged_button" in msg for msg in log_messages)

    def test_image_exists_logging_on_error(self, caplog):
        """Test that registry lookup failure logs error with helpful information.

        Verifies that error logging includes the image ID and appropriate
        error level when an image is not found in the registry.
        """
        # Arrange
        import logging

        context = ExecutionContext({})
        evaluator = ConditionEvaluator(context)
        condition = ConditionConfig(type="image_exists", image_id="missing_from_registry")

        with patch("qontinui.registry.get_image") as mock_get_image:
            mock_get_image.return_value = None

            # Act & Assert
            with caplog.at_level(logging.ERROR):
                with pytest.raises(ValueError):
                    evaluator._evaluate_image_exists_condition(condition)

            # Verify error was logged
            error_records = [r for r in caplog.records if r.levelname == "ERROR"]
            assert len(error_records) > 0
            assert any("missing_from_registry" in r.message for r in error_records)

    def test_image_exists_with_empty_string_image_id(self):
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
            evaluator._evaluate_image_exists_condition(condition)

        assert "Image condition requires 'image_id'" in str(exc_info.value)

    def test_multiple_image_checks_in_sequence(self):
        """Test multiple image existence checks can be performed sequentially.

        Ensures that the method can be called multiple times with different
        image IDs without state corruption.
        """
        # Arrange
        context = ExecutionContext({})
        evaluator = ConditionEvaluator(context)

        mock_image1 = Mock()
        mock_image2 = Mock()

        condition1 = ConditionConfig(type="image_exists", image_id="image1")
        condition2 = ConditionConfig(type="image_exists", image_id="image2")

        with (
            patch("qontinui.registry.get_image") as mock_get_image,
            patch("qontinui.model.state.StateImage") as mock_state_image_class,
        ):

            # First call - image1 exists
            mock_state_image1 = Mock()
            mock_state_image1.exists.return_value = True
            mock_get_image.return_value = mock_image1
            mock_state_image_class.return_value = mock_state_image1

            result1 = evaluator._evaluate_image_exists_condition(condition1)
            assert result1 is True

            # Second call - image2 doesn't exist
            mock_state_image2 = Mock()
            mock_state_image2.exists.return_value = False
            mock_get_image.return_value = mock_image2
            mock_state_image_class.return_value = mock_state_image2

            result2 = evaluator._evaluate_image_exists_condition(condition2)
            assert result2 is False

            # Verify both calls were made
            assert mock_get_image.call_count == 2
            assert mock_state_image_class.call_count == 2


class TestImageExistsIntegrationWithEvaluateCondition:
    """Test image_exists through the public evaluate_condition method."""

    def test_evaluate_condition_routes_to_image_exists(self):
        """Test that evaluate_condition correctly routes image_exists type.

        Verifies the public API properly delegates to the private method.
        """
        # Arrange
        context = ExecutionContext({})
        evaluator = ConditionEvaluator(context)
        condition = ConditionConfig(type="image_exists", image_id="public_test")

        mock_image = Mock()
        mock_state_image = Mock()
        mock_state_image.exists.return_value = True

        with (
            patch("qontinui.registry.get_image") as mock_get_image,
            patch("qontinui.model.state.StateImage") as mock_state_image_class,
        ):

            mock_get_image.return_value = mock_image
            mock_state_image_class.return_value = mock_state_image

            # Act
            result = evaluator.evaluate_condition(condition)

            # Assert
            assert result is True
            mock_get_image.assert_called_once_with("public_test")

    def test_evaluate_condition_with_image_exists_error_propagates(self):
        """Test that errors from _evaluate_image_exists_condition propagate correctly.

        Ensures error handling works through the public API.
        """
        # Arrange
        context = ExecutionContext({})
        evaluator = ConditionEvaluator(context)
        condition = ConditionConfig(type="image_exists", image_id=None)

        # Act & Assert
        with pytest.raises(ValueError, match="Image condition requires 'image_id'"):
            evaluator.evaluate_condition(condition)


class TestErrorMessages:
    """Test that error messages are helpful and actionable."""

    def test_missing_image_id_error_message_is_helpful(self):
        """Test that missing image_id error message guides the user."""
        context = ExecutionContext({})
        evaluator = ConditionEvaluator(context)
        condition = ConditionConfig(type="image_exists", image_id=None)

        with pytest.raises(ValueError) as exc_info:
            evaluator._evaluate_image_exists_condition(condition)

        error_msg = str(exc_info.value)
        # Error should mention what's required
        assert "image_id" in error_msg.lower()
        assert "requires" in error_msg.lower()

    def test_image_not_in_registry_error_message_includes_image_id(self):
        """Test that registry lookup failure includes the image ID in error."""
        context = ExecutionContext({})
        evaluator = ConditionEvaluator(context)
        image_id = "my_specific_button"
        condition = ConditionConfig(type="image_exists", image_id=image_id)

        with patch("qontinui.registry.get_image") as mock_get_image:
            mock_get_image.return_value = None

            with pytest.raises(ValueError) as exc_info:
                evaluator._evaluate_image_exists_condition(condition)

            error_msg = str(exc_info.value)
            # Error should include the specific image ID that failed
            assert image_id in error_msg
            # Error should indicate it's a registry issue
            assert "registry" in error_msg.lower()

    def test_error_message_format_for_troubleshooting(self):
        """Test that error messages follow a consistent, parseable format."""
        context = ExecutionContext({})
        evaluator = ConditionEvaluator(context)
        condition = ConditionConfig(type="image_exists", image_id="debug_button")

        with patch("qontinui.registry.get_image") as mock_get_image:
            mock_get_image.return_value = None

            with pytest.raises(ValueError) as exc_info:
                evaluator._evaluate_image_exists_condition(condition)

            error_msg = str(exc_info.value)
            # Error should be in format: "Image 'id' not found in registry"
            # This makes it easy to parse and troubleshoot
            assert "Image" in error_msg
            assert "debug_button" in error_msg
            assert "not found in registry" in error_msg
