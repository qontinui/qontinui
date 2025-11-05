"""Tests for RunProcess action with repetition support."""

from dataclasses import dataclass, field
from typing import Any

import pytest

from qontinui.actions.action_result import ActionResult
from qontinui.actions.action_type import ActionType
from qontinui.actions.composite.process import (
    ProcessRepetitionOptionsBuilder,
    RunProcess,
    RunProcessOptionsBuilder,
)


# Mock classes for testing
@dataclass
class MockAction:
    """Mock action for testing."""

    id: str
    type: str
    config: dict[str, Any]
    timeout: int = 5000
    retry_count: int = 3
    continue_on_error: bool = False


@dataclass
class MockProcess:
    """Mock process for testing."""

    id: str
    name: str
    description: str
    type: str
    actions: list[MockAction] = field(default_factory=list)


@dataclass
class MockQontinuiConfig:
    """Mock configuration for testing."""

    processes: list[MockProcess]
    process_map: dict[str, MockProcess] = field(default_factory=dict)

    def __post_init__(self):
        """Build process map."""
        self.process_map = {p.id: p for p in self.processes}


class TestProcessRepetitionOptions:
    """Test ProcessRepetitionOptions configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        options = ProcessRepetitionOptionsBuilder().build()

        assert options.get_enabled() is False
        assert options.get_max_repeats() == 10
        assert options.get_delay() == 0.0
        assert options.get_until_success() is False

    def test_custom_values(self):
        """Test custom configuration values."""
        options = (
            ProcessRepetitionOptionsBuilder()
            .set_enabled(True)
            .set_max_repeats(5)
            .set_delay(2.5)
            .set_until_success(True)
            .build()
        )

        assert options.get_enabled() is True
        assert options.get_max_repeats() == 5
        assert options.get_delay() == 2.5
        assert options.get_until_success() is True

    def test_builder_from_existing(self):
        """Test creating builder from existing options."""
        original = (
            ProcessRepetitionOptionsBuilder()
            .set_enabled(True)
            .set_max_repeats(3)
            .set_delay(1.0)
            .set_until_success(False)
            .build()
        )

        copy = original.to_builder().build()

        assert copy.get_enabled() == original.get_enabled()
        assert copy.get_max_repeats() == original.get_max_repeats()
        assert copy.get_delay() == original.get_delay()
        assert copy.get_until_success() == original.get_until_success()


class TestRunProcessOptions:
    """Test RunProcessOptions configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        options = RunProcessOptionsBuilder().build()

        assert options.get_process_id() == ""
        assert options.get_process_repetition().get_enabled() is False

    def test_process_id(self):
        """Test setting process ID."""
        options = RunProcessOptionsBuilder().set_process_id("test_process").build()

        assert options.get_process_id() == "test_process"

    def test_enable_repetition_shorthand(self):
        """Test enable_repetition convenience method."""
        options = (
            RunProcessOptionsBuilder()
            .set_process_id("test_process")
            .enable_repetition(max_repeats=5, delay=1.0, until_success=True)
            .build()
        )

        assert options.get_process_id() == "test_process"
        repetition = options.get_process_repetition()
        assert repetition.get_enabled() is True
        assert repetition.get_max_repeats() == 5
        assert repetition.get_delay() == 1.0
        assert repetition.get_until_success() is True

    def test_set_process_repetition(self):
        """Test setting repetition via builder."""
        repetition_builder = (
            ProcessRepetitionOptionsBuilder()
            .set_enabled(True)
            .set_max_repeats(3)
            .set_delay(0.5)
            .set_until_success(False)
        )

        options = (
            RunProcessOptionsBuilder()
            .set_process_id("test_process")
            .set_process_repetition(repetition_builder)
            .build()
        )

        repetition = options.get_process_repetition()
        assert repetition.get_enabled() is True
        assert repetition.get_max_repeats() == 3
        assert repetition.get_delay() == 0.5
        assert repetition.get_until_success() is False

    def test_builder_chaining(self):
        """Test fluent builder API with pause options."""
        options = (
            RunProcessOptionsBuilder()
            .set_process_id("test_process")
            .set_pause_before_begin(0.5)
            .set_pause_after_end(1.0)
            .enable_repetition(max_repeats=2, delay=0.1)
            .build()
        )

        assert options.get_process_id() == "test_process"
        assert options.get_pause_before_begin() == 0.5
        assert options.get_pause_after_end() == 1.0
        assert options.get_process_repetition().get_enabled() is True


class TestRunProcess:
    """Test RunProcess action implementation."""

    def test_action_type(self):
        """Test that action returns correct type."""
        action = RunProcess()
        assert action.get_action_type() == ActionType.RUN_PROCESS

    def test_config_injection(self):
        """Test setting configuration."""
        config = MockQontinuiConfig(processes=[])
        action = RunProcess()
        action.set_config(config)

        assert action.config is config

    def test_missing_config_raises_error(self):
        """Test that missing config raises error."""
        action = RunProcess()
        options = RunProcessOptionsBuilder().set_process_id("test").build()
        action_result = ActionResult(options)

        with pytest.raises(RuntimeError, match="requires QontinuiConfig"):
            action.perform(action_result)

    def test_no_process_id(self):
        """Test handling of missing process ID."""
        config = MockQontinuiConfig(processes=[])
        action = RunProcess(config)

        options = RunProcessOptionsBuilder().build()  # Empty process_id
        action_result = ActionResult(options)

        action.perform(action_result)

        assert action_result.is_success is False
        assert "No process ID" in action_result.get_output_text()

    def test_process_not_found(self):
        """Test handling of non-existent process."""
        config = MockQontinuiConfig(processes=[])
        action = RunProcess(config)

        options = RunProcessOptionsBuilder().set_process_id("nonexistent").build()
        action_result = ActionResult(options)

        action.perform(action_result)

        assert action_result.is_success is False
        assert "not found" in action_result.get_output_text()

    def test_single_execution_no_repetition(self):
        """Test executing process once without repetition."""
        process = MockProcess(
            id="test_process",
            name="Test Process",
            description="Test",
            type="sequence",
            actions=[
                MockAction(id="action1", type="FIND", config={}),
                MockAction(id="action2", type="CLICK", config={}),
            ],
        )
        config = MockQontinuiConfig(processes=[process])
        action = RunProcess(config)

        options = RunProcessOptionsBuilder().set_process_id("test_process").build()
        action_result = ActionResult(options)

        action.perform(action_result)

        # Should execute (success depends on mock implementation)
        assert "Test Process" in action_result.get_output_text()

    def test_invalid_options_type(self):
        """Test that invalid options type raises error."""
        from qontinui.actions.basic.click.click_options import ClickOptions

        config = MockQontinuiConfig(processes=[])
        action = RunProcess(config)

        # Pass wrong options type
        wrong_options = ClickOptions.builder().build()
        action_result = ActionResult(wrong_options)

        with pytest.raises(ValueError, match="requires RunProcessOptions"):
            action.perform(action_result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
