"""Tests for UI-TARS data models."""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add src to path for direct import
src_path = Path(__file__).parent.parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from qontinui.extraction.runtime.uitars.models import (  # noqa: E402
    ActionResult,
    ExplorationTrajectory,
    GroundingResult,
    UITARSAction,
    UITARSActionType,
    UITARSInferenceRequest,
    UITARSInferenceResult,
    UITARSStep,
    UITARSThought,
)


class TestUITARSActionType:
    """Test UITARSActionType enum."""

    def test_all_action_types_defined(self):
        """Test all expected action types exist."""
        expected_types = [
            "click",
            "double_click",
            "right_click",
            "type",
            "scroll",
            "hover",
            "drag",
            "hotkey",
            "wait",
            "done",
        ]
        for expected in expected_types:
            assert hasattr(UITARSActionType, expected.upper())
            assert UITARSActionType[expected.upper()].value == expected

    def test_click_action_type(self):
        """Test click action type."""
        assert UITARSActionType.CLICK.value == "click"

    def test_done_action_type(self):
        """Test done action type indicates completion."""
        assert UITARSActionType.DONE.value == "done"


class TestUITARSThought:
    """Test UITARSThought dataclass."""

    def test_create_thought_with_reasoning(self):
        """Test creating thought with just reasoning."""
        thought = UITARSThought(reasoning="I see a submit button")
        assert thought.reasoning == "I see a submit button"
        assert thought.observation is None
        assert thought.goal_progress is None
        assert thought.next_step is None

    def test_create_thought_with_all_fields(self):
        """Test creating thought with all optional fields."""
        thought = UITARSThought(
            reasoning="I need to click the button",
            observation="There is a blue submit button",
            goal_progress="First step of form submission",
            next_step="Click the submit button",
        )
        assert thought.reasoning == "I need to click the button"
        assert thought.observation == "There is a blue submit button"
        assert thought.goal_progress == "First step of form submission"
        assert thought.next_step == "Click the submit button"


class TestUITARSAction:
    """Test UITARSAction dataclass."""

    def test_create_click_action(self):
        """Test creating a click action."""
        action = UITARSAction(
            action_type=UITARSActionType.CLICK,
            x=100,
            y=200,
        )
        assert action.action_type == UITARSActionType.CLICK
        assert action.x == 100
        assert action.y == 200
        assert action.confidence == 1.0

    def test_create_type_action(self):
        """Test creating a type action."""
        action = UITARSAction(
            action_type=UITARSActionType.TYPE,
            x=150,
            y=250,
            text="hello world",
        )
        assert action.action_type == UITARSActionType.TYPE
        assert action.text == "hello world"

    def test_create_scroll_action(self):
        """Test creating a scroll action."""
        action = UITARSAction(
            action_type=UITARSActionType.SCROLL,
            scroll_direction="down",
            scroll_amount=100,
        )
        assert action.action_type == UITARSActionType.SCROLL
        assert action.scroll_direction == "down"
        assert action.scroll_amount == 100

    def test_create_drag_action(self):
        """Test creating a drag action."""
        action = UITARSAction(
            action_type=UITARSActionType.DRAG,
            x=100,
            y=100,
            end_x=200,
            end_y=300,
        )
        assert action.action_type == UITARSActionType.DRAG
        assert action.x == 100
        assert action.y == 100
        assert action.end_x == 200
        assert action.end_y == 300

    def test_create_hotkey_action(self):
        """Test creating a hotkey action."""
        action = UITARSAction(
            action_type=UITARSActionType.HOTKEY,
            keys=["ctrl", "s"],
        )
        assert action.action_type == UITARSActionType.HOTKEY
        assert action.keys == ["ctrl", "s"]

    def test_create_wait_action(self):
        """Test creating a wait action."""
        action = UITARSAction(
            action_type=UITARSActionType.WAIT,
            duration=2.5,
        )
        assert action.action_type == UITARSActionType.WAIT
        assert action.duration == 2.5

    def test_default_confidence(self):
        """Test default confidence is 1.0."""
        action = UITARSAction(action_type=UITARSActionType.CLICK)
        assert action.confidence == 1.0

    def test_custom_confidence(self):
        """Test setting custom confidence."""
        action = UITARSAction(
            action_type=UITARSActionType.CLICK,
            x=100,
            y=200,
            confidence=0.85,
        )
        assert action.confidence == 0.85


class TestUITARSStep:
    """Test UITARSStep dataclass."""

    def test_create_step(self):
        """Test creating an exploration step."""
        thought = UITARSThought(reasoning="Click the button")
        action = UITARSAction(action_type=UITARSActionType.CLICK, x=100, y=200)

        step = UITARSStep(
            step_index=0,
            thought=thought,
            action=action,
        )

        assert step.step_index == 0
        assert step.thought == thought
        assert step.action == action
        assert step.success is True
        assert step.error is None

    def test_step_with_screenshots(self):
        """Test step with screenshot arrays."""
        thought = UITARSThought(reasoning="Test")
        action = UITARSAction(action_type=UITARSActionType.CLICK)

        before = np.zeros((100, 100, 3), dtype=np.uint8)
        after = np.ones((100, 100, 3), dtype=np.uint8) * 255

        step = UITARSStep(
            step_index=1,
            thought=thought,
            action=action,
            screenshot_before=before,
            screenshot_after=after,
        )

        assert step.screenshot_before is not None
        assert step.screenshot_after is not None
        assert step.screenshot_before.shape == (100, 100, 3)

    def test_step_with_error(self):
        """Test step that failed."""
        thought = UITARSThought(reasoning="Failed action")
        action = UITARSAction(action_type=UITARSActionType.CLICK)

        step = UITARSStep(
            step_index=2,
            thought=thought,
            action=action,
            success=False,
            error="Click failed: element not found",
        )

        assert step.success is False
        assert step.error == "Click failed: element not found"


class TestExplorationTrajectory:
    """Test ExplorationTrajectory dataclass."""

    def test_create_trajectory(self):
        """Test creating an empty trajectory."""
        trajectory = ExplorationTrajectory(
            trajectory_id="traj_123",
            goal="Explore settings menu",
        )

        assert trajectory.trajectory_id == "traj_123"
        assert trajectory.goal == "Explore settings menu"
        assert trajectory.total_steps == 0
        assert trajectory.successful_steps == 0
        assert trajectory.final_status == "incomplete"
        assert len(trajectory.steps) == 0

    def test_add_step(self):
        """Test adding steps to trajectory."""
        trajectory = ExplorationTrajectory(
            trajectory_id="traj_456",
            goal="Test goal",
        )

        thought = UITARSThought(reasoning="Test")
        action = UITARSAction(action_type=UITARSActionType.CLICK)
        step = UITARSStep(step_index=0, thought=thought, action=action)

        trajectory.add_step(step)

        assert trajectory.total_steps == 1
        assert trajectory.successful_steps == 1
        assert len(trajectory.steps) == 1

    def test_add_failed_step(self):
        """Test adding failed step to trajectory."""
        trajectory = ExplorationTrajectory(
            trajectory_id="traj_789",
            goal="Test goal",
        )

        thought = UITARSThought(reasoning="Test")
        action = UITARSAction(action_type=UITARSActionType.CLICK)
        step = UITARSStep(step_index=0, thought=thought, action=action, success=False)

        trajectory.add_step(step)

        assert trajectory.total_steps == 1
        assert trajectory.successful_steps == 0  # Failed step not counted

    def test_complete_trajectory(self):
        """Test completing a trajectory."""
        trajectory = ExplorationTrajectory(
            trajectory_id="traj_complete",
            goal="Complete test",
        )

        trajectory.complete(status="completed")

        assert trajectory.final_status == "completed"
        assert trajectory.completed_at is not None
        assert isinstance(trajectory.completed_at, datetime)

    def test_complete_with_timeout(self):
        """Test completing trajectory with timeout status."""
        trajectory = ExplorationTrajectory(
            trajectory_id="traj_timeout",
            goal="Timeout test",
        )

        trajectory.complete(status="timeout")

        assert trajectory.final_status == "timeout"


class TestGroundingResult:
    """Test GroundingResult dataclass."""

    def test_create_successful_grounding(self):
        """Test creating successful grounding result."""
        result = GroundingResult(
            x=500,
            y=300,
            confidence=0.95,
            element_description="Submit button",
            found_description="Found blue submit button",
        )

        assert result.x == 500
        assert result.y == 300
        assert result.confidence == 0.95
        assert result.element_description == "Submit button"

    def test_grounding_with_bbox(self):
        """Test grounding result with bounding box."""
        result = GroundingResult(
            x=500,
            y=300,
            confidence=0.9,
            bbox=(450, 280, 100, 40),
        )

        assert result.bbox == (450, 280, 100, 40)

    def test_failed_grounding(self):
        """Test failed grounding result."""
        result = GroundingResult(
            x=0,
            y=0,
            confidence=0.0,
            element_description="Missing element",
            found_description="Failed to find element",
        )

        assert result.confidence == 0.0


class TestActionResult:
    """Test ActionResult dataclass."""

    def test_create_successful_action(self):
        """Test creating successful action result."""
        thought = UITARSThought(reasoning="Clicking submit")
        action = UITARSAction(action_type=UITARSActionType.CLICK, x=100, y=200)

        result = ActionResult(
            success=True,
            thought=thought,
            action=action,
            execution_time_ms=150.5,
            state_changed=True,
        )

        assert result.success is True
        assert result.execution_time_ms == 150.5
        assert result.state_changed is True
        assert result.error is None

    def test_create_failed_action(self):
        """Test creating failed action result."""
        thought = UITARSThought(reasoning="Attempting click")
        action = UITARSAction(action_type=UITARSActionType.CLICK)

        result = ActionResult(
            success=False,
            thought=thought,
            action=action,
            error="Element not visible",
        )

        assert result.success is False
        assert result.error == "Element not visible"


class TestUITARSInferenceRequest:
    """Test UITARSInferenceRequest dataclass."""

    def test_create_basic_request(self):
        """Test creating basic inference request."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        request = UITARSInferenceRequest(
            image=image,
            prompt="Click the submit button",
        )

        assert request.prompt == "Click the submit button"
        assert request.image.shape == (480, 640, 3)
        assert request.max_new_tokens == 512
        assert request.temperature == 0.0

    def test_request_with_history(self):
        """Test request with conversation history."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        request = UITARSInferenceRequest(
            image=image,
            prompt="Next step",
            history=[
                ("click(100, 200)", "Clicked submit button"),
                ("type('hello')", "Typed text"),
            ],
        )

        assert len(request.history) == 2

    def test_request_with_custom_params(self):
        """Test request with custom parameters."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        request = UITARSInferenceRequest(
            image=image,
            prompt="Test",
            max_new_tokens=256,
            temperature=0.7,
            system_prompt="Custom system prompt",
        )

        assert request.max_new_tokens == 256
        assert request.temperature == 0.7
        assert request.system_prompt == "Custom system prompt"


class TestUITARSInferenceResult:
    """Test UITARSInferenceResult dataclass."""

    def test_create_successful_result(self):
        """Test creating successful inference result."""
        thought = UITARSThought(reasoning="Test reasoning")
        action = UITARSAction(action_type=UITARSActionType.CLICK, x=100, y=200)

        result = UITARSInferenceResult(
            thought=thought,
            action=action,
            raw_output="Thought: Test\nAction: click(100, 200)",
            inference_time_ms=250.0,
            tokens_used=50,
            model_name="UI-TARS-2B",
            provider="local_transformers",
        )

        assert result.thought == thought
        assert result.action == action
        assert result.inference_time_ms == 250.0
        assert result.success is True

    def test_create_failed_result(self):
        """Test creating failed inference result."""
        thought = UITARSThought(reasoning="Error")
        action = UITARSAction(action_type=UITARSActionType.WAIT)

        result = UITARSInferenceResult(
            thought=thought,
            action=action,
            raw_output="",
            inference_time_ms=100.0,
            success=False,
            error="Model unavailable",
        )

        assert result.success is False
        assert result.error == "Model unavailable"
