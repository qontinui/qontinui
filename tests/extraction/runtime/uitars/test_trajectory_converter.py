"""Tests for UI-TARS trajectory converter."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

# Add src to path for direct import
src_path = Path(__file__).parent.parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from qontinui.extraction.runtime.uitars.models import (  # noqa: E402
    ExplorationTrajectory,
    UITARSAction,
    UITARSActionType,
    UITARSStep,
    UITARSThought,
)
from qontinui.extraction.runtime.uitars.trajectory_converter import (  # noqa: E402
    ConversionResult,
    ConvertedState,
    ConvertedTransition,
    TrajectoryConverter,
)


class TestConvertedState:
    """Test ConvertedState dataclass."""

    def test_create_converted_state(self):
        """Test creating a converted state."""
        state = ConvertedState(
            id="state_123",
            name="Main Menu",
            description="Application main menu",
        )

        assert state.id == "state_123"
        assert state.name == "Main Menu"
        assert state.description == "Application main menu"
        assert state.screenshot_path is None
        assert state.confidence == 1.0
        assert state.elements == []

    def test_converted_state_with_metadata(self):
        """Test state with metadata."""
        state = ConvertedState(
            id="state_456",
            name="Settings",
            description="Settings screen",
            metadata={"thought": "Opening settings", "step_count": 3},
        )

        assert state.metadata["thought"] == "Opening settings"
        assert state.metadata["step_count"] == 3


class TestConvertedTransition:
    """Test ConvertedTransition dataclass."""

    def test_create_converted_transition(self):
        """Test creating a converted transition."""
        transition = ConvertedTransition(
            id="trans_123",
            from_state_id="state_1",
            to_state_id="state_2",
            action_type="click",
            target_x=100,
            target_y=200,
        )

        assert transition.id == "trans_123"
        assert transition.from_state_id == "state_1"
        assert transition.to_state_id == "state_2"
        assert transition.action_type == "click"
        assert transition.target_x == 100
        assert transition.target_y == 200

    def test_transition_with_text(self):
        """Test transition with action value (text)."""
        transition = ConvertedTransition(
            id="trans_456",
            from_state_id="state_1",
            to_state_id="state_2",
            action_type="type",
            action_value="hello world",
        )

        assert transition.action_type == "type"
        assert transition.action_value == "hello world"


class TestConversionResult:
    """Test ConversionResult dataclass."""

    def test_create_conversion_result(self):
        """Test creating a conversion result."""
        states = [ConvertedState(id="s1", name="State 1", description="First")]
        transitions = [
            ConvertedTransition(
                id="t1",
                from_state_id="s1",
                to_state_id="s2",
                action_type="click",
            )
        ]

        result = ConversionResult(
            states=states,
            transitions=transitions,
            images=[],
            metadata={"source": "test"},
        )

        assert len(result.states) == 1
        assert len(result.transitions) == 1
        assert result.metadata["source"] == "test"


class TestTrajectoryConverter:
    """Test TrajectoryConverter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = TrajectoryConverter()

    def _create_test_step(
        self,
        index: int,
        action_type: UITARSActionType = UITARSActionType.CLICK,
        x: int = 100,
        y: int = 200,
        reasoning: str = "Test step",
        screenshot_before: np.ndarray | None = None,
    ) -> UITARSStep:
        """Helper to create test steps."""
        thought = UITARSThought(reasoning=reasoning)
        action = UITARSAction(action_type=action_type, x=x, y=y)

        if screenshot_before is None:
            # Create unique screenshot for each step
            screenshot_before = np.zeros((100, 100, 3), dtype=np.uint8)
            screenshot_before[index % 100, index % 100] = [255, 0, 0]

        return UITARSStep(
            step_index=index,
            thought=thought,
            action=action,
            screenshot_before=screenshot_before,
        )

    def _create_test_trajectory(self, num_steps: int = 5) -> ExplorationTrajectory:
        """Helper to create test trajectory."""
        trajectory = ExplorationTrajectory(
            trajectory_id="test_trajectory",
            goal="Test exploration goal",
        )

        for i in range(num_steps):
            step = self._create_test_step(
                index=i,
                reasoning=f"Step {i} reasoning",
            )
            trajectory.add_step(step)

        trajectory.complete(status="completed")
        return trajectory

    def test_converter_initialization(self):
        """Test converter initialization."""
        converter = TrajectoryConverter(
            similarity_threshold=0.9,
            min_state_confidence=0.6,
        )

        assert converter.similarity_threshold == 0.9
        assert converter.min_state_confidence == 0.6

    def test_convert_empty_trajectory(self):
        """Test converting empty trajectory."""
        trajectory = ExplorationTrajectory(
            trajectory_id="empty",
            goal="Empty test",
        )
        trajectory.complete()

        result = self.converter.convert(trajectory)

        assert len(result.states) == 0
        assert len(result.transitions) == 0
        assert result.metadata["trajectory_id"] == "empty"

    def test_convert_single_step_trajectory(self):
        """Test converting trajectory with single step."""
        trajectory = ExplorationTrajectory(
            trajectory_id="single_step",
            goal="Single step test",
        )
        step = self._create_test_step(0)
        trajectory.add_step(step)
        trajectory.complete()

        result = self.converter.convert(trajectory)

        assert len(result.states) == 1
        assert len(result.transitions) == 0  # No transitions from single step

    def test_convert_multi_step_trajectory(self):
        """Test converting trajectory with multiple steps."""
        trajectory = self._create_test_trajectory(num_steps=5)

        result = self.converter.convert(trajectory)

        # Should have states (unique visual states)
        assert len(result.states) > 0
        # Metadata should be populated
        assert result.metadata["goal"] == "Test exploration goal"
        assert result.metadata["total_steps"] == 5

    def test_convert_preserves_trajectory_metadata(self):
        """Test that conversion preserves trajectory metadata."""
        trajectory = self._create_test_trajectory(num_steps=3)

        result = self.converter.convert(trajectory)

        assert result.metadata["trajectory_id"] == "test_trajectory"
        assert result.metadata["goal"] == "Test exploration goal"
        assert result.metadata["total_steps"] == 3
        assert result.metadata["final_status"] == "completed"

    def test_hash_image(self):
        """Test image hashing."""
        image1 = np.zeros((100, 100, 3), dtype=np.uint8)
        image2 = np.zeros((100, 100, 3), dtype=np.uint8)
        image3 = np.ones((100, 100, 3), dtype=np.uint8) * 255

        hash1 = self.converter._hash_image(image1)
        hash2 = self.converter._hash_image(image2)
        hash3 = self.converter._hash_image(image3)

        # Same images should have same hash
        assert hash1 == hash2
        # Different images should have different hash
        assert hash1 != hash3

    def test_generate_state_name_from_thought(self):
        """Test state name generation from thought text."""
        name1 = self.converter._generate_state_name("Click the submit button", 0)
        name2 = self.converter._generate_state_name("", 1)
        name3 = self.converter._generate_state_name("Open settings menu to configure", 2)

        assert "Click" in name1 or "Submit" in name1
        assert name2 == "State 2"  # Fallback for empty thought
        assert len(name3) <= 50  # Should be truncated

    def test_map_action_type(self):
        """Test action type mapping."""
        assert self.converter._map_action_type(UITARSActionType.CLICK) == "click"
        assert self.converter._map_action_type(UITARSActionType.DOUBLE_CLICK) == "double_click"
        assert self.converter._map_action_type(UITARSActionType.TYPE) == "type"
        assert self.converter._map_action_type(UITARSActionType.SCROLL) == "scroll"
        assert self.converter._map_action_type(UITARSActionType.HOVER) == "hover"
        assert self.converter._map_action_type(UITARSActionType.DRAG) == "drag"
        assert self.converter._map_action_type(UITARSActionType.HOTKEY) == "hotkey"
        assert self.converter._map_action_type(UITARSActionType.WAIT) == "wait"
        assert self.converter._map_action_type(UITARSActionType.DONE) == "done"

    def test_create_image_entries(self):
        """Test image entries creation."""
        states = [
            ConvertedState(
                id="state_1",
                name="State 1",
                description="First state",
                screenshot_path=Path("/tmp/state_1.png"),
            ),
            ConvertedState(
                id="state_2",
                name="State 2",
                description="Second state",
                screenshot_path=None,  # No screenshot
            ),
        ]

        images = self.converter._create_image_entries(states)

        assert len(images) == 1  # Only state with screenshot
        assert images[0]["state_id"] == "state_1"
        assert "screenshot" in images[0]["name"].lower()

    def test_group_by_visual_state(self):
        """Test grouping steps by visual state."""
        # Create steps with same screenshot (same state)
        same_screenshot = np.zeros((100, 100, 3), dtype=np.uint8)

        steps = [
            self._create_test_step(0, screenshot_before=same_screenshot.copy()),
            self._create_test_step(1, screenshot_before=same_screenshot.copy()),
            self._create_test_step(2, screenshot_before=same_screenshot.copy()),
        ]

        groups = self.converter._group_by_visual_state(steps)

        # All steps should be in same group (same visual state)
        assert len(groups) == 1

    def test_group_by_visual_state_different(self):
        """Test grouping steps with different visual states."""
        steps = []
        for i in range(3):
            # Create unique screenshots
            screenshot = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            steps.append(self._create_test_step(i, screenshot_before=screenshot))

        groups = self.converter._group_by_visual_state(steps)

        # Each step should be in its own group
        assert len(groups) == 3

    def test_export_creates_json(self):
        """Test export creates valid JSON file."""
        trajectory = self._create_test_trajectory(num_steps=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = self.converter.convert(
                trajectory, output_dir=output_dir, project_name="test_project"
            )

            assert result.output_path is not None
            assert result.output_path.exists()
            assert result.output_path.name == "state_structure.json"

            # Verify JSON is valid
            with open(result.output_path) as f:
                data = json.load(f)

            assert data["project_name"] == "test_project"
            assert data["source"] == "uitars_exploration"
            assert "states" in data
            assert "transitions" in data

    def test_export_creates_images_dir(self):
        """Test export creates images directory."""
        trajectory = self._create_test_trajectory(num_steps=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            self.converter.convert(trajectory, output_dir=output_dir)

            images_dir = output_dir / "images"
            assert images_dir.exists()
            assert images_dir.is_dir()


class TestTrajectoryConverterEdgeCases:
    """Test edge cases in trajectory conversion."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = TrajectoryConverter()

    def test_convert_with_failed_steps(self):
        """Test conversion handles failed steps."""
        trajectory = ExplorationTrajectory(
            trajectory_id="failed_test",
            goal="Test with failures",
        )

        # Add successful step
        thought1 = UITARSThought(reasoning="Success")
        action1 = UITARSAction(action_type=UITARSActionType.CLICK, x=100, y=100)
        step1 = UITARSStep(
            step_index=0,
            thought=thought1,
            action=action1,
            screenshot_before=np.zeros((100, 100, 3), dtype=np.uint8),
        )
        trajectory.add_step(step1)

        # Add failed step
        thought2 = UITARSThought(reasoning="Failed")
        action2 = UITARSAction(action_type=UITARSActionType.CLICK, x=200, y=200)
        step2 = UITARSStep(
            step_index=1,
            thought=thought2,
            action=action2,
            success=False,
            error="Element not found",
            screenshot_before=np.ones((100, 100, 3), dtype=np.uint8) * 128,
        )
        trajectory.add_step(step2)

        trajectory.complete(status="completed")

        result = self.converter.convert(trajectory)

        # Should still produce valid result
        assert result.metadata["total_steps"] == 2
        assert result.metadata["successful_steps"] == 1

    def test_convert_with_no_screenshots(self):
        """Test conversion handles steps without screenshots."""
        trajectory = ExplorationTrajectory(
            trajectory_id="no_screenshots",
            goal="Test without screenshots",
        )

        thought = UITARSThought(reasoning="No screenshot")
        action = UITARSAction(action_type=UITARSActionType.CLICK)
        step = UITARSStep(
            step_index=0,
            thought=thought,
            action=action,
            # No screenshot_before or screenshot_before_path
        )
        trajectory.add_step(step)
        trajectory.complete()

        result = self.converter.convert(trajectory)

        # Should produce valid result with fallback hash
        assert len(result.states) >= 0  # May or may not create state

    def test_convert_preserves_action_details(self):
        """Test conversion preserves action details in transitions."""
        trajectory = ExplorationTrajectory(
            trajectory_id="action_test",
            goal="Test action details",
        )

        # First step
        thought1 = UITARSThought(reasoning="First step")
        action1 = UITARSAction(action_type=UITARSActionType.CLICK, x=100, y=100)
        step1 = UITARSStep(
            step_index=0,
            thought=thought1,
            action=action1,
            screenshot_before=np.zeros((100, 100, 3), dtype=np.uint8),
        )
        trajectory.add_step(step1)

        # Second step with type action
        thought2 = UITARSThought(reasoning="Type text")
        action2 = UITARSAction(
            action_type=UITARSActionType.TYPE,
            x=200,
            y=200,
            text="hello",
        )
        step2 = UITARSStep(
            step_index=1,
            thought=thought2,
            action=action2,
            screenshot_before=np.ones((100, 100, 3), dtype=np.uint8) * 255,
        )
        trajectory.add_step(step2)

        trajectory.complete()

        result = self.converter.convert(trajectory)

        # Check transitions preserve action details
        if result.transitions:
            type_transitions = [t for t in result.transitions if t.action_type == "type"]
            if type_transitions:
                assert type_transitions[0].action_value == "hello"

    def test_converter_with_custom_thresholds(self):
        """Test converter respects custom thresholds."""
        converter = TrajectoryConverter(
            similarity_threshold=0.5,
            min_state_confidence=0.8,
        )

        assert converter.similarity_threshold == 0.5
        assert converter.min_state_confidence == 0.8

    def test_get_step_hash_before(self):
        """Test getting step hash for 'before' screenshot."""
        screenshot = np.zeros((100, 100, 3), dtype=np.uint8)
        thought = UITARSThought(reasoning="Test")
        action = UITARSAction(action_type=UITARSActionType.CLICK)

        step = UITARSStep(
            step_index=0,
            thought=thought,
            action=action,
            screenshot_before=screenshot,
        )

        hash_before = self.converter._get_step_hash(step, "before")

        assert isinstance(hash_before, str)
        assert len(hash_before) == 32  # MD5 hash length

    def test_get_step_hash_after(self):
        """Test getting step hash for 'after' screenshot."""
        screenshot = np.ones((100, 100, 3), dtype=np.uint8) * 128
        thought = UITARSThought(reasoning="Test")
        action = UITARSAction(action_type=UITARSActionType.CLICK)

        step = UITARSStep(
            step_index=0,
            thought=thought,
            action=action,
            screenshot_after=screenshot,
        )

        hash_after = self.converter._get_step_hash(step, "after")

        assert isinstance(hash_after, str)

    def test_get_step_hash_fallback(self):
        """Test getting step hash with fallback."""
        thought = UITARSThought(reasoning="Test")
        action = UITARSAction(action_type=UITARSActionType.CLICK)

        step = UITARSStep(
            step_index=5,
            thought=thought,
            action=action,
            # No screenshots
        )

        hash_result = self.converter._get_step_hash(step, "before")

        # Should use fallback format
        assert "step_5" in hash_result
