"""Data models for UI-TARS integration.

This module defines the core data structures used by UI-TARS for:
- Inference requests and responses
- Thought-Action decomposition
- Grounding and execution results
- Exploration trajectories
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class UITARSActionType(Enum):
    """Action types supported by UI-TARS."""

    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    TYPE = "type"
    SCROLL = "scroll"
    HOVER = "hover"
    DRAG = "drag"
    HOTKEY = "hotkey"
    WAIT = "wait"
    DONE = "done"  # Indicates task completion


@dataclass
class UITARSThought:
    """Represents UI-TARS thought/reasoning output.

    UI-TARS produces explicit reasoning before each action,
    explaining its understanding and decision process.
    """

    reasoning: str  # Full reasoning text from the model
    observation: str | None = None  # What UI-TARS observes on screen
    goal_progress: str | None = None  # Progress toward the task goal
    next_step: str | None = None  # Planned next action explanation


@dataclass
class UITARSAction:
    """Represents a UI-TARS action output.

    UI-TARS outputs actions in a specific format with coordinates
    relative to the image dimensions.
    """

    action_type: UITARSActionType
    x: int | None = None  # X coordinate (absolute pixels)
    y: int | None = None  # Y coordinate (absolute pixels)
    text: str | None = None  # Text to type (for TYPE action)
    scroll_direction: str | None = None  # up/down/left/right
    scroll_amount: int | None = None  # Pixels to scroll
    keys: list[str] | None = None  # Hotkey combination
    duration: float | None = None  # Wait duration in seconds
    end_x: int | None = None  # End X for drag actions
    end_y: int | None = None  # End Y for drag actions
    confidence: float = 1.0  # Action confidence score
    raw_output: str | None = None  # Original model output


@dataclass
class UITARSStep:
    """A single step in a UI-TARS exploration trajectory.

    Combines thought, action, and the resulting state change.
    """

    step_index: int
    thought: UITARSThought
    action: UITARSAction
    screenshot_before: np.ndarray[Any, Any] | None = None
    screenshot_after: np.ndarray[Any, Any] | None = None
    screenshot_before_path: Path | None = None
    screenshot_after_path: Path | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    execution_time_ms: float = 0.0
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExplorationTrajectory:
    """Complete trajectory from a UI-TARS exploration session.

    Captures the entire sequence of steps taken during exploration,
    along with metadata about the session.
    """

    trajectory_id: str
    goal: str  # The exploration goal/task
    steps: list[UITARSStep] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    final_status: str = "incomplete"  # completed, incomplete, failed, timeout
    total_steps: int = 0
    successful_steps: int = 0
    unique_states_discovered: int = 0
    output_dir: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: UITARSStep) -> None:
        """Add a step to the trajectory."""
        self.steps.append(step)
        self.total_steps += 1
        if step.success:
            self.successful_steps += 1

    def complete(self, status: str = "completed") -> None:
        """Mark the trajectory as complete."""
        self.completed_at = datetime.now()
        self.final_status = status


@dataclass
class GroundingResult:
    """Result of UI-TARS visual grounding.

    When UI-TARS is asked to find an element, this captures
    the grounding result with coordinates and confidence.
    """

    x: int  # Center X coordinate of grounded element
    y: int  # Center Y coordinate of grounded element
    confidence: float  # Grounding confidence (0.0-1.0)
    bbox: tuple[int, int, int, int] | None = None  # x, y, width, height
    element_description: str | None = None  # What was searched for
    found_description: str | None = None  # What UI-TARS thinks it found
    raw_output: str | None = None  # Raw model output
    inference_time_ms: float = 0.0


@dataclass
class ActionResult:
    """Result of executing an action via UI-TARS.

    Captures the full result of a UI-TARS action execution,
    including the thought process and outcome.
    """

    success: bool
    thought: UITARSThought
    action: UITARSAction
    grounding: GroundingResult | None = None
    screenshot_before: np.ndarray[Any, Any] | None = None
    screenshot_after: np.ndarray[Any, Any] | None = None
    execution_time_ms: float = 0.0
    error: str | None = None
    state_changed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UITARSInferenceRequest:
    """Request for UI-TARS inference.

    Contains all information needed to make an inference request
    to a UI-TARS provider.
    """

    image: np.ndarray[Any, Any]  # Screenshot as numpy array (RGB)
    prompt: str  # Task or query for UI-TARS
    history: list[tuple[str, str]] | None = None  # Previous (action, observation) pairs
    system_prompt: str | None = None  # Optional system prompt override
    max_new_tokens: int = 512
    temperature: float = 0.0  # Use greedy decoding by default
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UITARSInferenceResult:
    """Result from UI-TARS inference.

    Contains the parsed thought and action along with raw output
    and timing information.
    """

    thought: UITARSThought
    action: UITARSAction
    raw_output: str
    inference_time_ms: float
    tokens_used: int = 0
    model_name: str = ""
    provider: str = ""
    success: bool = True
    error: str | None = None
