"""Trajectory converter for UI-TARS exploration results.

Converts UI-TARS ExplorationTrajectory to Qontinui StateStructure format,
enabling integration with the existing RAG system and state machine.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .models import ExplorationTrajectory, UITARSActionType, UITARSStep

logger = logging.getLogger(__name__)


@dataclass
class ConvertedState:
    """A state converted from UI-TARS exploration."""

    id: str
    name: str
    description: str
    screenshot_path: Path | None = None
    screenshot_hash: str | None = None
    elements: list[dict[str, Any]] = field(default_factory=list)
    source_step_index: int = 0
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConvertedTransition:
    """A transition converted from UI-TARS exploration."""

    id: str
    from_state_id: str
    to_state_id: str
    action_type: str
    target_x: int | None = None
    target_y: int | None = None
    action_value: str | None = None
    thought: str | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversionResult:
    """Result of trajectory to StateStructure conversion."""

    states: list[ConvertedState]
    transitions: list[ConvertedTransition]
    images: list[dict[str, Any]]
    metadata: dict[str, Any]
    output_path: Path | None = None


class TrajectoryConverter:
    """Converts UI-TARS exploration trajectories to Qontinui StateStructure format.

    The converter:
    1. Groups exploration steps by visual state (using screenshot hashing)
    2. Creates states from unique visual states
    3. Creates transitions from actions between states
    4. Exports to Qontinui-compatible JSON format

    Example:
        converter = TrajectoryConverter()
        result = converter.convert(trajectory, output_dir=Path("./output"))
        # result.output_path contains the state_structure.json path
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        min_state_confidence: float = 0.5,
    ) -> None:
        """Initialize converter.

        Args:
            similarity_threshold: Threshold for considering two screenshots as same state
            min_state_confidence: Minimum confidence for including a state
        """
        self.similarity_threshold = similarity_threshold
        self.min_state_confidence = min_state_confidence

    def convert(
        self,
        trajectory: ExplorationTrajectory,
        output_dir: Path | None = None,
        project_name: str | None = None,
    ) -> ConversionResult:
        """Convert exploration trajectory to StateStructure format.

        Args:
            trajectory: UI-TARS exploration trajectory
            output_dir: Directory to save output (creates state_structure.json)
            project_name: Optional project name for the output

        Returns:
            ConversionResult with states, transitions, and metadata
        """
        logger.info(f"Converting trajectory with {len(trajectory.steps)} steps")

        # Group steps by visual state
        state_groups = self._group_by_visual_state(trajectory.steps)

        # Create states from groups
        states = self._create_states(state_groups, trajectory.goal)

        # Create transitions from step sequences
        transitions = self._create_transitions(trajectory.steps, states)

        # Create image entries for StateImages
        images = self._create_image_entries(states)

        # Build metadata
        metadata = {
            "source": "uitars_exploration",
            "trajectory_id": trajectory.trajectory_id,
            "goal": trajectory.goal,
            "total_steps": trajectory.total_steps,
            "successful_steps": trajectory.successful_steps,
            "final_status": trajectory.final_status,
            "started_at": trajectory.started_at.isoformat(),
            "completed_at": (
                trajectory.completed_at.isoformat()
                if trajectory.completed_at
                else None
            ),
        }

        result = ConversionResult(
            states=states,
            transitions=transitions,
            images=images,
            metadata=metadata,
        )

        # Export if output_dir provided
        if output_dir:
            result.output_path = self._export(
                result, output_dir, project_name or "uitars_exploration"
            )

        return result

    def _group_by_visual_state(
        self,
        steps: list[UITARSStep],
    ) -> dict[str, list[UITARSStep]]:
        """Group steps by their visual state (screenshot hash).

        Args:
            steps: List of exploration steps

        Returns:
            Dictionary mapping state hash to list of steps
        """
        groups: dict[str, list[UITARSStep]] = {}

        for step in steps:
            # Use before screenshot for state identification
            if step.screenshot_before is not None:
                state_hash = self._hash_image(step.screenshot_before)
            elif step.screenshot_before_path:
                state_hash = self._hash_file(step.screenshot_before_path)
            else:
                # No screenshot, use step index as fallback
                state_hash = f"step_{step.step_index}"

            if state_hash not in groups:
                groups[state_hash] = []
            groups[state_hash].append(step)

        logger.debug(f"Grouped {len(steps)} steps into {len(groups)} visual states")
        return groups

    def _create_states(
        self,
        state_groups: dict[str, list[UITARSStep]],
        goal: str,
    ) -> list[ConvertedState]:
        """Create states from grouped steps.

        Args:
            state_groups: Steps grouped by visual state
            goal: Exploration goal for naming context

        Returns:
            List of ConvertedState objects
        """
        states = []

        for i, (state_hash, steps) in enumerate(state_groups.items()):
            # Use first step for state info
            first_step = steps[0]

            # Generate state name from thought
            thought_text = first_step.thought.reasoning[:50] if first_step.thought else ""
            state_name = self._generate_state_name(thought_text, i)

            # Get screenshot path
            screenshot_path = first_step.screenshot_before_path

            state = ConvertedState(
                id=f"state_{uuid.uuid4().hex[:8]}",
                name=state_name,
                description=f"State discovered during exploration: {goal}",
                screenshot_path=screenshot_path,
                screenshot_hash=state_hash,
                source_step_index=first_step.step_index,
                confidence=1.0,
                metadata={
                    "thought": first_step.thought.reasoning if first_step.thought else None,
                    "step_count": len(steps),
                    "step_indices": [s.step_index for s in steps],
                },
            )
            states.append(state)

        return states

    def _create_transitions(
        self,
        steps: list[UITARSStep],
        states: list[ConvertedState],
    ) -> list[ConvertedTransition]:
        """Create transitions from action steps.

        Args:
            steps: Exploration steps
            states: Converted states

        Returns:
            List of ConvertedTransition objects
        """
        transitions = []

        # Build hash to state mapping
        hash_to_state = {s.screenshot_hash: s for s in states}

        for i, step in enumerate(steps):
            if i >= len(steps) - 1:
                continue  # No transition from last step

            # Determine from and to states
            from_hash = self._get_step_hash(step, "before")
            to_hash = self._get_step_hash(steps[i + 1], "before")

            from_state = hash_to_state.get(from_hash)
            to_state = hash_to_state.get(to_hash)

            if from_state and to_state and from_state.id != to_state.id:
                # State changed, create transition
                action_type = self._map_action_type(step.action.action_type)

                transition = ConvertedTransition(
                    id=f"transition_{uuid.uuid4().hex[:8]}",
                    from_state_id=from_state.id,
                    to_state_id=to_state.id,
                    action_type=action_type,
                    target_x=step.action.x,
                    target_y=step.action.y,
                    action_value=step.action.text,
                    thought=step.thought.reasoning if step.thought else None,
                    confidence=step.action.confidence,
                    metadata={
                        "step_index": step.step_index,
                        "execution_time_ms": step.execution_time_ms,
                    },
                )
                transitions.append(transition)

        return transitions

    def _create_image_entries(
        self,
        states: list[ConvertedState],
    ) -> list[dict[str, Any]]:
        """Create image entries for the state structure.

        Args:
            states: Converted states

        Returns:
            List of image entry dictionaries
        """
        images = []

        for state in states:
            if state.screenshot_path:
                images.append({
                    "id": f"img_{state.id}",
                    "name": f"{state.name} screenshot",
                    "path": str(state.screenshot_path),
                    "state_id": state.id,
                    "source": "uitars_exploration",
                })

        return images

    def _export(
        self,
        result: ConversionResult,
        output_dir: Path,
        project_name: str,
    ) -> Path:
        """Export conversion result to state_structure.json.

        Args:
            result: Conversion result
            output_dir: Output directory
            project_name: Project name

        Returns:
            Path to exported JSON file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create images directory and copy screenshots
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        for state in result.states:
            if state.screenshot_path and state.screenshot_path.exists():
                import shutil

                dest = images_dir / f"{state.id}.png"
                shutil.copy2(state.screenshot_path, dest)

        # Build structure
        structure = {
            "version": "1.0",
            "project_name": project_name,
            "extracted_at": datetime.now().isoformat(),
            "source": "uitars_exploration",
            "states": [
                {
                    "id": s.id,
                    "name": s.name,
                    "description": s.description,
                    "screenshot": f"images/{s.id}.png",
                    "confidence": s.confidence,
                    "metadata": s.metadata,
                }
                for s in result.states
            ],
            "transitions": [
                {
                    "id": t.id,
                    "from_state": t.from_state_id,
                    "to_state": t.to_state_id,
                    "action_type": t.action_type,
                    "target": {"x": t.target_x, "y": t.target_y}
                    if t.target_x is not None
                    else None,
                    "action_value": t.action_value,
                    "thought": t.thought,
                    "confidence": t.confidence,
                    "metadata": t.metadata,
                }
                for t in result.transitions
            ],
            "images": result.images,
            "metadata": result.metadata,
        }

        # Save JSON
        output_path = output_dir / "state_structure.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(structure, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported state structure to {output_path}")
        return output_path

    def _hash_image(self, image: np.ndarray[Any, Any]) -> str:
        """Create hash from image array.

        Args:
            image: Image as numpy array

        Returns:
            MD5 hash string
        """
        # Downsample for faster hashing
        small = image[::10, ::10]
        return hashlib.md5(small.tobytes()).hexdigest()

    def _hash_file(self, path: Path) -> str:
        """Create hash from image file.

        Args:
            path: Path to image file

        Returns:
            MD5 hash string
        """
        try:
            img = Image.open(path)
            arr = np.array(img)
            return self._hash_image(arr)
        except Exception:
            return hashlib.md5(str(path).encode()).hexdigest()

    def _get_step_hash(self, step: UITARSStep, which: str = "before") -> str:
        """Get hash for a step's screenshot.

        Args:
            step: Exploration step
            which: "before" or "after"

        Returns:
            Hash string
        """
        if which == "before":
            if step.screenshot_before is not None:
                return self._hash_image(step.screenshot_before)
            elif step.screenshot_before_path:
                return self._hash_file(step.screenshot_before_path)
        else:
            if step.screenshot_after is not None:
                return self._hash_image(step.screenshot_after)
            elif step.screenshot_after_path:
                return self._hash_file(step.screenshot_after_path)

        return f"step_{step.step_index}_{which}"

    def _generate_state_name(self, thought: str, index: int) -> str:
        """Generate a state name from thought text.

        Args:
            thought: Thought reasoning text
            index: State index

        Returns:
            Generated state name
        """
        if not thought:
            return f"State {index + 1}"

        # Extract key words from thought
        words = thought.split()[:5]
        if words:
            name = " ".join(w.capitalize() for w in words if len(w) > 2)
            if name:
                return name[:50]  # Truncate

        return f"State {index + 1}"

    def _map_action_type(self, action_type: UITARSActionType) -> str:
        """Map UI-TARS action type to Qontinui action type.

        Args:
            action_type: UI-TARS action type

        Returns:
            Qontinui action type string
        """
        mapping = {
            UITARSActionType.CLICK: "click",
            UITARSActionType.DOUBLE_CLICK: "double_click",
            UITARSActionType.RIGHT_CLICK: "right_click",
            UITARSActionType.TYPE: "type",
            UITARSActionType.SCROLL: "scroll",
            UITARSActionType.HOVER: "hover",
            UITARSActionType.DRAG: "drag",
            UITARSActionType.HOTKEY: "hotkey",
            UITARSActionType.WAIT: "wait",
            UITARSActionType.DONE: "done",
        }
        return mapping.get(action_type, "click")
