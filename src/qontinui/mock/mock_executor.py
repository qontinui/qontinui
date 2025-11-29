"""Mock execution engine for integration testing with visualization support.

Executes processes in mock mode using recorded Action Snapshots and generates
visualization data for frontend display.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .snapshot import ActionHistory, ActionRecord
from .state_screenshot import ActionVisualization, StateScreenshotRegistry

logger = logging.getLogger(__name__)


@dataclass
class MockExecutionResult:
    """Result of mock process execution with visualization data.

    Contains complete execution trace with visualization information for
    each action, enabling frontend to replay and visualize the execution.

    Attributes:
        process_id: ID of process that was executed
        process_name: Name of process
        start_time: Execution start time
        end_time: Execution end time
        total_duration_ms: Total execution duration
        initial_states: Initial active states
        final_states: Final active states
        actions: List of action visualizations in execution order
        success: Whether entire process succeeded
        success_rate: Percentage of successful actions
    """

    process_id: str
    process_name: str
    start_time: datetime
    end_time: datetime | None = None
    total_duration_ms: float = 0.0
    initial_states: set[str] = field(default_factory=set)
    final_states: set[str] = field(default_factory=set)
    actions: list[ActionVisualization] = field(default_factory=list)
    success: bool = True
    success_rate: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "process_id": self.process_id,
            "process_name": self.process_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_ms": self.total_duration_ms,
            "initial_states": sorted(self.initial_states),
            "final_states": sorted(self.final_states),
            "actions": [a.to_dict() for a in self.actions],
            "success": self.success,
            "success_rate": self.success_rate,
            "total_actions": len(self.actions),
            "successful_actions": sum(1 for a in self.actions if a.success),
        }


class MockExecutor:
    """Mock execution engine for integration testing.

    Executes processes using recorded Action Snapshots, matching actions to
    historical executions based on active states. Generates visualization
    data for frontend replay.
    """

    def __init__(
        self,
        action_histories: dict[str, ActionHistory],
        screenshot_registry: StateScreenshotRegistry,
    ) -> None:
        """Initialize mock executor.

        Args:
            action_histories: Map of pattern_id -> ActionHistory
            screenshot_registry: Registry of state-screenshot associations
        """
        self.action_histories = action_histories
        self.screenshot_registry = screenshot_registry
        self.current_states: set[str] = set()

    def execute_process(
        self,
        process_id: str,
        process_name: str,
        actions: list[dict[str, Any]],
        initial_states: set[str],
    ) -> MockExecutionResult:
        """Execute a process in mock mode with visualization.

        Args:
            process_id: Process identifier
            process_name: Process name
            actions: List of action specifications
            initial_states: Initial active states

        Returns:
            MockExecutionResult with visualization data
        """
        logger.info(f"Starting mock execution: {process_name} (ID: {process_id})")
        logger.info(f"Initial states: {initial_states}")

        start_time = datetime.now()
        self.current_states = initial_states.copy()

        result = MockExecutionResult(
            process_id=process_id,
            process_name=process_name,
            start_time=start_time,
            initial_states=initial_states,
        )

        # Execute each action
        for action_spec in actions:
            action_viz = self._execute_action(action_spec)
            result.actions.append(action_viz)

            # Update states based on action results
            self._update_states_from_action(action_viz)

        # Finalize result
        end_time = datetime.now()
        result.end_time = end_time
        result.total_duration_ms = (end_time - start_time).total_seconds() * 1000
        result.final_states = self.current_states.copy()

        # Calculate success metrics
        successful_count = sum(1 for a in result.actions if a.success)
        total_count = len(result.actions)
        result.success_rate = successful_count / total_count if total_count > 0 else 0.0
        result.success = result.success_rate == 1.0

        logger.info(f"Mock execution complete: {successful_count}/{total_count} actions succeeded")
        logger.info(f"Final states: {result.final_states}")

        return result

    def _execute_action(self, action_spec: dict[str, Any]) -> ActionVisualization:
        """Execute a single action in mock mode.

        Args:
            action_spec: Action specification with type and parameters

        Returns:
            ActionVisualization with results
        """
        action_type = action_spec.get("type", "UNKNOWN")
        pattern_id = action_spec.get("pattern_id")

        logger.debug(
            f"Executing {action_type} (pattern: {pattern_id}, states: {self.current_states})"
        )

        # Find best screenshot for current states
        screenshot = self.screenshot_registry.find_screenshot(self.current_states)
        screenshot_path = screenshot.screenshot_path if screenshot else ""

        # Find historical action from action history
        action_record = None
        if pattern_id and pattern_id in self.action_histories:
            history = self.action_histories[pattern_id]
            # Get matches for current states
            matches = history.get_matches_for_states(self.current_states)

            # Find the ActionRecord that produced these matches
            for record in history.snapshots:
                if record.action_success and record.match_list == matches:
                    action_record = record
                    break

        # Create visualization based on action type
        if action_type in ["FIND", "FIND_ALL"]:
            return self._create_find_visualization(action_spec, action_record, screenshot_path)
        elif action_type == "CLICK":
            return self._create_click_visualization(action_spec, action_record, screenshot_path)
        elif action_type == "TYPE":
            return self._create_type_visualization(action_spec, action_record, screenshot_path)
        else:
            # Generic action
            return ActionVisualization(
                action_type=action_type,
                screenshot_path=screenshot_path,
                active_states=self.current_states.copy(),
                success=action_record.action_success if action_record else True,
                duration_ms=action_record.duration * 1000 if action_record else 100.0,
            )

    def _create_find_visualization(
        self,
        action_spec: dict[str, Any],
        action_record: ActionRecord | None,
        screenshot_path: str,
    ) -> ActionVisualization:
        """Create visualization for FIND action."""
        if action_record and action_record.match_list:
            # Extract match regions from Match objects
            matches = []
            for match in action_record.match_list:
                region = match.target.region if match.target and match.target.region else None
                if region:
                    matches.append(
                        {
                            "x": region.x,
                            "y": region.y,
                            "w": region.w,
                            "h": region.h,
                            "score": match.score,
                        }
                    )

            # Use first match region as action region
            action_region = matches[0] if matches else None

            return ActionVisualization(
                action_type="FIND",
                screenshot_path=screenshot_path,
                action_region=action_region,  # type: ignore[arg-type]
                success=action_record.action_success,
                matches=matches,
                active_states=self.current_states.copy(),
                timestamp=action_record.timestamp,
                duration_ms=action_record.duration * 1000,
            )
        else:
            # No match found
            return ActionVisualization(
                action_type="FIND",
                screenshot_path=screenshot_path,
                success=False,
                active_states=self.current_states.copy(),
                duration_ms=100.0,
            )

    def _create_click_visualization(
        self,
        action_spec: dict[str, Any],
        action_record: ActionRecord | None,
        screenshot_path: str,
    ) -> ActionVisualization:
        """Create visualization for CLICK action."""
        # Get click location from action record's first match
        action_location = None
        if action_record and action_record.match_list:
            match = action_record.match_list[0]
            region = match.target.region if match.target and match.target.region else None
            if region:
                # Click at center of match region
                action_location = (
                    region.x + region.w // 2,
                    region.y + region.h // 2,
                )

        return ActionVisualization(
            action_type="CLICK",
            screenshot_path=screenshot_path,
            action_location=action_location,
            success=action_record.action_success if action_record else True,
            active_states=self.current_states.copy(),
            timestamp=action_record.timestamp if action_record else datetime.now(),
            duration_ms=action_record.duration * 1000 if action_record else 50.0,
        )

    def _create_type_visualization(
        self,
        action_spec: dict[str, Any],
        action_record: ActionRecord | None,
        screenshot_path: str,
    ) -> ActionVisualization:
        """Create visualization for TYPE action."""
        text = action_spec.get("text", "")
        if action_record:
            text = action_record.text or text

        # Get type location from action record if available
        action_location = None
        if action_record and action_record.match_list:
            match = action_record.match_list[0]
            region = match.target.region if match.target and match.target.region else None
            if region:
                action_location = (region.x, region.y)

        return ActionVisualization(
            action_type="TYPE",
            screenshot_path=screenshot_path,
            action_location=action_location,
            text=text,
            success=action_record.action_success if action_record else True,
            active_states=self.current_states.copy(),
            timestamp=action_record.timestamp if action_record else datetime.now(),
            duration_ms=action_record.duration * 1000 if action_record else 200.0,
        )

    def _update_states_from_action(self, action_viz: ActionVisualization):
        """Update current states based on action results.

        Mock mode uses recorded snapshots and does not modify state during execution.
        State transitions are captured in the recorded data and replayed as-is.

        Args:
            action_viz: Action that was just executed
        """
        # States remain unchanged in mock mode - transitions are pre-recorded
        pass
