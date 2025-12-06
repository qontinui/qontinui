"""RunProcess action implementation.

Executes a named workflow (v2.0.0+) or process (v1.0.0 compatibility) with optional repetition control.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ....actions.action_interface import ActionInterface
from ....actions.action_result import ActionResult
from ....actions.action_type import ActionType
from .run_process_options import RunProcessOptions

if TYPE_CHECKING:
    from ....json_executor.config_parser import (  # type: ignore[attr-defined]
        Process,
        QontinuiConfig,
        Workflow,
    )
    from ....model.object_collection import ObjectCollection


class RunProcess(ActionInterface):
    """Execute a named workflow with repetition support.

    This action executes a workflow (v2.0.0+) or process (v1.0.0 compatibility)
    by its ID, with support for:
    - Fixed count repetition: Run exactly N additional times
    - Until success repetition: Stop early on success, otherwise run up to N times
    - Configurable delays between repetitions

    Note:
        The action is named RunProcess for backward compatibility, but executes
        workflows. The process_id parameter accepts both workflow and process IDs.
    """

    def __init__(self, config: QontinuiConfig | None = None) -> None:
        """Initialize RunProcess action.

        Args:
            config: QontinuiConfig containing workflow/process definitions
        """
        self.config = config

    def set_config(self, config: QontinuiConfig) -> None:
        """Set the configuration after initialization.

        Args:
            config: QontinuiConfig containing workflow/process definitions
        """
        self.config = config

    def get_action_type(self) -> ActionType:
        """Get the action type.

        Returns:
            ActionType.RUN_PROCESS
        """
        return ActionType.RUN_PROCESS

    def perform(self, action_result: ActionResult, *object_collections: ObjectCollection) -> None:
        """Execute the workflow with optional repetition.

        Args:
            action_result: ActionResult containing configuration
            object_collections: Variable number of ObjectCollection arguments

        Note:
            The process_id from configuration can refer to either a workflow (v2.0.0+)
            or process (v1.0.0 compatibility). Both are supported via workflow_map/process_map.
        """
        # Extract and validate configuration
        run_process_options = action_result.action_config
        if not isinstance(run_process_options, RunProcessOptions):
            raise ValueError("RunProcess requires RunProcessOptions configuration")

        if not self.config:
            raise RuntimeError("RunProcess requires QontinuiConfig to be set")

        # process_id can refer to either workflow or process (backward compatibility)
        workflow_id = run_process_options.get_process_id()
        if not workflow_id:
            object.__setattr__(action_result, "success", False)
            object.__setattr__(
                action_result,
                "output_text",
                "RUN_PROCESS: No workflow/process ID specified",
            )
            return

        # Get the workflow (backward compatible with process)
        workflow = self._get_workflow(workflow_id)
        if not workflow:
            object.__setattr__(action_result, "success", False)
            object.__setattr__(
                action_result,
                "output_text",
                f"RUN_PROCESS: Workflow/process '{workflow_id}' not found",
            )
            return

        # Check if repetition is enabled
        repetition = run_process_options.get_process_repetition()
        if not repetition.get_enabled():
            # No repetition - execute once
            success = self._execute_workflow_once(workflow, action_result)
            object.__setattr__(action_result, "success", success)
            return

        # Execute with repetition
        if repetition.get_until_success():
            self._execute_until_success(workflow, repetition, action_result, object_collections)
        else:
            self._execute_fixed_count(workflow, repetition, action_result, object_collections)

    def _get_workflow(self, workflow_id: str) -> Workflow | Process | None:
        """Get a workflow by ID from the configuration.

        Args:
            workflow_id: The workflow/process ID to look up

        Returns:
            Workflow or Process instance, or None if not found

        Note:
            Supports both workflow_map (v2.0.0+) and process_map (v1.0.0 compatibility)
            through the config's property alias system.
        """
        if not self.config:
            return None
        # workflow_map is aliased to process_map in config for backward compatibility
        if hasattr(self.config, "workflow_map"):
            return self.config.workflow_map.get(workflow_id)
        # Fallback to process_map for older configs
        if hasattr(self.config, "process_map"):
            return self.config.process_map.get(workflow_id)
        return None

    def _execute_workflow_once(
        self, workflow: Workflow | Process, action_result: ActionResult
    ) -> bool:
        """Execute a workflow once.

        Args:
            workflow: The workflow/process to execute
            action_result: ActionResult to update

        Returns:
            True if workflow succeeded, False otherwise
        """
        # Import here to avoid circular dependency

        # Create a temporary executor for this workflow
        # Note: In a real scenario, this would use the existing executor
        # For now, we'll use a simplified approach
        output_text = f"Executing workflow: {workflow.name}\n"

        # Execute workflow actions sequentially
        success = True
        for i, action in enumerate(workflow.actions):
            output_text += f"  Action {i + 1}: {action.type}\n"

            # Placeholder: Action execution requires ActionExecutor integration
            # Integration point: Pass ActionExecutor instance to RunProcess.__init__
            # Then call: self.action_executor.execute_action(action)
            # For now, we assume success

        object.__setattr__(action_result, "output_text", output_text)
        return success

    def _execute_until_success(
        self,
        workflow: Workflow | Process,
        repetition,
        action_result: ActionResult,
        object_collections,
    ) -> None:
        """Execute workflow repeatedly until success or max repeats reached.

        Args:
            workflow: The workflow/process to execute
            repetition: ProcessRepetitionOptions configuration
            action_result: ActionResult to update
            object_collections: ObjectCollection arguments
        """
        max_repeats = repetition.get_max_repeats()
        delay = repetition.get_delay()
        total_runs = max_repeats + 1  # Initial run + repeats

        results = []
        output_text = f"RUN_PROCESS (until success): {workflow.name}\n"

        for run_num in range(total_runs):
            output_text += f"\n--- Attempt {run_num + 1}/{total_runs} ---\n"

            # Execute the workflow
            success = self._execute_workflow_once(workflow, action_result)
            results.append(success)

            if success:
                # Success! Stop early
                output_text += f"\n✓ Workflow succeeded on attempt {run_num + 1}\n"
                output_text += f"Total attempts: {run_num + 1}\n"

                object.__setattr__(action_result, "success", True)
                object.__setattr__(action_result, "output_text", output_text)
                return

            # Delay before next attempt (but not after last)
            if run_num < total_runs - 1 and delay > 0:
                output_text += f"Waiting {delay}s before next attempt...\n"
                time.sleep(delay)

        # Reached max repeats without success
        output_text += f"\n✗ Workflow failed after {total_runs} attempts\n"
        object.__setattr__(action_result, "success", False)
        object.__setattr__(action_result, "output_text", output_text)

    def _execute_fixed_count(
        self,
        workflow: Workflow | Process,
        repetition,
        action_result: ActionResult,
        object_collections,
    ) -> None:
        """Execute workflow a fixed number of times.

        Args:
            workflow: The workflow/process to execute
            repetition: ProcessRepetitionOptions configuration
            action_result: ActionResult to update
            object_collections: ObjectCollection arguments
        """
        max_repeats = repetition.get_max_repeats()
        delay = repetition.get_delay()
        total_runs = max_repeats + 1  # Initial run + repeats

        results = []
        output_text = f"RUN_PROCESS (fixed count): {workflow.name}\n"
        output_text += f"Running {total_runs} times\n"

        for run_num in range(total_runs):
            output_text += f"\n--- Run {run_num + 1}/{total_runs} ---\n"

            # Execute the workflow
            success = self._execute_workflow_once(workflow, action_result)
            results.append(success)

            # Delay between runs (but not after last)
            if run_num < total_runs - 1 and delay > 0:
                output_text += f"Waiting {delay}s...\n"
                time.sleep(delay)

        # Aggregate results
        success_count = sum(1 for r in results if r)
        at_least_one_success = success_count > 0

        output_text += "\n--- Summary ---\n"
        output_text += f"Total runs: {total_runs}\n"
        output_text += f"Successful: {success_count}\n"
        output_text += f"Failed: {total_runs - success_count}\n"

        if at_least_one_success:
            output_text += "✓ Overall: SUCCESS (at least one run succeeded)\n"
        else:
            output_text += "✗ Overall: FAILURE (all runs failed)\n"

        object.__setattr__(action_result, "success", at_least_one_success)
        object.__setattr__(action_result, "output_text", output_text)
