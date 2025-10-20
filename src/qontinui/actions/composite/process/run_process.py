"""RunProcess action implementation.

Executes a named process with optional repetition control.
"""

import time
from typing import TYPE_CHECKING

from ....actions.action_interface import ActionInterface
from ....actions.action_result import ActionResult
from ....actions.action_type import ActionType
from .run_process_options import RunProcessOptions

if TYPE_CHECKING:
    from ....json_executor.config_parser import Process, QontinuiConfig
    from ....model.object_collection import ObjectCollection


class RunProcess(ActionInterface):
    """Execute a named process with repetition support.

    This action executes a process (sequence of actions) by its ID,
    with support for:
    - Fixed count repetition: Run exactly N additional times
    - Until success repetition: Stop early on success, otherwise run up to N times
    - Configurable delays between repetitions
    """

    def __init__(self, config: "QontinuiConfig | None" = None):
        """Initialize RunProcess action.

        Args:
            config: QontinuiConfig containing process definitions
        """
        self.config = config

    def set_config(self, config: "QontinuiConfig") -> None:
        """Set the configuration after initialization.

        Args:
            config: QontinuiConfig containing process definitions
        """
        self.config = config

    def get_action_type(self) -> ActionType:
        """Get the action type.

        Returns:
            ActionType.RUN_PROCESS
        """
        return ActionType.RUN_PROCESS

    def perform(self, action_result: ActionResult, *object_collections: "ObjectCollection") -> None:
        """Execute the process with optional repetition.

        Args:
            action_result: ActionResult containing configuration
            object_collections: Variable number of ObjectCollection arguments
        """
        # Extract and validate configuration
        run_process_options = action_result.action_config
        if not isinstance(run_process_options, RunProcessOptions):
            raise ValueError("RunProcess requires RunProcessOptions configuration")

        if not self.config:
            raise RuntimeError("RunProcess requires QontinuiConfig to be set")

        process_id = run_process_options.get_process_id()
        if not process_id:
            action_result.success = False
            action_result.output_text = "RUN_PROCESS: No process ID specified"
            return

        # Get the process
        process = self._get_process(process_id)
        if not process:
            action_result.success = False
            action_result.output_text = f"RUN_PROCESS: Process '{process_id}' not found"
            return

        # Check if repetition is enabled
        repetition = run_process_options.get_process_repetition()
        if not repetition.get_enabled():
            # No repetition - execute once
            success = self._execute_process_once(process, action_result)
            action_result.success = success
            return

        # Execute with repetition
        if repetition.get_until_success():
            self._execute_until_success(process, repetition, action_result, object_collections)
        else:
            self._execute_fixed_count(process, repetition, action_result, object_collections)

    def _get_process(self, process_id: str) -> "Process | None":
        """Get a process by ID from the configuration.

        Args:
            process_id: The process ID to look up

        Returns:
            Process instance or None if not found
        """
        if not self.config or not hasattr(self.config, "process_map"):
            return None
        return self.config.process_map.get(process_id)

    def _execute_process_once(self, process: "Process", action_result: ActionResult) -> bool:
        """Execute a process once.

        Args:
            process: The process to execute
            action_result: ActionResult to update

        Returns:
            True if process succeeded, False otherwise
        """
        # Import here to avoid circular dependency

        # Create a temporary executor for this process
        # Note: In a real scenario, this would use the existing executor
        # For now, we'll use a simplified approach
        output_text = f"Executing process: {process.name}\n"

        # Execute process actions sequentially
        success = True
        for i, action in enumerate(process.actions):
            output_text += f"  Action {i + 1}: {action.type}\n"

            # TODO: Actually execute the action using the action executor
            # For now, we assume success
            # In full implementation, this would call:
            # action_executor.execute_action(action)

        action_result.output_text = output_text
        return success

    def _execute_until_success(
        self, process: "Process", repetition, action_result: ActionResult, object_collections
    ) -> None:
        """Execute process repeatedly until success or max repeats reached.

        Args:
            process: The process to execute
            repetition: ProcessRepetitionOptions configuration
            action_result: ActionResult to update
            object_collections: ObjectCollection arguments
        """
        max_repeats = repetition.get_max_repeats()
        delay = repetition.get_delay()
        total_runs = max_repeats + 1  # Initial run + repeats

        results = []
        output_text = f"RUN_PROCESS (until success): {process.name}\n"

        for run_num in range(total_runs):
            output_text += f"\n--- Attempt {run_num + 1}/{total_runs} ---\n"

            # Execute the process
            success = self._execute_process_once(process, action_result)
            results.append(success)

            if success:
                # Success! Stop early
                output_text += f"\n✓ Process succeeded on attempt {run_num + 1}\n"
                output_text += f"Total attempts: {run_num + 1}\n"

                action_result.success = True
                action_result.output_text = output_text
                return

            # Delay before next attempt (but not after last)
            if run_num < total_runs - 1 and delay > 0:
                output_text += f"Waiting {delay}s before next attempt...\n"
                time.sleep(delay)

        # Reached max repeats without success
        output_text += f"\n✗ Process failed after {total_runs} attempts\n"
        action_result.success = False
        action_result.output_text = output_text

    def _execute_fixed_count(
        self, process: "Process", repetition, action_result: ActionResult, object_collections
    ) -> None:
        """Execute process a fixed number of times.

        Args:
            process: The process to execute
            repetition: ProcessRepetitionOptions configuration
            action_result: ActionResult to update
            object_collections: ObjectCollection arguments
        """
        max_repeats = repetition.get_max_repeats()
        delay = repetition.get_delay()
        total_runs = max_repeats + 1  # Initial run + repeats

        results = []
        output_text = f"RUN_PROCESS (fixed count): {process.name}\n"
        output_text += f"Running {total_runs} times\n"

        for run_num in range(total_runs):
            output_text += f"\n--- Run {run_num + 1}/{total_runs} ---\n"

            # Execute the process
            success = self._execute_process_once(process, action_result)
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

        action_result.success = at_least_one_success
        action_result.output_text = output_text
