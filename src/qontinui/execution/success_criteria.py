"""Success criteria evaluation for workflow execution.

This module provides comprehensive success criteria evaluation beyond simple
all-actions-pass logic. Supports match counting (critical for state discovery),
checkpoint validation, failure tolerance, and custom conditions.
"""

import ast
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .graph_executor import ExecutionState

logger = logging.getLogger(__name__)


class SuccessCriteriaType(Enum):
    """Types of success criteria for workflow evaluation."""

    ALL_ACTIONS_PASS = "all_actions_pass"
    """All actions must succeed (default behavior)."""

    MIN_MATCHES = "min_matches"
    """At least N matches must be found across all actions."""

    MAX_FAILURES = "max_failures"
    """Allow up to N action failures."""

    CHECKPOINT_PASSED = "checkpoint_passed"
    """Specific named checkpoint action must pass."""

    REQUIRED_STATES = "required_states"
    """Must reach all specified states."""

    CUSTOM = "custom"
    """Custom Python expression evaluated against execution state."""


@dataclass(frozen=True)
class SuccessCriteria:
    """Workflow success criteria definition.

    Immutable configuration that defines how to evaluate workflow success.
    Different criteria types use different parameters.

    Examples:
        # Require at least 5 matches for state discovery
        SuccessCriteria(
            criteria_type=SuccessCriteriaType.MIN_MATCHES,
            min_matches=5,
            description="Find at least 5 UI elements"
        )

        # Allow up to 2 failures
        SuccessCriteria(
            criteria_type=SuccessCriteriaType.MAX_FAILURES,
            max_failures=2,
            description="Tolerate up to 2 failed actions"
        )

        # Require specific checkpoint to pass
        SuccessCriteria(
            criteria_type=SuccessCriteriaType.CHECKPOINT_PASSED,
            checkpoint_name="login_success",
            description="Login must succeed"
        )

        # Require reaching certain states
        SuccessCriteria(
            criteria_type=SuccessCriteriaType.REQUIRED_STATES,
            required_states=("logged_in", "dashboard"),
            description="Must reach logged_in and dashboard states"
        )

        # Custom expression
        SuccessCriteria(
            criteria_type=SuccessCriteriaType.CUSTOM,
            custom_condition="match_count >= 3 and failed_count == 0",
            description="At least 3 matches with no failures"
        )
    """

    criteria_type: SuccessCriteriaType
    description: str | None = None

    # Type-specific parameters
    min_matches: int | None = None
    """Minimum number of matches required (for MIN_MATCHES)."""

    max_failures: int | None = None
    """Maximum number of action failures allowed (for MAX_FAILURES)."""

    checkpoint_name: str | None = None
    """Name of checkpoint action that must pass (for CHECKPOINT_PASSED)."""

    required_states: tuple[str, ...] | None = None
    """States that must be reached (for REQUIRED_STATES)."""

    custom_condition: str | None = None
    """Python expression to evaluate (for CUSTOM).

    Available variables:
        - total_actions: Total number of actions
        - successful_actions: Number of successful actions
        - failed_actions: Number of failed actions
        - skipped_actions: Number of skipped actions
        - match_count: Total matches across all actions
        - states_reached: Set of state names reached
        - checkpoints_passed: Set of checkpoint names that passed
        - checkpoints_failed: Set of checkpoint names that failed
    """


@dataclass(frozen=True)
class WorkflowResult:
    """Complete workflow execution result with success evaluation.

    Immutable record of workflow execution including success evaluation
    based on configured criteria. Provides detailed statistics and
    clear explanation of pass/fail decision.
    """

    workflow_name: str
    """Name of the workflow."""

    success: bool
    """Overall success based on success criteria evaluation."""

    success_criteria: SuccessCriteria | None
    """Success criteria used for evaluation (None = default ALL_ACTIONS_PASS)."""

    criteria_evaluation: str
    """Human-readable explanation of success/failure decision."""

    # Action statistics
    total_actions: int
    """Total number of actions in workflow."""

    successful_actions: int
    """Number of actions that completed successfully."""

    failed_actions: int
    """Number of actions that failed."""

    skipped_actions: int
    """Number of actions that were skipped."""

    # Match statistics (for state discovery)
    total_matches: int
    """Total number of matches found across all actions."""

    # Checkpoint tracking
    checkpoints_passed: tuple[str, ...]
    """Names of checkpoint actions that passed."""

    checkpoints_failed: tuple[str, ...]
    """Names of checkpoint actions that failed."""

    # State tracking
    states_reached: tuple[str, ...]
    """Names of states reached during execution."""

    # Timing
    duration_ms: int
    """Total execution duration in milliseconds."""

    # Error information
    error: str | None = None
    """Error message if workflow failed catastrophically."""

    def __str__(self) -> str:
        """Human-readable result summary."""
        status = "SUCCESS" if self.success else "FAILURE"
        return (
            f"WorkflowResult({self.workflow_name}): {status}\n"
            f"  Actions: {self.successful_actions}/{self.total_actions} successful, "
            f"{self.failed_actions} failed, {self.skipped_actions} skipped\n"
            f"  Matches: {self.total_matches}\n"
            f"  Duration: {self.duration_ms}ms\n"
            f"  Evaluation: {self.criteria_evaluation}"
        )


class SuccessCriteriaEvaluator:
    """Evaluates workflow execution against success criteria.

    Provides evaluation logic for all supported criteria types.
    Handles safe evaluation of custom expressions.
    """

    def evaluate(
        self,
        criteria: SuccessCriteria,
        execution_state: ExecutionState,
    ) -> tuple[bool, str]:
        """Evaluate if workflow succeeded based on criteria.

        Args:
            criteria: Success criteria to evaluate against
            execution_state: Current execution state with results

        Returns:
            Tuple of (success: bool, explanation: str)

        Raises:
            ValueError: If criteria configuration is invalid
        """
        # Dispatch to appropriate evaluator
        evaluators = {
            SuccessCriteriaType.ALL_ACTIONS_PASS: self._evaluate_all_actions_pass,
            SuccessCriteriaType.MIN_MATCHES: self._evaluate_min_matches,
            SuccessCriteriaType.MAX_FAILURES: self._evaluate_max_failures,
            SuccessCriteriaType.CHECKPOINT_PASSED: self._evaluate_checkpoint,
            SuccessCriteriaType.REQUIRED_STATES: self._evaluate_required_states,
            SuccessCriteriaType.CUSTOM: self._evaluate_custom,
        }

        evaluator = evaluators.get(criteria.criteria_type)
        if not evaluator:
            raise ValueError(f"Unknown criteria type: {criteria.criteria_type}")

        try:
            return evaluator(criteria, execution_state)
        except ValueError:
            # Re-raise ValueError for parameter validation errors
            raise
        except Exception as e:
            logger.error(f"Error evaluating success criteria: {e}", exc_info=True)
            return False, f"Evaluation error: {e}"

    def _evaluate_all_actions_pass(
        self, criteria: SuccessCriteria, state: ExecutionState
    ) -> tuple[bool, str]:
        """Evaluate ALL_ACTIONS_PASS criteria."""
        stats = self._compute_statistics(state)

        if stats["failed_actions"] == 0:
            return True, (
                f"All {stats['successful_actions']} actions passed successfully "
                f"({stats['skipped_actions']} skipped)"
            )
        else:
            return False, (f"{stats['failed_actions']} of {stats['total_actions']} actions failed")

    def _evaluate_min_matches(
        self, criteria: SuccessCriteria, state: ExecutionState
    ) -> tuple[bool, str]:
        """Evaluate MIN_MATCHES criteria."""
        if criteria.min_matches is None or criteria.min_matches == 0:
            raise ValueError("MIN_MATCHES criteria requires min_matches parameter")

        stats = self._compute_statistics(state)
        match_count = stats["match_count"]

        if match_count >= criteria.min_matches:
            return True, (f"Found {match_count} matches (required: {criteria.min_matches})")
        else:
            return False, (f"Only found {match_count} matches, needed {criteria.min_matches}")

    def _evaluate_max_failures(
        self, criteria: SuccessCriteria, state: ExecutionState
    ) -> tuple[bool, str]:
        """Evaluate MAX_FAILURES criteria."""
        if criteria.max_failures is None:
            raise ValueError("MAX_FAILURES criteria requires max_failures parameter")

        stats = self._compute_statistics(state)
        failed_count = stats["failed_actions"]

        if failed_count <= criteria.max_failures:
            return True, (
                f"{failed_count} failures (allowed: {criteria.max_failures}), "
                f"{stats['successful_actions']} succeeded"
            )
        else:
            return False, (f"{failed_count} failures exceeds limit of {criteria.max_failures}")

    def _evaluate_checkpoint(
        self, criteria: SuccessCriteria, state: ExecutionState
    ) -> tuple[bool, str]:
        """Evaluate CHECKPOINT_PASSED criteria."""
        if not criteria.checkpoint_name:
            raise ValueError("CHECKPOINT_PASSED criteria requires checkpoint_name parameter")

        stats = self._compute_statistics(state)
        checkpoints_passed = stats["checkpoints_passed"]
        checkpoints_failed = stats["checkpoints_failed"]

        if criteria.checkpoint_name in checkpoints_passed:
            return True, f"Checkpoint '{criteria.checkpoint_name}' passed"
        elif criteria.checkpoint_name in checkpoints_failed:
            return False, f"Checkpoint '{criteria.checkpoint_name}' failed"
        else:
            return False, (
                f"Checkpoint '{criteria.checkpoint_name}' was not executed "
                f"(available: {list(checkpoints_passed.union(checkpoints_failed))})"
            )

    def _evaluate_required_states(
        self, criteria: SuccessCriteria, state: ExecutionState
    ) -> tuple[bool, str]:
        """Evaluate REQUIRED_STATES criteria."""
        if not criteria.required_states:
            raise ValueError("REQUIRED_STATES criteria requires required_states parameter")

        stats = self._compute_statistics(state)
        states_reached = stats["states_reached"]
        required = set(criteria.required_states)

        missing_states = required - states_reached
        if not missing_states:
            return True, f"All required states reached: {sorted(required)}"
        else:
            return False, (
                f"Missing required states: {sorted(missing_states)} "
                f"(reached: {sorted(states_reached)})"
            )

    def _evaluate_custom(
        self, criteria: SuccessCriteria, state: ExecutionState
    ) -> tuple[bool, str]:
        """Evaluate CUSTOM criteria with safe expression evaluation."""
        if not criteria.custom_condition:
            raise ValueError("CUSTOM criteria requires custom_condition parameter")

        stats = self._compute_statistics(state)

        # Create evaluation context with available variables
        context = {
            "total_actions": stats["total_actions"],
            "successful_actions": stats["successful_actions"],
            "failed_actions": stats["failed_actions"],
            "skipped_actions": stats["skipped_actions"],
            "match_count": stats["match_count"],
            "states_reached": stats["states_reached"],
            "checkpoints_passed": stats["checkpoints_passed"],
            "checkpoints_failed": stats["checkpoints_failed"],
        }

        try:
            # Parse and validate expression (security check)
            parsed = ast.parse(criteria.custom_condition, mode="eval")
            self._validate_safe_expression(parsed)

            # Evaluate expression
            result = eval(
                compile(parsed, "<custom_condition>", "eval"),
                {"__builtins__": {}},  # No builtins for security
                context,
            )

            if not isinstance(result, bool):
                return False, (f"Custom condition returned non-boolean: {type(result).__name__}")

            explanation = (
                f"Custom condition '{criteria.custom_condition}' "
                f"evaluated to {result} with: {context}"
            )
            return result, explanation

        except SyntaxError as e:
            return False, f"Invalid custom condition syntax: {e}"
        except Exception as e:
            logger.error(f"Error evaluating custom condition: {e}", exc_info=True)
            return False, f"Custom condition evaluation failed: {e}"

    def _validate_safe_expression(self, parsed: ast.AST) -> None:
        """Validate that expression is safe to evaluate.

        Raises:
            ValueError: If expression contains unsafe operations
        """
        # Only allow safe node types
        safe_nodes = {
            ast.Expression,
            ast.BoolOp,
            ast.Compare,
            ast.BinOp,
            ast.UnaryOp,
            ast.Constant,
            ast.Name,
            ast.Load,
            ast.And,
            ast.Or,
            ast.Not,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.In,
            ast.NotIn,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Mod,
        }

        for node in ast.walk(parsed):
            if type(node) not in safe_nodes:
                raise ValueError(f"Unsafe operation in custom condition: {type(node).__name__}")

    def _compute_statistics(self, state: ExecutionState) -> dict[str, Any]:
        """Compute comprehensive statistics from execution state.

        Args:
            state: Execution state to analyze

        Returns:
            Dictionary with all statistics needed for evaluation
        """
        from .graph_traverser import TraversalState

        # Count action statuses
        total_actions = len(state.workflow.actions)
        successful_actions = sum(
            1 for s in state.action_states.values() if s == TraversalState.COMPLETED
        )
        failed_actions = sum(1 for s in state.action_states.values() if s == TraversalState.FAILED)
        skipped_actions = sum(
            1 for s in state.action_states.values() if s == TraversalState.SKIPPED
        )

        # Count total matches across all action results
        match_count = 0
        states_reached: set[str] = set()
        checkpoints_passed: set[str] = set()
        checkpoints_failed: set[str] = set()

        # Process action results for match counts and states
        for _action_id, result in state.action_results.items():
            # Count matches from result
            if isinstance(result, dict):
                # Handle dict results (from GraphExecutor)
                # Prioritize match_count if present, otherwise count matches list
                if "match_count" in result:
                    # Use explicit match_count
                    match_count += result["match_count"]
                elif "matches" in result:
                    # Fall back to counting matches list
                    matches = result["matches"]
                    if isinstance(matches, list | tuple):
                        match_count += len(matches)
                    elif isinstance(matches, int):
                        match_count += matches

                # Track states reached
                if "active_states" in result:
                    active_states = result["active_states"]
                    if isinstance(active_states, set | frozenset | list | tuple):
                        states_reached.update(active_states)

                if "states_reached" in result:
                    reached = result["states_reached"]
                    if isinstance(reached, set | frozenset | list | tuple):
                        states_reached.update(reached)

        # Track checkpoint actions (check ALL actions, not just those with results)
        for action in state.workflow.actions:
            if action.name and "checkpoint" in action.name.lower():
                action_state = state.action_states.get(action.id)
                if action_state == TraversalState.COMPLETED:
                    checkpoints_passed.add(action.name)
                elif action_state == TraversalState.FAILED:
                    checkpoints_failed.add(action.name)

        return {
            "total_actions": total_actions,
            "successful_actions": successful_actions,
            "failed_actions": failed_actions,
            "skipped_actions": skipped_actions,
            "match_count": match_count,
            "states_reached": states_reached,
            "checkpoints_passed": checkpoints_passed,
            "checkpoints_failed": checkpoints_failed,
        }

    def create_workflow_result(
        self,
        criteria: SuccessCriteria | None,
        execution_state: ExecutionState,
    ) -> WorkflowResult:
        """Create comprehensive workflow result with evaluation.

        Args:
            criteria: Success criteria (None = default ALL_ACTIONS_PASS)
            execution_state: Execution state to evaluate

        Returns:
            Complete workflow result with success evaluation
        """
        # Use default criteria if none provided
        if criteria is None:
            criteria = SuccessCriteria(
                criteria_type=SuccessCriteriaType.ALL_ACTIONS_PASS,
                description="All actions must pass",
            )

        # Evaluate success
        success, explanation = self.evaluate(criteria, execution_state)

        # Compute statistics
        stats = self._compute_statistics(execution_state)

        # Calculate duration
        duration_ms = int(execution_state.get_elapsed_time() * 1000)

        # Extract error if present
        error = None
        if execution_state.errors:
            error = "; ".join(str(e.get("error", "")) for e in execution_state.errors)

        return WorkflowResult(
            workflow_name=execution_state.workflow.name,
            success=success,
            success_criteria=criteria,
            criteria_evaluation=explanation,
            total_actions=stats["total_actions"],
            successful_actions=stats["successful_actions"],
            failed_actions=stats["failed_actions"],
            skipped_actions=stats["skipped_actions"],
            total_matches=stats["match_count"],
            checkpoints_passed=tuple(sorted(stats["checkpoints_passed"])),
            checkpoints_failed=tuple(sorted(stats["checkpoints_failed"])),
            states_reached=tuple(sorted(stats["states_reached"])),
            duration_ms=duration_ms,
            error=error,
        )


# Convenience function for common use cases
def evaluate_workflow_success(
    execution_state: ExecutionState,
    criteria: SuccessCriteria | None = None,
) -> WorkflowResult:
    """Convenience function to evaluate workflow success.

    Args:
        execution_state: Execution state to evaluate
        criteria: Success criteria (None = default ALL_ACTIONS_PASS)

    Returns:
        Complete workflow result with evaluation

    Example:
        # Default: all actions must pass
        result = evaluate_workflow_success(execution_state)

        # Custom: require at least 5 matches
        criteria = SuccessCriteria(
            criteria_type=SuccessCriteriaType.MIN_MATCHES,
            min_matches=5
        )
        result = evaluate_workflow_success(execution_state, criteria)
    """
    evaluator = SuccessCriteriaEvaluator()
    return evaluator.create_workflow_result(criteria, execution_state)
