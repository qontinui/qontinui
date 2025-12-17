"""TransitionFunction class - ported from Qontinui framework.

Defines executable functions for state transitions.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TransitionType(Enum):
    """Type of transition function."""

    ACTION = auto()  # Performs an action to transition
    WAIT = auto()  # Waits for condition to transition
    CONDITIONAL = auto()  # Checks condition for transition
    COMPOSITE = auto()  # Combines multiple functions
    FALLBACK = auto()  # Fallback if primary fails


@dataclass
class TransitionResult:
    """Result of executing a transition function."""

    success: bool
    next_state: str | None = None
    duration: float = 0.0
    error: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransitionFunction:
    """Executable function for state transitions.

    Port of TransitionFunction from Qontinui framework class.

    TransitionFunction encapsulates the logic needed to transition between states.
    This can be a simple action like clicking a button, a complex sequence of
    actions, or conditional logic that determines the next state dynamically.

    Key features:
    - Executable transition logic
    - Pre/post conditions
    - Retry mechanisms
    - Timeout handling
    - Success/failure callbacks
    - Composition of multiple functions

    Common patterns:
    - Click button to navigate to next screen
    - Wait for element to appear before proceeding
    - Conditional navigation based on UI state
    - Multi-step transitions with validation
    - Fallback transitions on failure

    Example:
        # Simple click transition
        transition = TransitionFunction(
            name="login_to_dashboard",
            function=lambda: click(login_button),
            pre_condition=lambda: is_visible(login_form),
            post_condition=lambda: is_visible(dashboard),
            timeout=10.0
        )

        # Execute transition
        result = transition.execute()
        if result.success:
            print(f"Transitioned in {result.duration}s")
    """

    # Unique name for the transition
    name: str = ""

    # Type of transition
    transition_type: TransitionType = TransitionType.ACTION

    # Main transition function
    function: Callable[[], Any] | None = None

    # Pre-condition to check before executing
    pre_condition: Callable[[], bool] | None = None

    # Post-condition to verify after executing
    post_condition: Callable[[], bool] | None = None

    # Timeout for the transition
    timeout: float = 30.0

    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0

    # Success/failure callbacks
    on_success: Callable[[TransitionResult], None] | None = None
    on_failure: Callable[[TransitionResult], None] | None = None

    # Child functions for composite transitions
    child_functions: list["TransitionFunction"] = field(default_factory=list)

    # Fallback function if this one fails
    fallback_function: Optional["TransitionFunction"] = None

    # Additional parameters
    parameters: dict[str, Any] = field(default_factory=dict)

    # Execution state
    _last_result: TransitionResult | None = field(default=None, init=False)
    _execution_count: int = field(default=0, init=False)

    def execute(self, **kwargs) -> TransitionResult:
        """Execute the transition function.

        Args:
            **kwargs: Additional parameters for execution

        Returns:
            TransitionResult with execution details
        """
        start_time = time.time()
        self._execution_count += 1

        # Merge kwargs with stored parameters
        params = {**self.parameters, **kwargs}

        try:
            # Check pre-condition
            if self.pre_condition and not self._check_condition(
                self.pre_condition, self.timeout
            ):
                result = TransitionResult(
                    success=False,
                    error="Pre-condition not met",
                    duration=time.time() - start_time,
                )
                self._handle_failure(result)
                return result

            # Execute based on type
            if self.transition_type == TransitionType.ACTION:
                result = self._execute_action(params)
            elif self.transition_type == TransitionType.WAIT:
                result = self._execute_wait(params)
            elif self.transition_type == TransitionType.CONDITIONAL:
                result = self._execute_conditional(params)
            elif self.transition_type == TransitionType.COMPOSITE:
                result = self._execute_composite(params)
            else:
                result = self._execute_with_retries(params)

            # Check post-condition
            if result.success and self.post_condition:
                if not self._check_condition(self.post_condition, self.timeout):
                    result.success = False
                    result.error = "Post-condition not met"

            # Handle result
            result.duration = time.time() - start_time
            if result.success:
                self._handle_success(result)
            else:
                self._handle_failure(result)

            self._last_result = result
            return result

        except Exception as e:
            logger.error(f"Transition function '{self.name}' failed: {e}")
            result = TransitionResult(
                success=False, error=str(e), duration=time.time() - start_time
            )
            self._handle_failure(result)
            self._last_result = result
            return result

    def _execute_action(self, params: dict[str, Any]) -> TransitionResult:
        """Execute action-based transition.

        Args:
            params: Execution parameters

        Returns:
            Transition result
        """
        if not self.function:
            return TransitionResult(success=False, error="No function defined")

        for attempt in range(self.max_retries):
            try:
                # Execute the function
                if params:
                    result = self.function(**params)
                else:
                    result = self.function()

                # Interpret result
                if isinstance(result, bool):
                    if result:
                        return TransitionResult(success=True)
                elif isinstance(result, TransitionResult):
                    return result
                elif result is not None:
                    return TransitionResult(success=True, data={"result": result})
                else:
                    return TransitionResult(success=True)

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return TransitionResult(success=False, error=str(e))

        return TransitionResult(success=False, error="Max retries exceeded")

    def _execute_wait(self, params: dict[str, Any]) -> TransitionResult:
        """Execute wait-based transition.

        Args:
            params: Execution parameters

        Returns:
            Transition result
        """
        if not self.function:
            return TransitionResult(success=False, error="No wait function defined")

        end_time = time.time() + self.timeout
        while time.time() < end_time:
            try:
                if params:
                    condition_met = self.function(**params)
                else:
                    condition_met = self.function()

                if condition_met:
                    return TransitionResult(success=True)

                time.sleep(0.5)  # Check every 500ms

            except Exception as e:
                logger.warning(f"Wait condition check failed: {e}")
                time.sleep(0.5)

        return TransitionResult(success=False, error="Wait timeout exceeded")

    def _execute_conditional(self, params: dict[str, Any]) -> TransitionResult:
        """Execute conditional transition.

        Args:
            params: Execution parameters

        Returns:
            Transition result with next state
        """
        if not self.function:
            return TransitionResult(
                success=False, error="No conditional function defined"
            )

        try:
            if params:
                next_state = self.function(**params)
            else:
                next_state = self.function()

            if next_state:
                return TransitionResult(success=True, next_state=str(next_state))
            else:
                return TransitionResult(success=False, error="No next state determined")

        except Exception as e:
            return TransitionResult(success=False, error=str(e))

    def _execute_composite(self, params: dict[str, Any]) -> TransitionResult:
        """Execute composite transition with multiple child functions.

        Args:
            params: Execution parameters

        Returns:
            Composite result
        """
        if not self.child_functions:
            return TransitionResult(success=False, error="No child functions defined")

        total_duration = 0.0
        combined_data = {}

        for child in self.child_functions:
            result = child.execute(**params)
            total_duration += result.duration
            combined_data.update(result.data)

            if not result.success:
                return TransitionResult(
                    success=False,
                    error=f"Child function '{child.name}' failed: {result.error}",
                    duration=total_duration,
                    data=combined_data,
                )

        return TransitionResult(
            success=True, duration=total_duration, data=combined_data
        )

    def _execute_with_retries(self, params: dict[str, Any]) -> TransitionResult:
        """Execute with retry logic.

        Args:
            params: Execution parameters

        Returns:
            Transition result
        """
        return self._execute_action(params)

    def _check_condition(self, condition: Callable[[], bool], timeout: float) -> bool:
        """Check a condition with timeout.

        Args:
            condition: Condition function to check
            timeout: Timeout in seconds

        Returns:
            True if condition met within timeout
        """
        end_time = time.time() + timeout
        while time.time() < end_time:
            try:
                if condition():
                    return True
            except Exception as e:
                logger.warning(f"Condition check failed: {e}")
            time.sleep(0.1)
        return False

    def _handle_success(self, result: TransitionResult):
        """Handle successful transition.

        Args:
            result: Transition result
        """
        if self.on_success:
            try:
                self.on_success(result)
            except Exception as e:
                logger.error(f"Success handler failed: {e}")

    def _handle_failure(self, result: TransitionResult):
        """Handle failed transition.

        Args:
            result: Transition result
        """
        if self.on_failure:
            try:
                self.on_failure(result)
            except Exception as e:
                logger.error(f"Failure handler failed: {e}")

        # Try fallback if available
        if self.fallback_function and not result.data.get("fallback_attempted"):
            logger.info(f"Attempting fallback for '{self.name}'")
            result.data["fallback_attempted"] = True
            fallback_result = self.fallback_function.execute(**self.parameters)
            if fallback_result.success:
                result.success = True
                result.error = None
                result.data.update(fallback_result.data)

    def add_child(self, child: "TransitionFunction") -> "TransitionFunction":
        """Add a child function for composite transitions.

        Args:
            child: Child transition function

        Returns:
            Self for fluent interface
        """
        self.child_functions.append(child)
        self.transition_type = TransitionType.COMPOSITE
        return self

    def set_fallback(self, fallback: "TransitionFunction") -> "TransitionFunction":
        """Set fallback function.

        Args:
            fallback: Fallback transition function

        Returns:
            Self for fluent interface
        """
        self.fallback_function = fallback
        return self

    def set_parameter(self, key: str, value: Any) -> "TransitionFunction":
        """Set an execution parameter.

        Args:
            key: Parameter key
            value: Parameter value

        Returns:
            Self for fluent interface
        """
        self.parameters[key] = value
        return self

    def get_last_result(self) -> TransitionResult | None:
        """Get the last execution result.

        Returns:
            Last TransitionResult or None
        """
        return self._last_result

    def get_execution_count(self) -> int:
        """Get number of times executed.

        Returns:
            Execution count
        """
        return self._execution_count

    def reset(self) -> "TransitionFunction":
        """Reset execution state.

        Returns:
            Self for fluent interface
        """
        self._last_result = None
        self._execution_count = 0
        return self

    def is_composite(self) -> bool:
        """Check if this is a composite transition.

        Returns:
            True if composite
        """
        return self.transition_type == TransitionType.COMPOSITE or bool(
            self.child_functions
        )

    def validate(self) -> bool:
        """Validate the transition function configuration.

        Returns:
            True if valid
        """
        if not self.name:
            logger.warning("TransitionFunction has no name")
            return False

        if self.transition_type == TransitionType.COMPOSITE:
            if not self.child_functions:
                logger.warning(f"Composite transition '{self.name}' has no children")
                return False
        elif not self.function:
            logger.warning(f"Transition '{self.name}' has no function")
            return False

        return True

    def __str__(self) -> str:
        """String representation."""
        children = (
            f", {len(self.child_functions)} children" if self.child_functions else ""
        )
        fallback = ", with fallback" if self.fallback_function else ""
        return f"TransitionFunction({self.name}, {self.transition_type.name}{children}{fallback})"

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"TransitionFunction(name='{self.name}', "
            f"type={self.transition_type}, "
            f"children={len(self.child_functions)}, "
            f"has_fallback={self.fallback_function is not None})"
        )
