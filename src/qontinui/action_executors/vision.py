"""Vision/image recognition action executor.

This module provides the VisionActionExecutor for handling image recognition
actions (FIND, VANISH) using template matching with mask support.
"""

import logging
from typing import Any

from ..config import FindActionConfig, VanishActionConfig
from ..config.schema import Action
from ..exceptions import ActionExecutionError
from .base import ActionExecutorBase
from .registry import register_executor
from .target_resolver import TargetResolver

logger = logging.getLogger(__name__)


@register_executor
class VisionActionExecutor(ActionExecutorBase):
    """Executor for vision/image recognition actions.

    Handles:
        - FIND: Locate image on screen, store complete ActionResult with all matches
        - VANISH: Wait for image to disappear

    Uses OpenCV template matching with configurable similarity thresholds.
    Results are stored in context.last_action_result for subsequent targeting.
    """

    def __init__(self, context: Any):
        """Initialize executor with context.

        Args:
            context: Execution context with time, actions, and result storage
        """
        super().__init__(context)

        # Don't use PatternLoader - use registry like IF actions do
        # Initialize target resolver without pattern loader
        self.target_resolver = TargetResolver(context, template_matcher=None)

    def get_supported_action_types(self) -> list[str]:
        """Get list of vision action types this executor handles.

        Returns:
            List containing ["FIND", "VANISH"]
        """
        return ["FIND", "VANISH"]

    def execute(self, action: Action, typed_config: Any) -> bool:
        """Execute vision action with validated configuration.

        Args:
            action: Pydantic Action model
            typed_config: Type-specific config (FindActionConfig, VanishActionConfig)

        Returns:
            bool: True if action succeeded, False otherwise

        Raises:
            ActionExecutionError: If action execution fails critically
        """
        logger.debug(f"Executing vision action: {action.type}")

        if action.type == "FIND":
            return self._execute_find(action, typed_config)
        elif action.type == "VANISH":
            return self._execute_vanish(action, typed_config)
        else:
            raise ActionExecutionError(
                action_type=action.type,
                reason=f"Unknown vision action type: {action.type}",
            )

    def _execute_find(self, action: Action, typed_config: FindActionConfig) -> bool:
        """Execute FIND action - locate image and store ActionResult.

        Args:
            action: Pydantic Action model
            typed_config: Pre-validated FindActionConfig

        Returns:
            bool: True if image was found
        """
        # Set up comprehensive debug logging
        import os
        import tempfile
        from datetime import datetime

        debug_log = os.path.join(tempfile.gettempdir(), "qontinui_find_debug.log")

        def log_debug(msg: str):
            """Helper to write timestamped debug messages."""
            try:
                with open(debug_log, "a", encoding="utf-8") as f:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    f.write(f"[{ts}] {msg}\n")
            except Exception:
                pass

        log_debug("=" * 80)
        log_debug(f"FIND ACTION EXECUTION START: {action.id}")
        log_debug(f"  Action type: {action.type}")
        log_debug(f"  Action config type: {type(typed_config)}")
        log_debug(f"  Target type: {type(typed_config.target)}")
        log_debug(f"  Target config: {typed_config.target}")

        logger.debug(f"Executing FIND action: {action.id}")

        try:
            log_debug("STEP 1: Calling target_resolver.resolve()")
            log_debug(f"  Target resolver type: {type(self.target_resolver)}")
            log_debug(f"  Target config being passed: {typed_config.target}")

            # Resolve target to ActionResult
            result = self.target_resolver.resolve(typed_config.target)

            log_debug("STEP 2: Target resolution completed")
            log_debug(f"  Result is None: {result is None}")

            if result:
                log_debug(f"  Result type: {type(result)}")
                log_debug(f"  Result.success: {result.success}")
                log_debug(f"  Result.matches: {result.matches}")
                log_debug(f"  Result.matches count: {len(result.matches) if result.matches else 0}")

                if result.matches:
                    for i, match in enumerate(result.matches):
                        log_debug(f"  Match {i}: {match}")
                        if hasattr(match, "score"):
                            log_debug(f"    Score: {match.score}")
                        if hasattr(match, "location"):
                            log_debug(f"    Location: {match.location}")
            else:
                log_debug("  Result is None - no matches found")

            if result and result.success:
                log_debug("STEP 3: SUCCESS - Result has matches")
                logger.info(f"FIND action succeeded: found {len(result.matches)} matches")
                log_debug("FIND ACTION RETURNING: True")
                log_debug("=" * 80 + "\n")
                return True
            else:
                log_debug("STEP 3: FAILURE - No matches or result failed")
                logger.warning("FIND action failed: target not found")
                log_debug("FIND ACTION RETURNING: False")
                log_debug("=" * 80 + "\n")
                return False

        except Exception as e:
            log_debug("STEP X: EXCEPTION occurred")
            log_debug(f"  Exception type: {type(e)}")
            log_debug(f"  Exception message: {str(e)}")

            error_msg = f"FIND action failed: {e}"
            logger.error(error_msg, exc_info=True)

            import traceback

            log_debug("  Full traceback:")
            try:
                with open(debug_log, "a", encoding="utf-8") as f:
                    traceback.print_exc(file=f)
            except Exception:
                pass

            self._emit_action_failure(action, error_msg)
            log_debug("FIND ACTION RETURNING: False (exception)")
            log_debug("=" * 80 + "\n")
            return False

    def _execute_vanish(self, action: Action, typed_config: VanishActionConfig) -> bool:
        """Execute VANISH action - wait for image to disappear.

        Args:
            action: Pydantic Action model
            typed_config: Pre-validated VanishActionConfig

        Returns:
            bool: True if element vanished within timeout
        """
        logger.debug(f"Executing VANISH action: {action.id}")

        # Get timeout from config
        max_wait_time = typed_config.max_wait_time or 30000  # Default 30 seconds
        poll_interval = typed_config.poll_interval or 500  # Default 500ms

        timeout_seconds = max_wait_time / 1000.0
        poll_seconds = poll_interval / 1000.0

        logger.debug(f"VANISH timeout: {timeout_seconds}s, poll interval: {poll_seconds}s")

        try:
            start_time = self.context.time.now()
            elapsed = 0.0

            while elapsed < timeout_seconds:
                # Check if target still exists
                result = self.target_resolver.resolve(typed_config.target)

                if result is None or not result.success:
                    logger.info(f"VANISH action succeeded: element vanished after {elapsed:.2f}s")
                    return True

                # Wait before next poll
                self.context.time.wait(poll_seconds)
                elapsed = (self.context.time.now() - start_time).total_seconds()

            logger.warning(
                f"VANISH action failed: element did not vanish within {timeout_seconds}s"
            )
            return False

        except Exception as e:
            error_msg = f"VANISH action failed: {e}"
            logger.error(error_msg, exc_info=True)
            self._emit_action_failure(action, error_msg)
            return False
