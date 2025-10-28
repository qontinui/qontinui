"""Debug manager - central singleton for managing debugging operations.

This module provides the DebugManager singleton which coordinates all
debugging activities including sessions, breakpoints, and execution recording.
"""

import logging
import threading
import time
from typing import Any, Optional

from .breakpoint_manager import BreakpointManager
from .debug_session import DebugSession
from .execution_recorder import ExecutionRecorder
from .types import DebugHookContext, ErrorHook, PostActionHook, PreActionHook

logger = logging.getLogger(__name__)


class DebugManager:
    """Singleton manager for coordinating all debugging operations.

    The DebugManager serves as the central coordination point for the
    debugging subsystem. It manages sessions, breakpoints, recording,
    and provides hooks for integration with the action execution system.

    This is implemented as a thread-safe singleton to ensure consistent
    debugging state across the application.
    """

    _instance: Optional["DebugManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "DebugManager":
        """Create or return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the debug manager (only runs once)."""
        if self._initialized:
            return

        self._sessions: dict[str, DebugSession] = {}
        self._active_session_id: str | None = None

        self._breakpoint_manager = BreakpointManager()
        self._execution_recorder = ExecutionRecorder()

        self._enabled = False
        self._instance_lock = threading.RLock()

        # Hook callbacks
        self._pre_action_hooks: list[PreActionHook] = []
        self._post_action_hooks: list[PostActionHook] = []
        self._error_hooks: list[ErrorHook] = []

        self._initialized = True
        logger.info("DebugManager initialized")

    @classmethod
    def get_instance(cls) -> "DebugManager":
        """Get the singleton instance.

        Returns:
            The DebugManager singleton instance
        """
        return cls()

    @property
    def enabled(self) -> bool:
        """Check if debugging is enabled."""
        with self._instance_lock:
            return self._enabled

    def enable_debugging(self) -> None:
        """Enable debugging globally."""
        with self._instance_lock:
            if not self._enabled:
                self._enabled = True
                self._execution_recorder.enable_recording()
                logger.info("Debugging enabled")

    def disable_debugging(self) -> None:
        """Disable debugging globally."""
        with self._instance_lock:
            if self._enabled:
                self._enabled = False
                self._execution_recorder.disable_recording()
                logger.info("Debugging disabled")

    def create_session(self, name: str = "") -> DebugSession:
        """Create a new debug session.

        Args:
            name: Human-readable name for the session

        Returns:
            Created DebugSession
        """
        with self._instance_lock:
            session = DebugSession(name)
            self._sessions[session.id] = session

            # Set as active if no active session
            if self._active_session_id is None:
                self._active_session_id = session.id

            logger.info(f"Created debug session: {session.name} ({session.id[:8]})")
            return session

    def get_session(self, session_id: str) -> DebugSession | None:
        """Get a debug session by ID.

        Args:
            session_id: Session ID to retrieve

        Returns:
            DebugSession if found, None otherwise
        """
        with self._instance_lock:
            return self._sessions.get(session_id)

    def get_active_session(self) -> DebugSession | None:
        """Get the currently active debug session.

        Returns:
            Active DebugSession if set, None otherwise
        """
        with self._instance_lock:
            if self._active_session_id:
                return self._sessions.get(self._active_session_id)
            return None

    def set_active_session(self, session_id: str) -> bool:
        """Set the active debug session.

        Args:
            session_id: Session ID to make active

        Returns:
            True if session was found and set active
        """
        with self._instance_lock:
            if session_id in self._sessions:
                self._active_session_id = session_id
                logger.info(f"Set active session: {session_id[:8]}")
                return True
            return False

    def list_sessions(self) -> list[DebugSession]:
        """List all debug sessions.

        Returns:
            List of all debug sessions
        """
        with self._instance_lock:
            return list(self._sessions.values())

    def remove_session(self, session_id: str) -> bool:
        """Remove a debug session.

        Args:
            session_id: Session ID to remove

        Returns:
            True if session was removed
        """
        with self._instance_lock:
            session = self._sessions.pop(session_id, None)
            if session:
                # Clear active if this was active
                if self._active_session_id == session_id:
                    self._active_session_id = None
                logger.info(f"Removed session: {session_id[:8]}")
                return True
            return False

    @property
    def breakpoints(self) -> BreakpointManager:
        """Get the breakpoint manager.

        Returns:
            BreakpointManager instance
        """
        return self._breakpoint_manager

    @property
    def recorder(self) -> ExecutionRecorder:
        """Get the execution recorder.

        Returns:
            ExecutionRecorder instance
        """
        return self._execution_recorder

    def register_pre_action_hook(self, hook: PreActionHook) -> None:
        """Register a pre-action hook.

        Args:
            hook: Callable to invoke before each action
        """
        with self._instance_lock:
            if hook not in self._pre_action_hooks:
                self._pre_action_hooks.append(hook)
                logger.debug("Registered pre-action hook")

    def register_post_action_hook(self, hook: PostActionHook) -> None:
        """Register a post-action hook.

        Args:
            hook: Callable to invoke after each action
        """
        with self._instance_lock:
            if hook not in self._post_action_hooks:
                self._post_action_hooks.append(hook)
                logger.debug("Registered post-action hook")

    def register_error_hook(self, hook: ErrorHook) -> None:
        """Register an error hook.

        Args:
            hook: Callable to invoke when action errors occur
        """
        with self._instance_lock:
            if hook not in self._error_hooks:
                self._error_hooks.append(hook)
                logger.debug("Registered error hook")

    def on_action_start(self, context: DebugHookContext) -> None:
        """Called when an action is about to start.

        This is the main integration point with the action system.

        Args:
            context: Action context information
        """
        if not self._enabled:
            return

        start_time = time.time()

        try:
            # Get active session
            session = self.get_active_session()
            if session:
                # Update session state
                session.start_action(context.action_id)

                # Check if we should pause for stepping
                session.check_pause()

                # Check breakpoints
                ctx_dict = context.to_dict()
                should_break, triggered = self._breakpoint_manager.check_breakpoint(ctx_dict)

                if should_break and triggered:
                    logger.info(f"Breakpoint hit: {', '.join(bp.type.value for bp in triggered)}")
                    session.pause()

            # Record action start
            self._execution_recorder.record_action_start(
                action_id=context.action_id,
                action_type=context.action_type,
                action_description=context.action_description,
                session_id=context.session_id,
                input_data=context.extra,
            )

            # Call registered hooks
            for hook in self._pre_action_hooks:
                try:
                    hook(context)
                except Exception as e:
                    logger.error(f"Pre-action hook error: {e}", exc_info=True)

        finally:
            # Track hook overhead
            overhead = (time.time() - start_time) * 1000
            if overhead > 10:  # Log if overhead is significant
                logger.warning(f"Pre-action hook overhead: {overhead:.2f}ms")

    def on_action_complete(self, context: DebugHookContext) -> None:
        """Called when an action completes.

        Args:
            context: Action context with result information
        """
        if not self._enabled:
            return

        try:
            # Get active session
            session = self.get_active_session()
            if session:
                # Update session state
                session.end_action(context.action_id)

                # Create variable snapshot if result contains data
                if context.result:
                    variables = {
                        "success": context.result.success,
                        "match_count": (
                            len(context.result.match_list)
                            if hasattr(context.result, "match_list")
                            else 0
                        ),
                    }
                    session.snapshot_variables(context.action_id, variables)

            # Record completion
            if context.result:
                duration_ms = (
                    context.result.duration.total_seconds() * 1000
                    if hasattr(context.result, "duration") and context.result.duration
                    else 0.0
                )

                record = self._execution_recorder.record_action_start(
                    action_id=context.action_id,
                    action_type=context.action_type,
                    action_description=context.action_description,
                    session_id=context.session_id,
                )

                match_count = (
                    len(context.result.match_list) if hasattr(context.result, "match_list") else 0
                )

                self._execution_recorder.record_action_complete(
                    record=record,
                    success=context.result.success,
                    duration_ms=duration_ms,
                    match_count=match_count,
                )

            # Call registered hooks
            for hook in self._post_action_hooks:
                try:
                    hook(context)
                except Exception as e:
                    logger.error(f"Post-action hook error: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Error in on_action_complete: {e}", exc_info=True)

    def on_error(self, context: DebugHookContext) -> None:
        """Called when an action encounters an error.

        Args:
            context: Action context with error information
        """
        if not self._enabled:
            return

        try:
            # Get active session
            session = self.get_active_session()
            if session:
                session.error()

            # Check error breakpoints
            ctx_dict = context.to_dict()
            ctx_dict["has_error"] = True
            should_break, _ = self._breakpoint_manager.check_breakpoint(ctx_dict)

            if should_break and session:
                logger.info("Error breakpoint hit")
                session.pause()

            # Call registered error hooks
            for hook in self._error_hooks:
                try:
                    hook(context)
                except Exception as e:
                    logger.error(f"Error hook error: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Error in on_error: {e}", exc_info=True)

    def get_statistics(self) -> dict[str, Any]:
        """Get debugging statistics.

        Returns:
            Dictionary containing statistics
        """
        with self._instance_lock:
            return {
                "enabled": self._enabled,
                "sessions": len(self._sessions),
                "active_session": self._active_session_id[:8] if self._active_session_id else None,
                "breakpoints": self._breakpoint_manager.get_statistics(),
                "execution": self._execution_recorder.get_statistics(),
            }

    def __repr__(self) -> str:
        """String representation of debug manager."""
        with self._instance_lock:
            return (
                f"DebugManager(enabled={self._enabled}, "
                f"sessions={len(self._sessions)}, "
                f"breakpoints={len(self._breakpoint_manager.list_breakpoints())})"
            )
