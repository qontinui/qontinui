"""Action task for multiple execution."""

import time
from typing import Any

from ...action_interface import ActionInterface


class ActionTask:
    """Individual action task for multiple execution."""

    def __init__(
        self,
        action: ActionInterface,
        target: Any | None = None,
        priority: int = 0,
        group: int = 0,
        name: str | None = None,
    ) -> None:
        """Initialize action task.

        Args:
            action: Action to execute
            target: Target for action
            priority: Priority (higher = earlier)
            group: Group number for grouped execution
            name: Optional name for identification
        """
        self.action = action
        self.target = target
        self.priority = priority
        self.group = group
        self.name = name or f"Task_{id(self)}"
        self.result: bool | None = None
        self.error: Exception | None = None
        self.start_time: float | None = None
        self.end_time: float | None = None

    def execute(self) -> bool:
        """Execute the action task.

        Returns:
            True if successful
        """
        self.start_time = time.time()
        try:
            if self.target is not None:
                if hasattr(self.action, "execute"):
                    self.result = self.action.execute(self.target)
                elif callable(self.action):
                    self.result = self.action(self.target)  # type: ignore
                else:
                    raise RuntimeError(f"Action {self.action} is not executable")
            else:
                if hasattr(self.action, "execute"):
                    self.result = self.action.execute()
                elif callable(self.action):
                    self.result = self.action()  # type: ignore
                else:
                    raise RuntimeError(f"Action {self.action} is not executable")

            self.end_time = time.time()
            return self.result

        except Exception as e:
            self.error = e
            self.result = False
            self.end_time = time.time()
            return False

    @property
    def duration(self) -> float:
        """Get execution duration.

        Returns:
            Duration in seconds
        """
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
