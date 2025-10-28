"""
Custom merge strategy implementation.

This strategy allows user-defined merge logic through a callable.
"""

from .merge_base import MergeStrategy


class CustomStrategy(MergeStrategy):
    """
    Custom strategy with user-defined logic.

    Allows implementing custom merge logic by providing a callable
    that determines when to execute.
    """

    def __init__(self, name: str, should_execute_func, description: str = "Custom merge strategy") -> None:
        """
        Initialize CustomStrategy.

        Args:
            name: Strategy name
            should_execute_func: Callable that takes (received_inputs, total_inputs, **kwargs)
                                and returns bool
            description: Strategy description
        """
        super().__init__(name=name, description=description)
        self.should_execute_func = should_execute_func

    def should_execute(self, received_inputs: int, total_inputs: int, **kwargs) -> bool:
        """Execute based on custom function."""
        return self.should_execute_func(received_inputs, total_inputs, **kwargs)
