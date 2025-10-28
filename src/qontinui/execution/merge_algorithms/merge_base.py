"""
Base interface for merge strategies.

This module provides the abstract base class that all merge strategies
must implement.
"""

from abc import ABC, abstractmethod


class MergeStrategy(ABC):
    """
    Abstract base class for merge strategies.

    A merge strategy determines when a merge node should execute based on
    which input paths have completed.
    """

    def __init__(self, name: str, description: str) -> None:
        """
        Initialize strategy.

        Args:
            name: Strategy name
            description: Human-readable description
        """
        self.name = name
        self.description = description

    @abstractmethod
    def should_execute(self, received_inputs: int, total_inputs: int, **kwargs) -> bool:
        """
        Determine if the merge node should execute.

        Args:
            received_inputs: Number of inputs received so far
            total_inputs: Total number of expected inputs
            **kwargs: Additional strategy-specific parameters

        Returns:
            True if the merge should execute now
        """
        pass

    def get_merge_mode(self) -> str:
        """
        Get the merge mode for context merging.

        Returns:
            One of: 'all', 'any', 'first'
        """
        return "all"

    def should_wait_for_more(self, received_inputs: int, total_inputs: int, **kwargs) -> bool:
        """
        Determine if we should wait for more inputs.

        Args:
            received_inputs: Number of inputs received
            total_inputs: Total number of expected inputs
            **kwargs: Additional parameters

        Returns:
            True if should wait for more inputs
        """
        return not self.should_execute(received_inputs, total_inputs, **kwargs)

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"
