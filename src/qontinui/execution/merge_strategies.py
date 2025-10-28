"""
Merge strategy factory and registry.

This module provides factory functions and a registry for creating
and managing merge strategy instances.
"""

from .merge_algorithms import (
    CustomStrategy,
    MajorityStrategy,
    MergeStrategy,
    TimeoutStrategy,
    WaitAllStrategy,
    WaitAnyStrategy,
    WaitFirstStrategy,
)

# Registry of available strategies
STRATEGY_REGISTRY: dict[str, type[MergeStrategy]] = {
    "wait_all": WaitAllStrategy,
    "wait_any": WaitAnyStrategy,
    "wait_first": WaitFirstStrategy,
    "timeout": TimeoutStrategy,
    "majority": MajorityStrategy,
    "custom": CustomStrategy,
}


def create_strategy(strategy_type: str, **kwargs) -> MergeStrategy:
    """
    Create a merge strategy instance.

    Args:
        strategy_type: Type of strategy to create
        **kwargs: Strategy-specific parameters

    Returns:
        MergeStrategy instance

    Raises:
        ValueError: If strategy type is unknown
    """
    strategy_class = STRATEGY_REGISTRY.get(strategy_type)
    if strategy_class is None:
        raise ValueError(
            f"Unknown strategy type: {strategy_type}. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )

    return strategy_class(**kwargs)


def get_available_strategies() -> list[str]:
    """Get list of available strategy types."""
    return list(STRATEGY_REGISTRY.keys())


# Re-export for backward compatibility with imports
__all__ = [
    "MergeStrategy",
    "WaitAllStrategy",
    "WaitAnyStrategy",
    "WaitFirstStrategy",
    "TimeoutStrategy",
    "MajorityStrategy",
    "CustomStrategy",
    "create_strategy",
    "get_available_strategies",
    "STRATEGY_REGISTRY",
]
