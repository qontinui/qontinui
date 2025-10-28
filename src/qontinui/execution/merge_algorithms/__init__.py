"""
Merge strategy algorithms.

This package contains individual merge strategy implementations.
"""

from .custom_strategy import CustomStrategy
from .majority_strategy import MajorityStrategy
from .merge_base import MergeStrategy
from .timeout_strategy import TimeoutStrategy
from .wait_all_strategy import WaitAllStrategy
from .wait_any_strategy import WaitAnyStrategy
from .wait_first_strategy import WaitFirstStrategy

__all__ = [
    "MergeStrategy",
    "WaitAllStrategy",
    "WaitAnyStrategy",
    "WaitFirstStrategy",
    "TimeoutStrategy",
    "MajorityStrategy",
    "CustomStrategy",
]
