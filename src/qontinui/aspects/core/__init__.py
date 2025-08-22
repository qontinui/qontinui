"""Core aspects - ported from Qontinui framework.

Core cross-cutting concerns for the framework.
"""

from .action_lifecycle_aspect import (
    ActionLifecycleAspect,
    ActionContext,
    with_lifecycle,
    get_lifecycle_aspect
)

__all__ = [
    'ActionLifecycleAspect',
    'ActionContext',
    'with_lifecycle',
    'get_lifecycle_aspect',
]