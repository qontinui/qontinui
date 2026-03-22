"""Aspects package - ported from Qontinui framework.

Provides aspect-oriented programming features for
cross-cutting concerns.
"""

# Annotations
from .annotations import CollectedData  # Data collection; Monitoring; Recovery
from .annotations import (
    MonitoredConfig,
    clear_collected_data,
    collect_data,
    get_collect_data_config,
    get_collected_data,
    get_monitored_config,
    get_recoverable_config,
    is_collecting_data,
    is_monitored,
    is_recoverable,
    monitored,
    recoverable,
)

# Registry
from .aspect_registry import (
    AspectConfiguration,
    AspectRegistry,
    configure_aspects,
    get_aspect_registry,
)

# Core aspects
from .core import (
    ActionContext,
    ActionLifecycleAspect,
    get_lifecycle_aspect,
    with_lifecycle,
)

# Monitoring aspects
from .monitoring import (
    MethodPerformanceStats,
    PerformanceMonitoringAspect,
    StateNode,
    StateTransitionAspect,
    TransitionStats,
    get_performance_aspect,
    get_state_transition_aspect,
    performance_monitored,
    track_state_transition,
)

# Recovery aspects
from .recovery import (
    CircuitBreaker,
    DefaultRecoveryHandler,
    ErrorRecoveryAspect,
    RecoveryHandler,
    RetryPolicy,
    RetryStrategy,
    get_error_recovery_aspect,
    with_error_recovery,
)

__all__ = [
    # Annotations
    "monitored",
    "is_monitored",
    "get_monitored_config",
    "MonitoredConfig",
    "recoverable",
    "is_recoverable",
    "get_recoverable_config",
    "collect_data",
    "is_collecting_data",
    "get_collect_data_config",
    "get_collected_data",
    "clear_collected_data",
    "CollectedData",
    # Core aspects
    "ActionLifecycleAspect",
    "ActionContext",
    "with_lifecycle",
    "get_lifecycle_aspect",
    # Monitoring
    "PerformanceMonitoringAspect",
    "MethodPerformanceStats",
    "performance_monitored",
    "get_performance_aspect",
    "StateTransitionAspect",
    "TransitionStats",
    "StateNode",
    "track_state_transition",
    "get_state_transition_aspect",
    # Recovery
    "ErrorRecoveryAspect",
    "RetryPolicy",
    "RetryStrategy",
    "RecoveryHandler",
    "DefaultRecoveryHandler",
    "CircuitBreaker",
    "with_error_recovery",
    "get_error_recovery_aspect",
    # Registry
    "AspectRegistry",
    "AspectConfiguration",
    "get_aspect_registry",
    "configure_aspects",
]
