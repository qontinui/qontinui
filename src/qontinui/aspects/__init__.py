"""Aspects package - ported from Qontinui framework.

Provides aspect-oriented programming features for
cross-cutting concerns.
"""

# Annotations
from .annotations import (
    # Monitoring
    monitored,
    is_monitored,
    get_monitored_config,
    MonitoredConfig,
    
    # Recovery
    recoverable,
    is_recoverable,
    get_recoverable_config,
    
    # Data collection
    collect_data,
    is_collecting_data,
    get_collect_data_config,
    get_collected_data,
    clear_collected_data,
    CollectedData,
)

# Core aspects
from .core import (
    ActionLifecycleAspect,
    ActionContext,
    with_lifecycle,
    get_lifecycle_aspect,
)

# Monitoring aspects
from .monitoring import (
    PerformanceMonitoringAspect,
    MethodPerformanceStats,
    performance_monitored,
    get_performance_aspect,
    StateTransitionAspect,
    TransitionStats,
    StateNode,
    track_state_transition,
    get_state_transition_aspect,
)

# Recovery aspects
from .recovery import (
    ErrorRecoveryAspect,
    RetryPolicy,
    RetryStrategy,
    RecoveryHandler,
    DefaultRecoveryHandler,
    CircuitBreaker,
    with_error_recovery,
    get_error_recovery_aspect,
)

# Registry
from .aspect_registry import (
    AspectRegistry,
    AspectConfiguration,
    get_aspect_registry,
    configure_aspects,
)

__all__ = [
    # Annotations
    'monitored',
    'is_monitored',
    'get_monitored_config',
    'MonitoredConfig',
    'recoverable',
    'is_recoverable',
    'get_recoverable_config',
    'collect_data',
    'is_collecting_data',
    'get_collect_data_config',
    'get_collected_data',
    'clear_collected_data',
    'CollectedData',
    
    # Core aspects
    'ActionLifecycleAspect',
    'ActionContext',
    'with_lifecycle',
    'get_lifecycle_aspect',
    
    # Monitoring
    'PerformanceMonitoringAspect',
    'MethodPerformanceStats',
    'performance_monitored',
    'get_performance_aspect',
    'StateTransitionAspect',
    'TransitionStats',
    'StateNode',
    'track_state_transition',
    'get_state_transition_aspect',
    
    # Recovery
    'ErrorRecoveryAspect',
    'RetryPolicy',
    'RetryStrategy',
    'RecoveryHandler',
    'DefaultRecoveryHandler',
    'CircuitBreaker',
    'with_error_recovery',
    'get_error_recovery_aspect',
    
    # Registry
    'AspectRegistry',
    'AspectConfiguration',
    'get_aspect_registry',
    'configure_aspects',
]