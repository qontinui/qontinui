"""Recovery aspects - ported from Qontinui framework.

Error recovery and resilience patterns.
"""

from .error_recovery_aspect import (
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
    "ErrorRecoveryAspect",
    "RetryPolicy",
    "RetryStrategy",
    "RecoveryHandler",
    "DefaultRecoveryHandler",
    "CircuitBreaker",
    "with_error_recovery",
    "get_error_recovery_aspect",
]
