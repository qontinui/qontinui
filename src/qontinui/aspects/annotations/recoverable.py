"""Recoverable annotation - ported from Qontinui framework.

Marks methods for automatic recovery on failure.
"""

from typing import Optional, Callable, Any
from functools import wraps
import logging
import time

logger = logging.getLogger(__name__)


def recoverable(
    max_retries: int = 3,
    backoff_multiplier: float = 2.0,
    initial_delay_ms: int = 1000,
    max_delay_ms: int = 30000,
    recoverable_exceptions: Optional[tuple] = None,
    fallback: Optional[Callable] = None
) -> Callable:
    """Mark a method for automatic recovery on failure.
    
    Direct port of Brobot's @Recoverable annotation.
    
    Methods decorated with @recoverable will automatically retry
    on failure with exponential backoff.
    
    Example usage:
        @recoverable(max_retries=5, initial_delay_ms=500)
        def fetch_data():
            # Operation that might fail
            pass
        
        @recoverable(
            recoverable_exceptions=(ConnectionError, TimeoutError),
            fallback=lambda: "default_value"
        )
        def connect_to_service():
            # Connection that might fail with fallback
            pass
    
    Args:
        max_retries: Maximum number of retry attempts.
                    Default is 3.
        backoff_multiplier: Multiplier for exponential backoff.
                          Default is 2.0.
        initial_delay_ms: Initial delay between retries in milliseconds.
                         Default is 1000ms.
        max_delay_ms: Maximum delay between retries in milliseconds.
                     Default is 30000ms.
        recoverable_exceptions: Tuple of exception types to recover from.
                              If None, recovers from all exceptions.
        fallback: Fallback function to call if all retries fail.
                 Should take no arguments and return a default value.
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay_ms = initial_delay_ms
            
            for attempt in range(max_retries + 1):
                try:
                    if attempt > 0:
                        logger.info(f"Retry attempt {attempt}/{max_retries} for {func.__name__}")
                    
                    result = func(*args, **kwargs)
                    
                    if attempt > 0:
                        logger.info(f"Recovery successful for {func.__name__} after {attempt} attempts")
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    # Check if exception is recoverable
                    if recoverable_exceptions and not isinstance(e, recoverable_exceptions):
                        logger.error(
                            f"Non-recoverable exception in {func.__name__}: {e.__class__.__name__}"
                        )
                        raise
                    
                    if attempt < max_retries:
                        logger.warning(
                            f"Recoverable failure in {func.__name__} (attempt {attempt + 1}/{max_retries + 1}): {e}"
                        )
                        
                        # Wait before retry with exponential backoff
                        time.sleep(delay_ms / 1000.0)
                        delay_ms = min(int(delay_ms * backoff_multiplier), max_delay_ms)
                    else:
                        logger.error(
                            f"All recovery attempts failed for {func.__name__} after {max_retries + 1} attempts"
                        )
            
            # All retries failed
            if fallback:
                logger.info(f"Using fallback for {func.__name__}")
                try:
                    return fallback()
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed for {func.__name__}", exc_info=fallback_error)
                    raise last_exception
            
            raise last_exception
        
        # Store configuration on wrapper
        wrapper._recoverable = True
        wrapper._recoverable_config = {
            'max_retries': max_retries,
            'backoff_multiplier': backoff_multiplier,
            'initial_delay_ms': initial_delay_ms,
            'max_delay_ms': max_delay_ms,
            'recoverable_exceptions': recoverable_exceptions,
            'fallback': fallback
        }
        
        return wrapper
    
    return decorator


def is_recoverable(obj: Any) -> bool:
    """Check if an object is recoverable.
    
    Args:
        obj: Object to check
        
    Returns:
        True if object is decorated with @recoverable
    """
    return hasattr(obj, '_recoverable') and obj._recoverable


def get_recoverable_config(obj: Any) -> Optional[dict]:
    """Get recovery configuration from an object.
    
    Args:
        obj: Recoverable object
        
    Returns:
        Recovery configuration or None
    """
    if not is_recoverable(obj):
        return None
    
    return getattr(obj, '_recoverable_config', None)