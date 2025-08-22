"""Collect data annotation - ported from Qontinui framework.

Marks methods for data collection during execution.
"""

from typing import Optional, Callable, Any, Dict, List
from functools import wraps
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CollectedData:
    """Container for collected method execution data."""
    
    method_name: str
    """Name of the method that was executed."""
    
    start_time: float
    """Start time of execution (timestamp)."""
    
    end_time: float
    """End time of execution (timestamp)."""
    
    duration_ms: float
    """Duration of execution in milliseconds."""
    
    args: tuple = ()
    """Arguments passed to the method."""
    
    kwargs: dict = field(default_factory=dict)
    """Keyword arguments passed to the method."""
    
    result: Any = None
    """Result returned by the method."""
    
    exception: Optional[Exception] = None
    """Exception raised during execution, if any."""
    
    custom_data: Dict[str, Any] = field(default_factory=dict)
    """Custom data collected during execution."""
    
    tags: List[str] = field(default_factory=list)
    """Tags associated with this execution."""
    
    @property
    def successful(self) -> bool:
        """Check if execution was successful.
        
        Returns:
            True if no exception occurred
        """
        return self.exception is None


# Global data collector for simplicity
_collected_data: List[CollectedData] = []


def collect_data(
    name: str = "",
    collect_args: bool = False,
    collect_result: bool = False,
    tags: Optional[List[str]] = None,
    custom_collector: Optional[Callable] = None
) -> Callable:
    """Mark a method for data collection during execution.
    
    Direct port of Brobot's @CollectData annotation.
    
    Methods decorated with @collect_data will have execution
    data automatically collected for analysis and debugging.
    
    Example usage:
        @collect_data(collect_args=True, collect_result=True)
        def process_data(input_data):
            # Process and return result
            return processed_data
        
        @collect_data(
            tags=["critical", "user_action"],
            custom_collector=lambda: {"user_id": get_current_user()}
        )
        def perform_user_action():
            # Perform action with custom data collection
            pass
    
    Args:
        name: Custom name for the collected data.
             Default is the method name.
        collect_args: Whether to collect method arguments.
                     Default is False for privacy.
        collect_result: Whether to collect method result.
                       Default is False for privacy.
        tags: Tags to associate with collected data.
             Useful for filtering and analysis.
        custom_collector: Function to collect custom data.
                         Called with no arguments, should return
                         a dictionary of custom data.
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Initialize data collection
            data = CollectedData(
                method_name=name or func.__name__,
                start_time=time.time(),
                end_time=0,
                duration_ms=0,
                tags=tags or []
            )
            
            # Collect arguments if requested
            if collect_args:
                data.args = args
                data.kwargs = kwargs
            
            # Collect custom data if collector provided
            if custom_collector:
                try:
                    data.custom_data = custom_collector()
                except Exception as e:
                    logger.warning(f"Custom collector failed for {func.__name__}: {e}")
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Collect result if requested
                if collect_result:
                    data.result = result
                
                return result
                
            except Exception as e:
                # Record exception
                data.exception = e
                raise
                
            finally:
                # Record end time and duration
                data.end_time = time.time()
                data.duration_ms = (data.end_time - data.start_time) * 1000
                
                # Store collected data
                _collected_data.append(data)
                
                # Log collection
                logger.debug(
                    f"Collected data for {data.method_name}: "
                    f"duration={data.duration_ms:.2f}ms, "
                    f"success={data.successful}"
                )
        
        # Store configuration on wrapper
        wrapper._collect_data = True
        wrapper._collect_data_config = {
            'name': name,
            'collect_args': collect_args,
            'collect_result': collect_result,
            'tags': tags,
            'custom_collector': custom_collector
        }
        
        return wrapper
    
    return decorator


def is_collecting_data(obj: Any) -> bool:
    """Check if an object is collecting data.
    
    Args:
        obj: Object to check
        
    Returns:
        True if object is decorated with @collect_data
    """
    return hasattr(obj, '_collect_data') and obj._collect_data


def get_collect_data_config(obj: Any) -> Optional[dict]:
    """Get data collection configuration from an object.
    
    Args:
        obj: Data collecting object
        
    Returns:
        Data collection configuration or None
    """
    if not is_collecting_data(obj):
        return None
    
    return getattr(obj, '_collect_data_config', None)


def get_collected_data(
    method_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    successful_only: bool = False
) -> List[CollectedData]:
    """Get collected data with optional filtering.
    
    Args:
        method_name: Filter by method name
        tags: Filter by tags (data must have all specified tags)
        successful_only: Only return successful executions
        
    Returns:
        List of collected data matching filters
    """
    result = _collected_data.copy()
    
    # Filter by method name
    if method_name:
        result = [d for d in result if d.method_name == method_name]
    
    # Filter by tags
    if tags:
        result = [d for d in result if all(tag in d.tags for tag in tags)]
    
    # Filter by success
    if successful_only:
        result = [d for d in result if d.successful]
    
    return result


def clear_collected_data() -> None:
    """Clear all collected data."""
    global _collected_data
    _collected_data.clear()
    logger.info("Cleared all collected data")