"""Aspect registry - central management for all aspects.

Provides configuration and lifecycle management for aspects.
"""

from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass
from .core import get_lifecycle_aspect, ActionLifecycleAspect
from .monitoring import (
    get_performance_aspect, 
    get_state_transition_aspect,
    PerformanceMonitoringAspect,
    StateTransitionAspect
)
from .recovery import get_error_recovery_aspect, ErrorRecoveryAspect

logger = logging.getLogger(__name__)


@dataclass
class AspectConfiguration:
    """Configuration for all aspects."""
    
    # Action lifecycle
    lifecycle_enabled: bool = True
    lifecycle_log_events: bool = True
    lifecycle_pre_pause: float = 0.0
    lifecycle_post_pause: float = 0.0
    lifecycle_capture_before: bool = False
    lifecycle_capture_after: bool = True
    
    # Performance monitoring
    performance_enabled: bool = True
    performance_alert_threshold_ms: float = 10000
    performance_warning_threshold_ms: float = 5000
    performance_report_interval: int = 300
    performance_track_memory: bool = True
    
    # State transitions
    state_transition_enabled: bool = True
    state_transition_track_success: bool = True
    state_transition_visualizations: bool = True
    
    # Error recovery
    recovery_enabled: bool = True
    recovery_max_attempts: int = 3
    recovery_initial_delay_ms: float = 1000
    recovery_max_delay_ms: float = 30000
    recovery_backoff_multiplier: float = 2.0


class AspectRegistry:
    """Central registry for all aspects.
    
    Manages configuration and lifecycle of aspects,
    providing a single point of control for cross-cutting concerns.
    """
    
    def __init__(self, config: Optional[AspectConfiguration] = None):
        """Initialize the registry.
        
        Args:
            config: Aspect configuration
        """
        self.config = config or AspectConfiguration()
        
        # Get aspect instances
        self.lifecycle_aspect = get_lifecycle_aspect()
        self.performance_aspect = get_performance_aspect()
        self.state_transition_aspect = get_state_transition_aspect()
        self.error_recovery_aspect = get_error_recovery_aspect()
        
        # Apply configuration
        self.apply_configuration()
    
    def apply_configuration(self) -> None:
        """Apply configuration to all aspects."""
        # Configure lifecycle aspect
        self.lifecycle_aspect.pre_action_pause = self.config.lifecycle_pre_pause
        self.lifecycle_aspect.post_action_pause = self.config.lifecycle_post_pause
        self.lifecycle_aspect.log_events = self.config.lifecycle_log_events
        self.lifecycle_aspect.capture_before_screenshot = self.config.lifecycle_capture_before
        self.lifecycle_aspect.capture_after_screenshot = self.config.lifecycle_capture_after
        
        # Configure performance aspect
        self.performance_aspect.enabled = self.config.performance_enabled
        self.performance_aspect.alert_threshold_ms = self.config.performance_alert_threshold_ms
        self.performance_aspect.warning_threshold_ms = self.config.performance_warning_threshold_ms
        self.performance_aspect.report_interval_seconds = self.config.performance_report_interval
        self.performance_aspect.track_memory = self.config.performance_track_memory
        
        # Configure state transition aspect
        self.state_transition_aspect.enabled = self.config.state_transition_enabled
        self.state_transition_aspect.track_success_rates = self.config.state_transition_track_success
        self.state_transition_aspect.generate_visualizations = self.config.state_transition_visualizations
        
        # Configure error recovery aspect
        self.error_recovery_aspect.enabled = self.config.recovery_enabled
        self.error_recovery_aspect.default_policy.max_attempts = self.config.recovery_max_attempts
        self.error_recovery_aspect.default_policy.initial_delay_ms = self.config.recovery_initial_delay_ms
        self.error_recovery_aspect.default_policy.max_delay_ms = self.config.recovery_max_delay_ms
        self.error_recovery_aspect.default_policy.backoff_multiplier = self.config.recovery_backoff_multiplier
        
        logger.info("Aspect configuration applied")
    
    def update_configuration(self, **kwargs) -> None:
        """Update configuration dynamically.
        
        Args:
            **kwargs: Configuration values to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")
        
        self.apply_configuration()
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all aspects.
        
        Returns:
            Dictionary with aspect status
        """
        return {
            'lifecycle': {
                'metrics': self.lifecycle_aspect.get_metrics(),
                'current_context': self.lifecycle_aspect.get_current_context()
            },
            'performance': {
                'enabled': self.performance_aspect.enabled,
                'stats_count': len(self.performance_aspect.get_performance_stats()),
                'slowest_operations': self.performance_aspect.get_slowest_operations(5)
            },
            'state_transition': {
                'enabled': self.state_transition_aspect.enabled,
                'state_count': len(self.state_transition_aspect.get_state_graph()),
                'transition_count': len(self.state_transition_aspect.get_transition_stats())
            },
            'error_recovery': {
                'enabled': self.error_recovery_aspect.enabled,
                'error_stats': self.error_recovery_aspect.get_error_stats()
            }
        }
    
    def reset_all(self) -> None:
        """Reset all aspect data."""
        self.lifecycle_aspect.reset_metrics()
        self.performance_aspect.reset_stats()
        self.state_transition_aspect.reset_tracking()
        logger.info("All aspects reset")
    
    def generate_reports(self) -> Dict[str, str]:
        """Generate reports from all aspects.
        
        Returns:
            Dictionary of report names to report content
        """
        reports = {}
        
        # Performance report
        reports['performance'] = self.performance_aspect.generate_report()
        
        # State transition visualization
        reports['state_graph'] = self.state_transition_aspect.generate_dot_graph()
        
        # Navigation patterns
        patterns = self.state_transition_aspect.get_navigation_patterns()
        if patterns:
            pattern_lines = ["Navigation Patterns:"]
            for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:10]:
                pattern_lines.append(f"  {' -> '.join(pattern)}: {count} occurrences")
            reports['navigation_patterns'] = "\\n".join(pattern_lines)
        
        return reports
    
    def enable_all(self) -> None:
        """Enable all aspects."""
        self.config.lifecycle_enabled = True
        self.config.performance_enabled = True
        self.config.state_transition_enabled = True
        self.config.recovery_enabled = True
        self.apply_configuration()
        logger.info("All aspects enabled")
    
    def disable_all(self) -> None:
        """Disable all aspects."""
        self.config.lifecycle_enabled = False
        self.config.performance_enabled = False
        self.config.state_transition_enabled = False
        self.config.recovery_enabled = False
        self.apply_configuration()
        logger.info("All aspects disabled")


# Global registry instance
_aspect_registry: Optional[AspectRegistry] = None


def get_aspect_registry(config: Optional[AspectConfiguration] = None) -> AspectRegistry:
    """Get or create the global aspect registry.
    
    Args:
        config: Configuration to use if creating new registry
        
    Returns:
        The aspect registry
    """
    global _aspect_registry
    
    if _aspect_registry is None:
        _aspect_registry = AspectRegistry(config)
    
    return _aspect_registry


def configure_aspects(**kwargs) -> None:
    """Configure aspects with keyword arguments.
    
    Args:
        **kwargs: Configuration values
    """
    registry = get_aspect_registry()
    registry.update_configuration(**kwargs)