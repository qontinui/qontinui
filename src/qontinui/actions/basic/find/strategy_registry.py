"""Strategy registry - ported from Qontinui framework.

Registry for find strategy implementations.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from .options.base_find_options import FindStrategy

logger = logging.getLogger(__name__)


@dataclass
class StrategyRegistry:
    """Registry for find strategy implementations.

    Port of ModernFindStrategyRegistry from Qontinui framework.

    Maps FindStrategy enum values to their implementation classes,
    allowing dynamic strategy selection at runtime.
    """

    # Registry of strategy implementations
    _implementations: dict[FindStrategy, Any] = field(default_factory=dict)

    # Lazy initialization flags
    _initialized: bool = False

    def __post_init__(self):
        """Initialize default implementations."""
        if not self._initialized:
            self._register_default_implementations()
            self._initialized = True

    def _register_default_implementations(self):
        """Register default strategy implementations."""
        # Template matching (removed - use RealFindImplementation instead)
        # self.register(FindStrategy.TEMPLATE, ImageFinder())

        # Text finding (when implemented)
        # self.register(FindStrategy.TEXT, TextFinder())

        # Color matching (when implemented)
        # self.register(FindStrategy.COLOR, ColorFinder())

        # Motion detection (when implemented)
        # self.register(FindStrategy.MOTION, MotionFinder())

        logger.debug(f"Registered {len(self._implementations)} default implementations")

    def register(self, strategy: FindStrategy, implementation: Any):
        """Register a strategy implementation.

        Args:
            strategy: Strategy enum value
            implementation: Implementation instance
        """
        self._implementations[strategy] = implementation
        logger.debug(f"Registered implementation for {strategy.name}")

    def unregister(self, strategy: FindStrategy):
        """Unregister a strategy implementation.

        Args:
            strategy: Strategy to unregister
        """
        if strategy in self._implementations:
            del self._implementations[strategy]
            logger.debug(f"Unregistered implementation for {strategy.name}")

    def get_implementation(self, strategy: FindStrategy) -> Any | None:
        """Get implementation for a strategy.

        Args:
            strategy: Strategy to get implementation for

        Returns:
            Implementation instance or None
        """
        impl = self._implementations.get(strategy)
        if impl is None:
            logger.warning(f"No implementation registered for {strategy.name}")
        return impl

    def has_implementation(self, strategy: FindStrategy) -> bool:
        """Check if strategy has an implementation.

        Args:
            strategy: Strategy to check

        Returns:
            True if implementation exists
        """
        return strategy in self._implementations

    def get_available_strategies(self) -> list[FindStrategy]:
        """Get list of strategies with implementations.

        Returns:
            List of available strategies
        """
        return list(self._implementations.keys())

    def clear(self):
        """Clear all registered implementations."""
        self._implementations.clear()
        self._initialized = False
        logger.debug("Registry cleared")

    def replace_implementation(self, strategy: FindStrategy, implementation: Any):
        """Replace an existing implementation.

        Args:
            strategy: Strategy to replace
            implementation: New implementation
        """
        old_impl = self._implementations.get(strategy)
        self._implementations[strategy] = implementation

        if old_impl:
            logger.info(f"Replaced implementation for {strategy.name}")
        else:
            logger.info(f"Added new implementation for {strategy.name}")


class MLStrategyRegistry(StrategyRegistry):
    """Registry with ML-enhanced implementations.

    Extended registry that provides ML-based implementations
    when available, falling back to traditional methods.
    """

    def _register_default_implementations(self):
        """Register ML-enhanced implementations."""
        super()._register_default_implementations()

        # Register ML object detection if available
        ml_detector = self._create_ml_detector()
        if ml_detector:
            self.register(FindStrategy.ML_OBJECT, ml_detector)
            logger.info("ML object detector registered")

    def _create_ml_detector(self) -> Any | None:
        """Create ML detector if dependencies available.

        Returns:
            ML detector instance or None
        """
        try:
            # Try to import YOLO or other ML framework
            # This is where you'd integrate modern ML
            logger.debug("Checking for ML frameworks...")

            # Placeholder for actual ML integration
            # from ..modern.yolo_finder import YOLOFinder
            # return YOLOFinder()

            return None

        except ImportError as e:
            logger.debug(f"ML framework not available: {e}")
            return None

    def upgrade_to_ml(self, strategy: FindStrategy):
        """Upgrade a strategy to use ML implementation.

        Args:
            strategy: Strategy to upgrade
        """
        ml_impl = None

        if strategy == FindStrategy.TEMPLATE:
            # Upgrade template matching to object detection
            ml_impl = self._create_ml_detector()
        elif strategy == FindStrategy.TEXT:
            # Upgrade to modern OCR
            # ml_impl = self._create_ml_ocr()
            pass

        if ml_impl:
            self.replace_implementation(strategy, ml_impl)
            logger.info(f"Upgraded {strategy.name} to ML implementation")
        else:
            logger.warning(f"No ML upgrade available for {strategy.name}")


@dataclass
class CompositeRegistry:
    """Composite registry that tries multiple registries.

    Allows chaining registries for fallback behavior.
    """

    # Ordered list of registries to try
    registries: list[StrategyRegistry] = field(default_factory=list)

    def add_registry(self, registry: StrategyRegistry):
        """Add a registry to the chain.

        Args:
            registry: Registry to add
        """
        self.registries.append(registry)

    def get_implementation(self, strategy: FindStrategy) -> Any | None:
        """Get implementation from first registry that has it.

        Args:
            strategy: Strategy to find

        Returns:
            Implementation or None
        """
        for registry in self.registries:
            impl = registry.get_implementation(strategy)
            if impl is not None:
                return impl
        return None

    def has_implementation(self, strategy: FindStrategy) -> bool:
        """Check if any registry has the strategy.

        Args:
            strategy: Strategy to check

        Returns:
            True if any registry has it
        """
        return any(r.has_implementation(strategy) for r in self.registries)


# Global default registry instance
_default_registry = StrategyRegistry()


def get_default_registry() -> StrategyRegistry:
    """Get the default global registry.

    Returns:
        Default registry instance
    """
    return _default_registry


def set_default_registry(registry: StrategyRegistry):
    """Set the default global registry.

    Args:
        registry: New default registry
    """
    global _default_registry
    _default_registry = registry
    logger.info("Default registry replaced")
