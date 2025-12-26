"""ActionService - manages the registry of available actions.

This service maintains a registry of action implementations and provides
the correct action for a given configuration.

MIGRATION NOTE: All find operations now use FindAction internally.
"""

import logging

from ..actions.find import FindAction
from .action_config import ActionConfig
from .action_interface import ActionInterface
from .basic.click.click import Click

# Import action options
from .basic.click.click_options import ClickOptions

# Import basic actions
from .basic.find.find import Find

# Additional actions will be imported as they are implemented
# Import find options
from .basic.find.pattern_find_options import PatternFindOptions

# Import composite actions
from .composite.process.run_process import RunProcess
from .composite.process.run_process_options import RunProcessOptions

logger = logging.getLogger(__name__)


class ActionService:
    """Service for managing action registry and instantiation.

    Port of Brobot's ActionService class.

    This service maintains a registry mapping ActionConfig types to their
    corresponding ActionInterface implementations. When the Action class
    needs to execute an action, it uses this service to get the appropriate
    implementation based on the configuration type.

    The registry ensures that:
    - Each config type maps to exactly one action implementation
    - Actions are properly instantiated with dependencies
    - Unknown config types are handled gracefully
    """

    def __init__(self) -> None:
        """Initialize the action service with default registry."""
        self._registry: dict[type[ActionConfig], type[ActionInterface]] = {}
        self._action_instances: dict[type[ActionInterface], ActionInterface] = {}
        self._register_default_actions()
        logger.debug("ActionService initialized with default actions")

    def _register_default_actions(self) -> None:
        """Register all default action mappings."""
        # Register find actions
        self.register(PatternFindOptions, Find)

        # Register basic actions
        self.register(ClickOptions, Click)

        # Register composite actions
        self.register(RunProcessOptions, RunProcess)

        # Additional actions will be registered as they are implemented

    def register(self, config_type: type[ActionConfig], action_type: type[ActionInterface]) -> None:
        """Register an action mapping.

        Args:
            config_type: The configuration class type
            action_type: The action implementation class type
        """
        self._registry[config_type] = action_type
        logger.debug(f"Registered {action_type.__name__} for {config_type.__name__}")

    def get_action(self, action_config: ActionConfig) -> ActionInterface | None:
        """Get the action implementation for a configuration.

        This method looks up the appropriate action implementation based on
        the configuration type and returns an instance of it.

        Args:
            action_config: The action configuration

        Returns:
            The action implementation instance, or None if not found
        """
        config_type = type(action_config)

        # Look up the action type in registry
        action_type = self._registry.get(config_type)
        if not action_type:
            logger.warning(f"No action registered for config type: {config_type.__name__}")
            return None

        # Get or create action instance
        if action_type not in self._action_instances:
            try:
                # Create instance with default dependencies
                # In a real implementation, this would use dependency injection
                self._action_instances[action_type] = self._create_action_instance(action_type)
            except Exception as e:
                logger.error(f"Failed to create action instance: {e}")
                return None

        return self._action_instances[action_type]

    def _create_action_instance(self, action_type: type[ActionInterface]) -> ActionInterface:
        """Create an action instance with dependencies.

        Args:
            action_type: The action class to instantiate

        Returns:
            The action instance
        """
        # Import here to avoid circular dependencies
        from .basic.click.click import Click, SingleClickExecutor, TimeProvider
        from .basic.find.find import Find

        # For now, create with default constructor
        # In a real implementation, this would inject proper dependencies
        if action_type == Find:
            # Find now delegates to FindAction internally
            return Find(find_action=FindAction())
        elif action_type == Click:
            # Click only needs click-specific dependencies (no Find - it's atomic)
            return Click(
                options=None,
                click_location_once=SingleClickExecutor(),
                time=TimeProvider(),
            )
        else:
            # Create with no-arg constructor
            return action_type()

    def get_registered_types(self) -> dict[type[ActionConfig], type[ActionInterface]]:
        """Get all registered action mappings.

        Returns:
            Dictionary of config type to action type mappings
        """
        return self._registry.copy()

    def is_registered(self, config_type: type[ActionConfig]) -> bool:
        """Check if a config type is registered.

        Args:
            config_type: The configuration class type

        Returns:
            True if registered, False otherwise
        """
        return config_type in self._registry

    def clear_instances(self) -> None:
        """Clear cached action instances."""
        self._action_instances.clear()
        logger.debug("Cleared action instance cache")
