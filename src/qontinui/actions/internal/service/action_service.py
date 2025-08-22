"""Action service - ported from Qontinui framework.

Central service for resolving action implementations based on configuration.
"""

from typing import Optional, Callable, List, Dict
from ...action_interface import ActionInterface
from ...action_config import ActionConfig
from ...action_type import ActionType
from ...action_result import ActionResult
from ...object_collection import ObjectCollection


class ActionService:
    """Central service for resolving action implementations based on configuration.
    
    Port of ActionService from Qontinui framework.
    
    ActionService acts as a factory that examines ActionConfig and returns
    the appropriate ActionInterface implementation. It intelligently routes
    between basic and composite actions based on the complexity of the requested operation.
    
    Resolution logic:
    - Multiple Find operations → Composite FIND action
    - Single Find operation → Basic FIND action
    - Other actions → Checks basic registry first, then composite
    
    This service encapsulates the complexity of action selection, allowing the
    framework to transparently handle both simple and complex operations through
    a uniform interface.
    """
    
    def __init__(self,
                 basic_action: Optional['BasicActionRegistry'] = None,
                 find_functions: Optional['FindStrategyRegistry'] = None):
        """Construct the ActionService with required action registries.
        
        Args:
            basic_action: Registry of atomic GUI operations
            find_functions: Service for custom find implementations
        """
        self.basic_action = basic_action
        self.find_functions = find_functions
        self._custom_find = None
    
    def set_custom_find(self, custom_find: Callable[[ActionResult, List[ObjectCollection]], None]) -> None:
        """Register a custom Find implementation for one-time use.
        
        Allows runtime registration of specialized find logic without modifying
        the core framework. The custom implementation will be available for
        actions configured with custom find strategies.
        
        Custom finds are useful for:
        - Application-specific pattern matching
        - Complex multi-stage searches
        - Integration with external vision systems
        
        Args:
            custom_find: Callable that accepts ActionResult to populate and
                        ObjectCollections to search within
        """
        if self.find_functions:
            self.find_functions.add_custom_find(custom_find)
        else:
            self._custom_find = custom_find
    
    def get_action(self, action_config: ActionConfig) -> Optional[ActionInterface]:
        """Resolve the appropriate action implementation for the given ActionConfig.
        
        Since ActionConfig is type-specific (e.g., ClickOptions, FindOptions),
        this method determines the action type based on the config class name
        and returns the corresponding implementation.
        
        Args:
            action_config: Configuration specifying the desired action
            
        Returns:
            Optional containing the action implementation, or empty if not found
        """
        # Map config class names to action types
        config_class_name = action_config.__class__.__name__
        
        # Pattern-based find operations
        if 'FindOptions' in config_class_name or 'PatternFindOptions' in config_class_name:
            if self.basic_action:
                return self.basic_action.get_action(ActionType.FIND)
        # Color-based find operations
        elif 'ColorFindOptions' in config_class_name:
            if self.basic_action:
                return self.basic_action.get_action(ActionType.FIND)
        # Click operations
        elif 'ClickOptions' in config_class_name:
            if self.basic_action:
                return self.basic_action.get_action(ActionType.CLICK)
        # Type operations
        elif 'TypeOptions' in config_class_name:
            if self.basic_action:
                return self.basic_action.get_action(ActionType.TYPE)
        # Mouse move operations
        elif 'MouseMoveOptions' in config_class_name:
            if self.basic_action:
                return self.basic_action.get_action(ActionType.MOVE)
        # Scroll operations
        elif 'ScrollOptions' in config_class_name:
            if self.basic_action:
                return self.basic_action.get_action(ActionType.SCROLL_MOUSE_WHEEL)
        # Drag operations
        elif 'DragOptions' in config_class_name:
            if self.basic_action:
                return self.basic_action.get_action(ActionType.DRAG)
        # Highlight operations
        elif 'HighlightOptions' in config_class_name:
            if self.basic_action:
                return self.basic_action.get_action(ActionType.HIGHLIGHT)
        # Vanish operations
        elif 'VanishOptions' in config_class_name:
            if self.basic_action:
                return self.basic_action.get_action(ActionType.WAIT_VANISH)
        
        return None


class BasicActionRegistry:
    """Registry of basic action implementations.
    
    Placeholder for BasicActionRegistry class.
    """
    
    def __init__(self):
        """Initialize the registry with action implementations."""
        self._actions: Dict[ActionType, ActionInterface] = {}
    
    def register(self, action_type: ActionType, action: ActionInterface) -> None:
        """Register an action implementation.
        
        Args:
            action_type: The type of action
            action: The action implementation
        """
        self._actions[action_type] = action
    
    def get_action(self, action_type: ActionType) -> Optional[ActionInterface]:
        """Get an action implementation by type.
        
        Args:
            action_type: The type of action to retrieve
            
        Returns:
            The action implementation or None if not found
        """
        return self._actions.get(action_type)


class FindStrategyRegistry:
    """Registry for find strategy implementations.
    
    Placeholder for FindStrategyRegistry class.
    """
    
    def __init__(self):
        """Initialize the registry."""
        self._custom_finds = []
    
    def add_custom_find(self, custom_find: Callable) -> None:
        """Add a custom find implementation.
        
        Args:
            custom_find: The custom find callable
        """
        self._custom_finds.append(custom_find)