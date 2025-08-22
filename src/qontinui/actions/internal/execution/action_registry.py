"""ActionRegistry - ported from Qontinui framework.

Registry for action types and factories.
"""

from typing import Dict, Type, Optional, List, Callable, Any
from dataclasses import dataclass
import threading
import logging
from ....action_interface import ActionInterface
from ....action_config import ActionConfig


logger = logging.getLogger(__name__)


@dataclass
class ActionMetadata:
    """Metadata for registered action."""
    
    action_class: Type[ActionInterface]
    name: str
    category: str
    description: str
    version: str = "1.0.0"
    tags: List[str] = None
    factory: Optional[Callable[..., ActionInterface]] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ActionRegistry:
    """Registry for action types and factories.
    
    Port of ActionRegistry from Qontinui framework.
    
    Provides centralized registration and discovery of action types with:
    - Action registration by name/type
    - Factory pattern support
    - Category-based organization
    - Tag-based discovery
    - Version management
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for registry."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize registry."""
        if not hasattr(self, '_initialized'):
            self._actions: Dict[str, ActionMetadata] = {}
            self._categories: Dict[str, List[str]] = {}
            self._tags: Dict[str, List[str]] = {}
            self._factories: Dict[str, Callable] = {}
            self._initialized = True
            self._register_builtin_actions()
            logger.info("ActionRegistry initialized")
    
    def _register_builtin_actions(self):
        """Register built-in action types."""
        # Register basic actions
        try:
            from ...basic.click.click import Click, ClickOptions
            self.register(
                "click",
                Click,
                category="basic",
                description="Mouse click action",
                tags=["mouse", "input", "basic"]
            )
        except ImportError:
            pass
        
        try:
            from ...basic.type.type_action import TypeAction, TypeOptions
            self.register(
                "type",
                TypeAction,
                category="basic",
                description="Text typing action",
                tags=["keyboard", "input", "text", "basic"]
            )
        except ImportError:
            pass
        
        try:
            from ...basic.wait.wait import Wait, WaitOptions
            self.register(
                "wait",
                Wait,
                category="basic",
                description="Wait action",
                tags=["timing", "synchronization", "basic"]
            )
        except ImportError:
            pass
        
        # Register composite actions
        try:
            from ...composite.drag.drag import Drag, DragOptions
            self.register(
                "drag",
                Drag,
                category="composite",
                description="Drag operation",
                tags=["mouse", "composite", "gesture"]
            )
        except ImportError:
            pass
        
        try:
            from ...composite.chains.action_chain import ActionChain
            self.register(
                "chain",
                ActionChain,
                category="composite",
                description="Sequential action chain",
                tags=["composite", "sequence", "workflow"]
            )
        except ImportError:
            pass
        
        try:
            from ...composite.multiple.multiple_actions import MultipleActions
            self.register(
                "multiple",
                MultipleActions,
                category="composite",
                description="Parallel action execution",
                tags=["composite", "parallel", "concurrent"]
            )
        except ImportError:
            pass
    
    def register(self, name: str, 
                action_class: Type[ActionInterface],
                category: str = "custom",
                description: str = "",
                version: str = "1.0.0",
                tags: Optional[List[str]] = None,
                factory: Optional[Callable[..., ActionInterface]] = None) -> bool:
        """Register an action type.
        
        Args:
            name: Unique name for action
            action_class: Action class type
            category: Action category
            description: Action description
            version: Action version
            tags: Tags for discovery
            factory: Optional factory function
            
        Returns:
            True if registered successfully
        """
        with self._lock:
            if name in self._actions:
                logger.warning(f"Action '{name}' already registered")
                return False
            
            metadata = ActionMetadata(
                action_class=action_class,
                name=name,
                category=category,
                description=description,
                version=version,
                tags=tags or [],
                factory=factory
            )
            
            self._actions[name] = metadata
            
            # Update category index
            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(name)
            
            # Update tag index
            for tag in metadata.tags:
                if tag not in self._tags:
                    self._tags[tag] = []
                self._tags[tag].append(name)
            
            # Register factory if provided
            if factory:
                self._factories[name] = factory
            
            logger.debug(f"Registered action '{name}' in category '{category}'")
            return True
    
    def unregister(self, name: str) -> bool:
        """Unregister an action type.
        
        Args:
            name: Action name
            
        Returns:
            True if unregistered successfully
        """
        with self._lock:
            if name not in self._actions:
                return False
            
            metadata = self._actions[name]
            
            # Remove from category index
            if metadata.category in self._categories:
                self._categories[metadata.category].remove(name)
                if not self._categories[metadata.category]:
                    del self._categories[metadata.category]
            
            # Remove from tag index
            for tag in metadata.tags:
                if tag in self._tags:
                    self._tags[tag].remove(name)
                    if not self._tags[tag]:
                        del self._tags[tag]
            
            # Remove factory
            if name in self._factories:
                del self._factories[name]
            
            # Remove metadata
            del self._actions[name]
            
            logger.debug(f"Unregistered action '{name}'")
            return True
    
    def get(self, name: str) -> Optional[ActionMetadata]:
        """Get action metadata by name.
        
        Args:
            name: Action name
            
        Returns:
            Action metadata or None
        """
        return self._actions.get(name)
    
    def create(self, name: str, **kwargs) -> Optional[ActionInterface]:
        """Create action instance by name.
        
        Args:
            name: Action name
            **kwargs: Arguments for action creation
            
        Returns:
            Action instance or None
        """
        metadata = self.get(name)
        if not metadata:
            logger.warning(f"Action '{name}' not found")
            return None
        
        try:
            # Use factory if available
            if metadata.factory:
                return metadata.factory(**kwargs)
            
            # Otherwise create directly
            return metadata.action_class(**kwargs)
            
        except Exception as e:
            logger.error(f"Failed to create action '{name}': {e}")
            return None
    
    def list_all(self) -> List[str]:
        """List all registered action names.
        
        Returns:
            List of action names
        """
        return list(self._actions.keys())
    
    def list_by_category(self, category: str) -> List[str]:
        """List actions by category.
        
        Args:
            category: Category name
            
        Returns:
            List of action names in category
        """
        return self._categories.get(category, []).copy()
    
    def list_by_tag(self, tag: str) -> List[str]:
        """List actions by tag.
        
        Args:
            tag: Tag name
            
        Returns:
            List of action names with tag
        """
        return self._tags.get(tag, []).copy()
    
    def list_categories(self) -> List[str]:
        """List all categories.
        
        Returns:
            List of category names
        """
        return list(self._categories.keys())
    
    def list_tags(self) -> List[str]:
        """List all tags.
        
        Returns:
            List of tag names
        """
        return list(self._tags.keys())
    
    def search(self, query: str) -> List[str]:
        """Search for actions by name/description/tag.
        
        Args:
            query: Search query
            
        Returns:
            List of matching action names
        """
        query_lower = query.lower()
        matches = []
        
        for name, metadata in self._actions.items():
            # Check name
            if query_lower in name.lower():
                matches.append(name)
                continue
            
            # Check description
            if query_lower in metadata.description.lower():
                matches.append(name)
                continue
            
            # Check tags
            for tag in metadata.tags:
                if query_lower in tag.lower():
                    matches.append(name)
                    break
        
        return matches
    
    def register_factory(self, name: str, factory: Callable[..., ActionInterface]):
        """Register a factory function for an action.
        
        Args:
            name: Action name
            factory: Factory function
        """
        if name in self._actions:
            self._actions[name].factory = factory
            self._factories[name] = factory
            logger.debug(f"Registered factory for '{name}'")
    
    def get_metadata_list(self) -> List[ActionMetadata]:
        """Get all action metadata.
        
        Returns:
            List of all metadata
        """
        return list(self._actions.values())
    
    def clear(self):
        """Clear all registrations."""
        with self._lock:
            self._actions.clear()
            self._categories.clear()
            self._tags.clear()
            self._factories.clear()
            logger.info("Registry cleared")
    
    def reload_builtins(self):
        """Reload built-in action registrations."""
        # Remove existing built-ins
        builtins = ["click", "type", "wait", "drag", "chain", "multiple"]
        for name in builtins:
            self.unregister(name)
        
        # Re-register
        self._register_builtin_actions()
        logger.info("Built-in actions reloaded")
    
    @classmethod
    def get_instance(cls) -> 'ActionRegistry':
        """Get singleton instance.
        
        Returns:
            ActionRegistry instance
        """
        return cls()


# Global registry instance
registry = ActionRegistry.get_instance()