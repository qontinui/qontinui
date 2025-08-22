"""Action step model - ported from Qontinui framework.

Represents a single automation action in a task sequence.
"""

from typing import Optional
from dataclasses import dataclass


@dataclass
class ActionStep:
    """Represents a single automation action.
    
    Port of ActionStep from Qontinui framework class.
    
    ActionStep combines an action configuration (ActionOptions) with the
    target objects (ObjectCollection) to perform a specific automation task.
    This is the fundamental unit of execution in the DSL.
    
    Example in JSON:
        {
            "actionOptions": {
                "action": "CLICK",
                "pauseAfter": 0.5,
                "similarity": 0.95
            },
            "objectCollection": {
                "stateImages": [{"name": "submitButton"}],
                "searchRegions": [{"x": 100, "y": 100, "width": 200, "height": 50}]
            }
        }
    """
    
    action_options: Optional['ActionOptions'] = None
    """Configuration for the action including type, timing, and behavior."""
    
    object_collection: Optional['ObjectCollection'] = None
    """Target objects for the action (images, regions, strings, etc.)."""
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ActionStep':
        """Create ActionStep from dictionary.
        
        Args:
            data: Dictionary with action step data
            
        Returns:
            ActionStep instance
        """
        action_options = None
        if 'actionOptions' in data:
            # Would need actual ActionOptions implementation
            action_options = ActionOptions.from_dict(data['actionOptions'])
        
        object_collection = None
        if 'objectCollection' in data:
            # Would need actual ObjectCollection implementation
            object_collection = ObjectCollection.from_dict(data['objectCollection'])
        
        return cls(
            action_options=action_options,
            object_collection=object_collection
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        result = {}
        if self.action_options:
            result['actionOptions'] = self.action_options.to_dict()
        if self.object_collection:
            result['objectCollection'] = self.object_collection.to_dict()
        return result


# Placeholder classes - these would come from the actions package
class ActionOptions:
    """Placeholder for ActionOptions class."""
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ActionOptions':
        """Create from dictionary."""
        instance = cls()
        instance.__dict__.update(data)
        return instance
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self.__dict__.copy()


class ObjectCollection:
    """Placeholder for ObjectCollection class."""
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ObjectCollection':
        """Create from dictionary."""
        instance = cls()
        instance.__dict__.update(data)
        return instance
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self.__dict__.copy()