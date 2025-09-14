"""State component extractor - ported from Qontinui framework.

Extracts StateImage, StateString, and StateObject components from annotated classes.
"""

from dataclasses import dataclass, field
from typing import List, Any
import logging
import inspect
from ..model.state.state_image import StateImage
from ..model.state.state_string import StateString
from ..model.state.state_object import StateObject
from ..model.state.state_location import StateLocation
from ..model.state.state_region import StateRegion

logger = logging.getLogger(__name__)


@dataclass
class StateComponents:
    """Container for extracted state components.
    
    Groups the different types of components extracted from a state class.
    """
    state_images: List[StateImage] = field(default_factory=list)
    state_strings: List[StateString] = field(default_factory=list)
    state_objects: List[StateObject] = field(default_factory=list)
    
    def get_total_components(self) -> int:
        """Get total number of components.
        
        Returns:
            Total component count
        """
        return len(self.state_images) + len(self.state_strings) + len(self.state_objects)


class StateComponentExtractor:
    """Extracts state components from annotated state classes.
    
    Port of StateComponentExtractor from Qontinui framework.
    
    This class is responsible for extracting StateImage, StateString,
    and StateObject components from @state annotated classes through
    reflection.
    
    The extractor:
    - Scans all fields in the state class
    - Identifies Qontinui state components by type
    - Extracts and collects them for state building
    - Handles nested StateObject structures
    """
    
    def extract_components(self, state_instance: Any) -> StateComponents:
        """Extract all state components from a state instance.
        
        Args:
            state_instance: Instance of a @state annotated class
            
        Returns:
            Container with all extracted components
        """
        components = StateComponents()
        state_class = state_instance.__class__
        
        logger.debug(f"Extracting components from state class: {state_class.__name__}")
        
        # Get all attributes of the instance
        for attr_name in dir(state_instance):
            # Skip private and magic attributes
            if attr_name.startswith('_'):
                continue
            
            try:
                attr_value = getattr(state_instance, attr_name)
                
                # Skip methods and non-state components
                if callable(attr_value):
                    continue
                
                # Check for state component types
                if isinstance(attr_value, StateImage):
                    components.state_images.append(attr_value)
                    logger.trace(f"Found StateImage: {attr_name}")
                    
                elif isinstance(attr_value, StateString):
                    components.state_strings.append(attr_value)
                    logger.trace(f"Found StateString: {attr_name}")
                    
                elif isinstance(attr_value, StateObject):
                    components.state_objects.append(attr_value)
                    logger.trace(f"Found StateObject: {attr_name} ({attr_value.__class__.__name__})")
                    
                    # Also extract nested components from StateObject
                    nested = self._extract_nested_components(attr_value)
                    components.state_images.extend(nested.state_images)
                    components.state_strings.extend(nested.state_strings)
                    
                # Check for lists of state components
                elif isinstance(attr_value, list):
                    self._extract_from_list(attr_value, components)
                    
            except Exception as e:
                logger.warning(f"Error extracting attribute {attr_name}: {e}")
        
        logger.info(
            f"Extracted {len(components.state_images)} StateImages, "
            f"{len(components.state_strings)} StateStrings, "
            f"{len(components.state_objects)} StateObjects from {state_class.__name__}"
        )
        
        return components
    
    def _extract_nested_components(self, state_object: StateObject) -> StateComponents:
        """Extract nested components from a StateObject.
        
        StateObjects can contain other state components like StateImages.
        
        Args:
            state_object: StateObject to extract from
            
        Returns:
            Extracted nested components
        """
        nested = StateComponents()
        
        # StateObject might have images, strings, etc.
        # This depends on the specific StateObject implementation
        if hasattr(state_object, 'state_images'):
            images = getattr(state_object, 'state_images', [])
            if images:
                nested.state_images.extend(images)
                logger.trace(f"Found {len(images)} nested StateImages in StateObject")
        
        if hasattr(state_object, 'state_strings'):
            strings = getattr(state_object, 'state_strings', [])
            if strings:
                nested.state_strings.extend(strings)
                logger.trace(f"Found {len(strings)} nested StateStrings in StateObject")
        
        return nested
    
    def _extract_from_list(self, item_list: List[Any], components: StateComponents) -> None:
        """Extract state components from a list.
        
        Args:
            item_list: List that might contain state components
            components: Container to add extracted components to
        """
        for item in item_list:
            if isinstance(item, StateImage):
                components.state_images.append(item)
            elif isinstance(item, StateString):
                components.state_strings.append(item)
            elif isinstance(item, StateObject):
                components.state_objects.append(item)
                # Also extract nested components
                nested = self._extract_nested_components(item)
                components.state_images.extend(nested.state_images)
                components.state_strings.extend(nested.state_strings)