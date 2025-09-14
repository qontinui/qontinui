"""Action class - ported from Qontinui framework.

Entry point for executing GUI automation actions.
"""

from typing import Optional, List
from .action_interface import ActionInterface
from .action_config import ActionConfig
from .action_result import ActionResult
from .object_collection import ObjectCollection
from .action_execution import ActionExecution
from .action_service import ActionService


class Action:
    """Entry point for executing GUI automation actions in the Qontinui model-based framework.
    
    Port of Action from Qontinui framework class.
    
    The Action class serves as the central dispatcher for all GUI operations, implementing 
    the Action Model (α) described in the theoretical foundations. It processes 
    ActionConfig to determine what operation to perform and delegates execution to the 
    appropriate action implementation using the action function f_α.
    
    Key responsibilities:
    - Parse ActionConfig to identify the requested action type
    - Route execution to Basic or Composite action implementations
    - Manage the action lifecycle and error handling
    - Return comprehensive results via ActionResult objects
    
    Action types supported:
    - Basic Actions: Atomic operations like Find, Click, Type, Drag
    - Composite Actions: Complex operations that combine multiple basic actions
    
    This class abstracts the complexity of GUI interaction, allowing automation code to 
    focus on what to do rather than how to do it. The model-based approach ensures actions 
    are executed in the context of the current State, making them more reliable and robust.
    """
    
    def __init__(self,
                 action_execution: Optional[ActionExecution] = None,
                 action_service: Optional[ActionService] = None,
                 action_chain_executor: Optional['ActionChainExecutor'] = None):
        """Construct an Action instance with required dependencies.

        Uses dependency injection to wire the action execution engine, service
        layer, and chain executor. The ActionExecution handles the lifecycle management,
        the ActionService provides the registry of available actions, and the
        ActionChainExecutor handles chained action sequences.

        If dependencies are not provided, creates default instances to ensure
        all actions go through proper lifecycle management.

        Args:
            action_execution: Handles action lifecycle, timing, and cross-cutting concerns
            action_service: Provides access to registered action implementations
            action_chain_executor: Handles execution of chained action sequences
        """
        # Use provided instances or create defaults
        self.action_execution = action_execution or ActionExecution()
        self.action_service = action_service or ActionService()
        self.action_chain_executor = action_chain_executor
    
    def perform(self, 
                action_config: ActionConfig, 
                *object_collections: ObjectCollection) -> ActionResult:
        """Execute a GUI automation action with the specified configuration and target objects.
        
        This method uses the ActionConfig approach for type-safe action configuration.
        
        Args:
            action_config: Configuration specifying the action type and parameters
            object_collections: Target GUI elements to act upon (images, regions, locations, etc.)
            
        Returns:
            An ActionResult containing all results from the action execution
        """
        return self.perform_with_description("", action_config, *object_collections)
    
    def perform_with_description(self,
                                 action_description: str,
                                 action_config: ActionConfig,
                                 *object_collections: ObjectCollection) -> ActionResult:
        """Execute a GUI automation action with a descriptive label using ActionConfig.
        
        This method uses the ActionConfig approach for type-safe action configuration,
        while still providing human-readable descriptions for debugging and logging.
        
        Args:
            action_description: Human-readable description of what this action accomplishes
            action_config: Configuration specifying the action type and parameters
            object_collections: Target GUI elements to act upon
            
        Returns:
            An ActionResult containing all results from the action execution
        """
        # Reset times acted on for all objects
        for obj_coll in object_collections:
            obj_coll.reset_times_acted_on()
        
        # Check if this config has subsequent actions chained
        subsequent_actions = action_config.get_subsequent_actions()
        if subsequent_actions:
            # Execute the chain
            if self.action_chain_executor:
                return self.action_chain_executor.execute_chain(
                    action_config, ActionResult(), object_collections
                )
            else:
                print(f"Warning: Action chain executor not available for chained actions")
                return ActionResult()
        
        # Single action execution
        action = self.action_service.get_action(action_config)
        if action is None:
            print(f"Not a valid Action for {action_config.__class__.__name__}")
            return ActionResult()

        # Always use action execution for lifecycle management
        return self.action_execution.perform(
            action, action_description, action_config, object_collections
        )
    
    def find(self, *state_images: 'StateImage') -> ActionResult:
        """Perform a Find action with default options on the specified images.

        This convenience method simplifies the common case of searching for images
        on screen. The images are automatically wrapped in an ObjectCollection and
        searched using default Find parameters.

        Args:
            state_images: Variable number of StateImage objects to search for

        Returns:
            ActionResult containing found matches
        """
        from .object_collection import ObjectCollectionBuilder
        from .basic.find.pattern_find_options import PatternFindOptionsBuilder

        collection = ObjectCollectionBuilder().with_images(*state_images).build()
        config = PatternFindOptionsBuilder().build()
        return self.perform(config, collection)
    
    def click(self, *targets) -> ActionResult:
        """Perform a Click action with default options on the specified targets.

        This convenience method chains Find and Click for StateImages, or directly
        clicks for Locations and Regions.

        For StateImages: Chains Find -> Click (composite behavior)
        For Locations/Regions: Direct atomic Click

        Args:
            targets: Variable number of targets (StateImages, Regions, Locations)

        Returns:
            ActionResult indicating click success/failure
        """
        from .object_collection import ObjectCollectionBuilder
        from .basic.click.click_options import ClickOptionsBuilder
        from .basic.find.pattern_find_options import PatternFindOptionsBuilder

        # Separate targets by type
        state_images = []
        locations = []
        regions = []

        for target in targets:
            if hasattr(target, '__class__'):
                if target.__class__.__name__ == 'StateImage':
                    state_images.append(target)
                elif target.__class__.__name__ == 'Region':
                    regions.append(target)
                elif target.__class__.__name__ == 'Location':
                    locations.append(target)

        # If we have StateImages, we need to chain Find -> Click
        if state_images:
            # First, find the images
            image_collection = ObjectCollectionBuilder().with_images(*state_images).build()
            find_config = PatternFindOptionsBuilder().build()
            find_result = self.perform(find_config, image_collection)

            if not find_result.success or not find_result.match_list:
                # Find failed, return the find result
                return find_result

            # Now click on the found matches
            # Build collection with matches and any direct locations/regions
            click_builder = ObjectCollectionBuilder()
            click_builder.with_matches(*find_result.match_list)
            if locations:
                click_builder.with_locations(*locations)
            if regions:
                click_builder.with_regions(*regions)

            click_collection = click_builder.build()
            click_config = ClickOptionsBuilder().build()
            return self.perform(click_config, click_collection)
        else:
            # No StateImages, just direct click on locations/regions
            builder = ObjectCollectionBuilder()
            if locations:
                builder.with_locations(*locations)
            if regions:
                builder.with_regions(*regions)

            collection = builder.build()
            config = ClickOptionsBuilder().build()
            return self.perform(config, collection)
    
    def type_text(self, text: str) -> ActionResult:
        """Type the specified text using keyboard input.
        
        This convenience method simplifies text input operations.
        
        Args:
            text: Text to type
            
        Returns:
            ActionResult indicating typing success/failure
        """
        from .object_collection import ObjectCollectionBuilder
        
        collection = ObjectCollectionBuilder().with_strings(text).build()
        # Would need TypeOptions to be implemented
        # config = TypeOptions()
        # return self.perform(config, collection)
        return ActionResult()  # Placeholder


# Forward reference for dependencies not yet implemented
class ActionChainExecutor:
    """Placeholder for ActionChainExecutor class."""
    def execute_chain(self, config: ActionConfig, result: ActionResult,
                     collections: tuple) -> ActionResult:
        return result