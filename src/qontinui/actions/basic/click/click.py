"""Click action - ported from Qontinui framework.

Performs mouse click operations on GUI elements.
"""

from typing import Optional
from ...action_interface import ActionInterface
from ...action_type import ActionType
from ...action_result import ActionResult
from ...object_collection import ObjectCollection
from ..find.find import Find
from ..find.pattern_find_options import PatternFindOptions, PatternFindOptionsBuilder
from .click_options import ClickOptions
from ....model.location import Location
from ....model.match import Match


class Click(ActionInterface):
    """Performs mouse click operations on GUI elements in the Qontinui framework.
    
    Port of Click from Qontinui framework class.
    
    Click is one of the most fundamental actions in GUI automation, enabling interaction 
    with buttons, links, menus, and other clickable elements. It combines the visual 
    recognition capabilities of Find with precise mouse control to reliably interact with 
    GUI elements across different applications and platforms.
    
    Click targets supported:
    - Image Matches: Clicks on visually identified elements
    - Regions: Clicks within defined screen areas
    - Locations: Clicks at specific screen coordinates
    - Previous Matches: Reuses results from earlier Find operations
    
    Advanced features:
    - Multi-click support for double-clicks, triple-clicks, etc.
    - Configurable click types (left, right, middle button)
    - Batch clicking on multiple matches
    - Post-click mouse movement to avoid hover effects
    - Precise timing control between clicks
    - Integration with state management for context-aware clicking
    
    In the model-based approach, Click actions are more than simple mouse events. They 
    update the framework's understanding of the GUI state, track interaction history, and 
    can trigger state transitions. This integration ensures that the automation maintains 
    an accurate model of the application state throughout execution.
    """
    
    def __init__(self, 
                 find: Optional[Find] = None,
                 click_location_once: Optional['SingleClickExecutor'] = None,
                 time: Optional['TimeProvider'] = None,
                 after_click: Optional['PostClickHandler'] = None,
                 action_result_factory: Optional['ActionResultFactory'] = None):
        """Initialize Click action.
        
        Args:
            find: Find action for locating targets
            click_location_once: Executor for single clicks
            time: Time provider for delays
            after_click: Handler for post-click actions
            action_result_factory: Factory for creating action results
        """
        self.find = find
        self.click_location_once = click_location_once
        self.time = time
        self.after_click = after_click
        self.action_result_factory = action_result_factory
    
    def get_action_type(self) -> ActionType:
        """Return the action type.
        
        Returns:
            ActionType.CLICK
        """
        return ActionType.CLICK
    
    def perform(self, matches: ActionResult, *object_collections: ObjectCollection) -> None:
        """Execute click operations on all found matches up to the maximum allowed.
        
        This method orchestrates the complete click process:
        1. Uses Find to locate target elements in the first ObjectCollection
        2. Iterates through matches up to configured maximum
        3. Performs configured number of clicks on each match
        4. Applies pauses between clicking different matches
        
        The method modifies the Match objects by incrementing their "times acted on" counter,
        which tracks interactions during this action execution.
        
        Args:
            matches: The ActionResult containing configuration and to which found matches
                    are added. The matches collection is populated by the Find operation.
            object_collections: The collections containing objects to find and click. Only
                              the first collection is used for finding targets.
                              
        Raises:
            ValueError: If matches does not contain ClickOptions configuration
        """
        # Get the configuration - expecting ClickOptions
        action_config = matches.get_action_config()
        if not isinstance(action_config, ClickOptions):
            raise ValueError("Click requires ClickOptions configuration")
        
        click_options = action_config
        
        # Create a separate ActionResult for Find with PatternFindOptions
        # This is necessary because Find expects BaseFindOptions, not ClickOptions
        find_options = PatternFindOptionsBuilder().build()
        
        if self.action_result_factory:
            find_result = self.action_result_factory.init(find_options, "Click->Find", object_collections)
        else:
            # Create a basic ActionResult if factory not available
            find_result = ActionResult(find_options)
            find_result.action_description = "Click->Find"
        
        # Perform find operation
        if self.find:
            self.find.perform(find_result, *object_collections)  # find performs only on 1st collection
        
        # Copy find results back to the original matches
        matches.match_list.extend(find_result.match_list)
        matches.set_success(find_result.is_success())
        
        # Find should have already limited matches based on maxMatchesToActOn
        match_index = 0
        for match in matches.match_list:
            location = match.get_target()
            self._click(location, click_options, match)
            match_index += 1
            # pause only between clicking different matches, not after the last match
            if match_index < len(matches.match_list):
                if self.time:
                    self.time.wait(click_options.get_pause_between_individual_actions())
    
    def _click(self, location: Location, click_options: ClickOptions, match: Match) -> None:
        """Perform multiple clicks on a single location with configurable timing and post-click behavior.
        
        This method handles the low-level click execution for a single match, including:
        - Repeating clicks based on configuration
        - Tracking click count on the match object
        - Managing timing between repeated clicks
        - Delegating post-click mouse movement to AfterClick when configured
        
        Example: With timesToRepeatIndividualAction=2, this method will:
        1. Click the location
        2. Increment match's acted-on counter
        3. Move mouse if configured
        4. Pause
        5. Click again
        6. Increment counter again
        7. Move mouse if configured (no pause after last click)
        
        Args:
            location: The screen coordinates where clicks will be performed. This is the
                     final, adjusted location from the match's target.
            click_options: Configuration containing:
                          - timesToRepeatIndividualAction: number of clicks per location
                          - pauseBetweenIndividualActions: delay between clicks (ms)
                          - moveMouseAfterAction: whether to move mouse after clicking
            match: The Match object being acted upon. This object is modified by
                  incrementing its timesActedOn counter for each click.
        """
        times_to_repeat = click_options.get_times_to_repeat_individual_action()
        for i in range(times_to_repeat):
            # SingleClickExecutor now accepts ActionConfig
            if self.click_location_once:
                self.click_location_once.click(location, click_options)
            match.increment_times_acted_on()
            # TODO: Handle mouse movement after action when PostClickHandler is implemented
            if i < times_to_repeat - 1:
                if self.time:
                    self.time.wait(click_options.get_pause_between_individual_actions())


class SingleClickExecutor:
    """Placeholder for SingleClickExecutor class.
    
    Executes a single click at a location.
    """
    
    def click(self, location: Location, click_options: ClickOptions) -> None:
        """Execute a single click.
        
        Args:
            location: Location to click
            click_options: Click configuration
        """
        print(f"Clicking at {location} with options {click_options}")


class PostClickHandler:
    """Placeholder for PostClickHandler class.
    
    Handles post-click mouse movement.
    """
    pass


class TimeProvider:
    """Placeholder for TimeProvider class.
    
    Provides time and delay functionality.
    """
    
    def wait(self, seconds: float) -> None:
        """Wait for specified duration.
        
        Args:
            seconds: Duration to wait
        """
        import time
        time.sleep(seconds)


class ActionResultFactory:
    """Placeholder for ActionResultFactory class.
    
    Creates ActionResult instances.
    """
    
    def init(self, action_config, description: str, object_collections: tuple) -> ActionResult:
        """Initialize an ActionResult.
        
        Args:
            action_config: Configuration for the action
            description: Description of the action
            object_collections: Object collections
            
        Returns:
            New ActionResult instance
        """
        result = ActionResult(action_config)
        result.action_description = description
        return result