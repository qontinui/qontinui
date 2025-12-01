"""Action interface - ported from Qontinui framework.

Core interface for all actions.
"""

from abc import abstractmethod
from typing import Protocol

from .action_result import ActionResult
from .action_type import ActionType
from .object_collection import ObjectCollection


class ActionInterface(Protocol):
    """Core interface for all actions in the Qontinui model-based GUI automation framework.

    Port of ActionInterface from Qontinui framework.

    ActionInterface defines the contract that all action implementations must follow,
    establishing a uniform execution pattern across the entire Action Model (α). This
    interface is the foundation of Qontinui's action architecture, enabling polymorphic
    dispatch of diverse GUI operations through a single, consistent API.

    Key design principles:
    - Uniform Execution: All actions, from simple clicks to complex workflows,
      implement the same perform() method signature
    - Result Accumulation: Actions modify the provided ActionResult object to record
      their results and maintain execution context
    - Flexible Input: Accepts variable ObjectCollections to support actions
      requiring different numbers of targets
    - Composability: Enables actions to be combined into composite operations

    The perform method contract:
    - Receives an ActionResult object containing ActionOptions and accumulating results
    - Processes one or more ObjectCollections containing the action targets
    - Updates the ActionResult object with results of the action
    - May throw runtime exceptions for error conditions

    Implementation categories:
    - Basic Actions: Click, Type, Find, Drag, etc.
    - Composite Actions: Multi-step operations built from basic actions
    - Custom Actions: Application-specific operations
    - Mock Actions: Test implementations for development and testing

    In the model-based approach, ActionInterface enables the framework to treat all
    GUI operations uniformly, regardless of their complexity. This abstraction is crucial
    for building maintainable automation scripts where actions can be easily substituted,
    extended, or composed without changing the calling code.
    """

    @abstractmethod
    def get_action_type(self) -> ActionType:
        """Return the standard type of this action implementation.

        Returns:
            The ActionType enum value
        """
        ...

    @abstractmethod
    def perform(
        self, matches: ActionResult, *object_collections: ObjectCollection
    ) -> None:
        """Execute the action with the provided configuration and target objects.

        This method is the core execution point for all GUI automation actions in Qontinui.
        Implementations should follow these guidelines:
        - Read action configuration from the ActionOptions within matches
        - Process target objects from the ObjectCollections
        - Execute the GUI operation (click, type, find, etc.)
        - Update the matches object with results (found elements, success status)
        - Handle errors gracefully, updating matches with failure information

        Side effects:
        - Modifies the GUI state through mouse/keyboard operations
        - Updates the matches parameter with execution results
        - May capture screenshots or log execution details

        Implementation note: The matches parameter serves dual purposes -
        it provides input configuration via ActionOptions and accumulates output results.
        This design enables action chaining and comprehensive result tracking.

        Args:
            matches: Contains ActionOptions for configuration and accumulates execution results.
                    This object is modified during execution to record matches found,
                    success/failure status, and timing information.
            object_collections: Variable number of collections containing target objects
                              (StateImages, Regions, Locations, Strings) that the action
                              will operate on. Actions may use zero, one, or multiple collections.
        """
        ...
