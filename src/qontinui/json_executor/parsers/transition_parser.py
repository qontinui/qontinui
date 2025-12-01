"""Parses transitions and assigns them to states."""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config_parser import (
        IncomingTransition,
        OutgoingTransition,
        QontinuiConfig,
        Transition,
    )

logger = logging.getLogger(__name__)


class TransitionParser:
    """Parses transitions and assigns them to their states.

    TransitionParser handles the logic of parsing transition data and properly
    assigning OutgoingTransition and IncomingTransition objects to their respective
    states in the state machine.

    Transition Assignment Rules:
    - OutgoingTransition: Assigned to from_state's outgoing_transitions list
    - IncomingTransition: Assigned to to_state's incoming_transitions list
    - Warns if transition references unknown states

    Example:
        >>> parser = TransitionParser()
        >>> parser.parse_and_assign_transitions(config, transitions_data)
        [DEBUG] Assigned 5 outgoing and 3 incoming transitions
    """

    def parse_and_assign_transitions(
        self, config: "QontinuiConfig", transitions_data: list[dict[str, Any]]
    ) -> None:
        """Parse transitions and assign them to states.

        Parses each transition from data, determines its type (Outgoing/Incoming),
        and assigns it to the appropriate state's transition list.

        Args:
            config: Configuration with state_map for assignment.
            transitions_data: List of transition dictionaries from JSON.

        Note:
            - Prints warnings for transitions referencing unknown states
            - Modifies state objects in config.state_map in-place
        """
        for trans_data in transitions_data:
            transition = self._parse_transition(trans_data)

            if isinstance(transition, self._get_outgoing_transition_class()):
                self._assign_outgoing_transition(config, transition)

            if isinstance(transition, self._get_incoming_transition_class()):
                self._assign_incoming_transition(config, transition)

    def _parse_transition(self, data: dict[str, Any]) -> "Transition":
        """Parse transition from dictionary using Pydantic validation.

        Infers transition type based on presence of fromState field.

        Args:
            data: Transition dictionary from JSON.

        Returns:
            OutgoingTransition or IncomingTransition object.

        Note:
            OutgoingTransition has fromState, IncomingTransition does not.
        """
        from ..config_parser import IncomingTransition, OutgoingTransition

        # Infer transition type based on presence of fromState
        transition_type = data.get("type")
        if transition_type is None:
            transition_type = (
                "OutgoingTransition" if "fromState" in data else "IncomingTransition"
            )

        # Use Pydantic validation
        if transition_type == "OutgoingTransition":
            return OutgoingTransition.model_validate(data)
        else:
            return IncomingTransition.model_validate(data)

    def _assign_outgoing_transition(
        self, config: "QontinuiConfig", transition: "OutgoingTransition"
    ) -> None:
        """Assign OutgoingTransition to its from_state.

        Args:
            config: Configuration with state_map.
            transition: OutgoingTransition to assign.
        """
        if transition.from_state in config.state_map:
            config.state_map[transition.from_state].outgoing_transitions.append(
                transition
            )
        else:
            logger.warning(
                f"Transition {transition.id} references unknown fromState: {transition.from_state}"
            )

    def _assign_incoming_transition(
        self, config: "QontinuiConfig", transition: "IncomingTransition"
    ) -> None:
        """Assign IncomingTransition to its to_state.

        Args:
            config: Configuration with state_map.
            transition: IncomingTransition to assign.
        """
        if transition.to_state in config.state_map:
            config.state_map[transition.to_state].incoming_transitions.append(
                transition
            )
        else:
            logger.warning(
                f"Transition {transition.id} references unknown toState: {transition.to_state}"
            )

    @staticmethod
    def _get_outgoing_transition_class():
        """Get OutgoingTransition class for isinstance checks."""
        from ..config_parser import OutgoingTransition

        return OutgoingTransition

    @staticmethod
    def _get_incoming_transition_class():
        """Get IncomingTransition class for isinstance checks."""
        from ..config_parser import IncomingTransition

        return IncomingTransition
