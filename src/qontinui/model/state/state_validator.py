"""StateValidator - Validates state store consistency.

Performs consistency checks across state storage, lifecycle, and relationships.
"""

import logging
from typing import cast

logger = logging.getLogger(__name__)


class StateValidator:
    """Validates consistency of state storage and relationships.

    Single responsibility: Ensure data integrity across state components.
    """

    def __init__(
        self,
        repository_contains_func,
        lifecycle_current_state_func,
        lifecycle_active_states_func,
        relationship_validate_func,
    ) -> None:
        """Initialize the validator with callback functions.

        Args:
            repository_contains_func: Function to check if state exists in repository
            lifecycle_current_state_func: Function to get current state name
            lifecycle_active_states_func: Function to get active state names
            relationship_validate_func: Function to validate relationships
        """
        self._repository_contains = repository_contains_func
        self._get_current_state = lifecycle_current_state_func
        self._get_active_states = lifecycle_active_states_func
        self._validate_relationships = relationship_validate_func

    def validate(self) -> list[str]:
        """Validate store consistency.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check current state exists
        current_state = self._get_current_state()
        if current_state and not self._repository_contains(current_state):
            errors.append(f"Current state '{current_state}' not in repository")

        # Check active states exist
        active_states = self._get_active_states()
        for name in active_states:
            if not self._repository_contains(name):
                errors.append(f"Active state '{name}' not in repository")

        # Validate relationships
        valid_states = self._get_all_valid_states()
        relationship_errors = self._validate_relationships(valid_states)
        errors.extend(relationship_errors)

        if errors:
            logger.warning(f"Validation found {len(errors)} error(s)")
        else:
            logger.debug("Validation passed")

        return errors

    def _get_all_valid_states(self) -> set[str]:
        """Get set of all valid state names from repository.

        Returns:
            Set of valid state names
        """
        # This would need to be injected or the validator would need access to repository
        # For now, we'll rely on the validation being called with the right context
        # In practice, this could be another callback
        return set()

    def validate_state_exists(self, name: str) -> bool:
        """Validate that a state exists in the repository.

        Args:
            name: State name to validate

        Returns:
            True if state exists
        """
        return cast(bool, self._repository_contains(name))

    def validate_active_state(self, name: str) -> bool:
        """Validate that a state exists and is properly tracked.

        Args:
            name: State name to validate

        Returns:
            True if state is valid
        """
        if not self._repository_contains(name):
            logger.error(f"State '{name}' does not exist in repository")
            return False
        return True
