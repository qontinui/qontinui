"""StateRegistry - Automatic state and transition discovery and management.

This module provides a registry that:
- Automatically discovers decorated states and transitions
- Manages profile-based state filtering
- Handles state and transition registration
- Provides lookup functionality with clear lifecycle (mutable → frozen)
"""

import importlib
import inspect
import logging
import pkgutil
import threading
from dataclasses import dataclass, field
from typing import Any

from qontinui.model.transition.enhanced_joint_table import StateTransitionsJointTable
from qontinui.model.transition.enhanced_state_transition import (
    CodeStateTransition,
    TaskSequenceStateTransition,
)
from qontinui.state_exceptions import StateAlreadyExistsException, StateNotFoundException

from .enhanced_state import get_state_metadata, is_state
from .transition_set import (
    TransitionSetMetadata,
    get_transition_metadata,
    get_transitions_from_state,
    get_transitions_to_state,
    is_transition_set,
)

logger = logging.getLogger(__name__)


class RegistryFrozenError(Exception):
    """Raised when attempting to modify a frozen registry."""

    def __init__(self, message: str = "Cannot modify frozen registry") -> None:
        """Initialize with message.

        Args:
            message: Error message
        """
        super().__init__(message)


@dataclass
class StateRegistry:
    """Automatic state and transition discovery and management.

    This is the Python equivalent of Spring's component scanning,
    providing automatic discovery and registration of decorated
    states and transitions.

    Lifecycle:
        1. Construction phase - states/transitions can be registered
        2. Call freeze() to lock the registry
        3. Frozen phase - no more registrations, only queries allowed

    Thread Safety:
        All operations are protected by an internal RLock (reentrant lock).
        Safe to use from multiple threads concurrently.

    Design Philosophy:
        - Clear lifecycle: mutable (construction) → frozen (runtime)
        - Raise exceptions instead of returning None for missing items
        - No Optional returns - explicit error handling
        - Immutable after freeze() - prevents runtime corruption
    """

    # Registered states by name
    states: dict[str, type] = field(default_factory=dict)

    # Registered transitions by name
    transitions: dict[str, type] = field(default_factory=dict)

    # State groups for collective management
    groups: dict[str, set[str]] = field(default_factory=dict)

    # States by profile
    profiles: dict[str, set[str]] = field(default_factory=dict)

    # State ID mapping (for runtime)
    state_ids: dict[str, int] = field(default_factory=dict)
    next_state_id: int = 1

    # Joint table for transitions
    joint_table: StateTransitionsJointTable = field(default_factory=StateTransitionsJointTable)

    # Current active profile
    active_profile: str = "default"

    # Frozen status
    _frozen: bool = field(default=False, init=False, repr=False)

    # Thread safety lock
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def freeze(self) -> None:
        """Freeze registry. No more registrations allowed.

        After calling this method, attempting to register states or transitions
        will raise RegistryFrozenError. This provides a clear lifecycle boundary
        between construction and runtime phases.

        Thread-safe: Can be called from any thread.
        """
        with self._lock:
            self._frozen = True
            logger.info("Registry frozen - no more registrations allowed")

    def is_frozen(self) -> bool:
        """Check if registry is frozen.

        Returns:
            True if frozen, False if still mutable
        """
        with self._lock:
            return self._frozen

    def auto_discover(self, modules: list[str]) -> None:
        """Scan modules for @state and @transition_set decorators.

        This is similar to Spring's component scanning - it automatically
        finds and registers all decorated classes.

        Args:
            modules: List of module names to scan (e.g., ["myapp.states"])

        Raises:
            RegistryFrozenError: If registry is frozen
        """
        with self._lock:
            if self._frozen:
                raise RegistryFrozenError("Cannot auto-discover on frozen registry")

        logger.info(f"Auto-discovering states and transitions in modules: {modules}")

        discovered_states = 0
        discovered_transitions = 0

        for module_name in modules:
            try:
                # Import the module
                module = importlib.import_module(module_name)

                # Scan the module and its submodules
                for _finder, name, _ispkg in pkgutil.walk_packages(
                    module.__path__, prefix=module_name + "."
                ):
                    try:
                        submodule = importlib.import_module(name)

                        # Scan for decorated classes
                        for _item_name, item in inspect.getmembers(submodule):
                            if inspect.isclass(item):
                                # Check for @state decorator
                                if is_state(item):
                                    self.register_state(item)
                                    discovered_states += 1

                                # Check for @transition_set decorator
                                if is_transition_set(item):
                                    self.register_transition(item)
                                    discovered_transitions += 1

                    except Exception as e:
                        logger.warning(f"Error scanning module {name}: {e}")

            except Exception as e:
                logger.error(f"Error importing module {module_name}: {e}")

        logger.info(
            f"Auto-discovery complete: "
            f"{discovered_states} states, {discovered_transitions} transitions"
        )

    def register_state(self, state_class: type) -> int:
        """Register a state class with the registry.

        Args:
            state_class: The state class to register

        Returns:
            Assigned state ID

        Raises:
            RegistryFrozenError: If registry is frozen
            StateAlreadyExistsException: If state name already registered
            ValueError: If state_class is not decorated with @state
        """
        metadata = get_state_metadata(state_class)
        if not metadata:
            raise ValueError(f"{state_class} is not decorated with @state")

        state_name = metadata.name

        # Entire check-then-act sequence is atomic
        with self._lock:
            if self._frozen:
                raise RegistryFrozenError(
                    f"Cannot register state '{state_name}' - registry is frozen"
                )

            # Check if already registered
            if state_name in self.states:
                # For idempotency, return existing ID if same class
                if self.states[state_name] is state_class:
                    logger.debug(f"State {state_name} already registered (same class)")
                    return self.state_ids[state_name]
                # Different class with same name - error
                raise StateAlreadyExistsException(
                    state_name,
                    existing_class=self.states[state_name],
                    new_class=state_class,
                )

            # Register the state
            self.states[state_name] = state_class

            # Assign ID
            state_id = self.next_state_id
            self.state_ids[state_name] = state_id
            self.next_state_id += 1

            # Register in group if specified
            if metadata.group:
                if metadata.group not in self.groups:
                    self.groups[metadata.group] = set()
                self.groups[metadata.group].add(state_name)

            # Register in profiles if initial state
            if metadata.initial:
                if metadata.profiles:
                    for profile in metadata.profiles:
                        if profile not in self.profiles:
                            self.profiles[profile] = set()
                        self.profiles[profile].add(state_name)
                else:
                    # Initial in all profiles
                    if "default" not in self.profiles:
                        self.profiles["default"] = set()
                    self.profiles["default"].add(state_name)

                # Add to joint table as initial state
                self.joint_table.add_initial_state(
                    state_id,
                    profile=metadata.profiles[0] if metadata.profiles else "default",
                )

            logger.info(
                f"Registered state: {state_name} (ID={state_id}, "
                f"group={metadata.group}, initial={metadata.initial})"
            )

            return state_id

    def register_transition(self, transition_class: type) -> None:
        """Register a transition class with the registry.

        Args:
            transition_class: The transition class to register

        Raises:
            RegistryFrozenError: If registry is frozen
            ValueError: If transition_class is not decorated with @transition_set
        """
        metadata = get_transition_metadata(transition_class)
        if not metadata:
            raise ValueError(f"{transition_class} is not decorated with @transition_set")

        transition_name = metadata.name

        # Entire check-then-act sequence is atomic
        with self._lock:
            if self._frozen:
                raise RegistryFrozenError(
                    f"Cannot register transition '{transition_name}' - registry is frozen"
                )

            # Check if already registered (idempotent)
            if transition_name in self.transitions:
                if self.transitions[transition_name] is transition_class:
                    logger.debug(f"Transition {transition_name} already registered")
                    return

            # Register the transition
            self.transitions[transition_name] = transition_class

            # Create StateTransition instances and add to joint table
            self._create_transition_instances(transition_class, metadata)

            logger.info(
                f"Registered transition: {transition_name} "
                f"({len(metadata.from_states)} -> {len(metadata.to_states)} states)"
            )

    def _create_transition_instances(
        self, transition_class: type, metadata: TransitionSetMetadata
    ) -> None:
        """Create StateTransition instances from decorated class.

        Args:
            transition_class: The transition class
            metadata: Transition metadata
        """
        # Get state IDs
        from_ids = set()
        for state_class in metadata.from_states:
            from_metadata = get_state_metadata(state_class)
            if from_metadata is not None:
                state_name = from_metadata.name
                if state_name in self.state_ids:
                    from_ids.add(self.state_ids[state_name])

        to_ids = set()
        for state_class in metadata.to_states:
            to_metadata = get_state_metadata(state_class)
            if to_metadata is not None:
                state_name = to_metadata.name
                if state_name in self.state_ids:
                    to_ids.add(self.state_ids[state_name])

        # Create transition instance
        transition: CodeStateTransition | TaskSequenceStateTransition
        if hasattr(transition_class, "execute"):
            # Code-based transition
            instance = transition_class()
            transition = CodeStateTransition(
                activate=to_ids if metadata.activate_all else set(),
                exit=from_ids if metadata.exit_all else set(),
                path_cost=metadata.path_cost,
                stays_visible=metadata.stays_visible,
                name=metadata.name,
                description=metadata.description,
                action=instance.execute if hasattr(instance, "execute") else None,
            )
        else:
            # Task sequence transition (placeholder)
            transition = TaskSequenceStateTransition(
                activate=to_ids if metadata.activate_all else set(),
                exit=from_ids if metadata.exit_all else set(),
                path_cost=metadata.path_cost,
                stays_visible=metadata.stays_visible,
                name=metadata.name,
                description=metadata.description,
            )

        # Add to joint table
        for from_id in from_ids:
            self.joint_table.add_transition(transition, from_id)

    def get_state(self, name: str) -> type:
        """Get a state class by name.

        Args:
            name: State name

        Returns:
            State class

        Raises:
            StateNotFoundException: If state doesn't exist
        """
        with self._lock:
            if name not in self.states:
                raise StateNotFoundException(name)
            return self.states[name]

    def get_state_id(self, name: str) -> int:
        """Get a state's ID by name.

        Args:
            name: State name

        Returns:
            State ID

        Raises:
            StateNotFoundException: If state doesn't exist
        """
        with self._lock:
            if name not in self.state_ids:
                raise StateNotFoundException(name)
            return self.state_ids[name]

    def get_state_by_id(self, state_id: int) -> type:
        """Get a state class by ID.

        Args:
            state_id: State ID

        Returns:
            State class

        Raises:
            StateNotFoundException: If state ID doesn't exist
        """
        with self._lock:
            for name, sid in self.state_ids.items():
                if sid == state_id:
                    return self.states[name]
            raise StateNotFoundException(f"state with ID {state_id}")

    def get_transition(self, name: str) -> type:
        """Get a transition class by name.

        Args:
            name: Transition name

        Returns:
            Transition class

        Raises:
            StateNotFoundException: If transition doesn't exist (using same exception for consistency)
        """
        with self._lock:
            if name not in self.transitions:
                raise StateNotFoundException(f"transition '{name}'")
            return self.transitions[name]

    def has_state(self, name: str) -> bool:
        """Check if a state exists.

        Args:
            name: State name

        Returns:
            True if state exists, False otherwise
        """
        with self._lock:
            return name in self.states

    def has_transition(self, name: str) -> bool:
        """Check if a transition exists.

        Args:
            name: Transition name

        Returns:
            True if transition exists, False otherwise
        """
        with self._lock:
            return name in self.transitions

    def get_initial_states(self, profile: str | None = None) -> list[type]:
        """Get initial states for a profile.

        Args:
            profile: Profile name (uses active_profile if None)

        Returns:
            List of initial state classes (empty list if none)
        """
        with self._lock:
            profile = profile or self.active_profile

            initial_names = self.profiles.get(profile, set())
            initial_states = []

            for name in initial_names:
                if name in self.states:
                    initial_states.append(self.states[name])

            # Sort by priority
            def get_priority(s):
                metadata = get_state_metadata(s)
                return metadata.priority if metadata is not None else 0

            initial_states.sort(key=get_priority, reverse=True)

            return initial_states

    def get_group_states(self, group_name: str) -> list[type]:
        """Get all states in a group.

        Args:
            group_name: Group name

        Returns:
            List of state classes in the group (empty list if group doesn't exist)
        """
        with self._lock:
            state_names = self.groups.get(group_name, set())
            states = []

            for name in state_names:
                if name in self.states:
                    states.append(self.states[name])

            return states

    def set_active_profile(self, profile: str) -> None:
        """Set the active profile for state filtering.

        Args:
            profile: Profile name
        """
        with self._lock:
            self.active_profile = profile
            logger.info(f"Active profile set to: {profile}")

    def get_transitions_for_state(self, state_class: type) -> tuple[list[type], list[type]]:
        """Get all transitions for a state.

        Args:
            state_class: The state class

        Returns:
            Tuple of (outgoing_transitions, incoming_transitions)
        """
        with self._lock:
            all_transitions = list(self.transitions.values())

            outgoing = get_transitions_from_state(all_transitions, state_class)
            incoming = get_transitions_to_state(all_transitions, state_class)

            return outgoing, incoming

    def get_statistics(self) -> dict[str, Any]:
        """Get registry statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            return {
                "total_states": len(self.states),
                "total_transitions": len(self.transitions),
                "total_groups": len(self.groups),
                "total_profiles": len(self.profiles),
                "active_profile": self.active_profile,
                "frozen": self._frozen,
                "joint_table_stats": self.joint_table.get_statistics(),
            }

    def clear(self) -> None:
        """Clear all registered states and transitions.

        Raises:
            RegistryFrozenError: If registry is frozen
        """
        with self._lock:
            if self._frozen:
                raise RegistryFrozenError("Cannot clear frozen registry")

            self.states.clear()
            self.transitions.clear()
            self.groups.clear()
            self.profiles.clear()
            self.state_ids.clear()
            self.next_state_id = 1
            self.joint_table.clear()
            logger.info("Registry cleared")

    def __str__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        frozen_str = " [FROZEN]" if stats["frozen"] else ""
        return (
            f"StateRegistry("
            f"states={stats['total_states']}, "
            f"transitions={stats['total_transitions']}, "
            f"groups={stats['total_groups']}, "
            f"profile={self.active_profile}{frozen_str})"
        )
