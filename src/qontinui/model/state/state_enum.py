"""StateEnum - ported from Qontinui framework.

Enumeration for state names to ensure consistency.
"""

from enum import Enum, auto


class StateEnum(Enum):
    """Base class for state enumerations.

    Port of StateEnum from Qontinui framework.
    Applications should extend this to define their states.
    """

    # Common states that most applications have
    UNKNOWN = auto()
    LOADING = auto()
    MAIN_MENU = auto()
    ERROR = auto()

    @classmethod
    def from_string(cls, name: str) -> "StateEnum":
        """Get enum value from string name.

        Args:
            name: State name string

        Returns:
            Corresponding enum value

        Raises:
            ValueError: If name not found
        """
        for state in cls:
            if state.name == name.upper():
                return state
        raise ValueError(f"No state enum with name: {name}")

    def get_name(self) -> str:
        """Get the state name as string.

        Returns:
            State name
        """
        return self.name
