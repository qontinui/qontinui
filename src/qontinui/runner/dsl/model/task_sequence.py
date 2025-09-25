"""Task sequence model - ported from Qontinui framework.

Represents an ordered list of automation steps.
"""

from dataclasses import dataclass, field


@dataclass
class TaskSequence:
    """Represents an ordered list of automation steps.

    Port of TaskSequence from Qontinui framework class.

    TaskSequence is used to define a series of automation actions that should
    be executed in order. This is a core concept in the DSL that allows
    complex automation workflows to be defined declaratively.

    Built internally by ActionSequenceBuilder in the fluent API or
    defined directly in JSON.

    Example in JSON:
        {
            "steps": [
                {
                    "actionOptions": {"action": "FIND"},
                    "objectCollection": {"stateImages": [{"name": "loginButton"}]}
                },
                {
                    "actionOptions": {"action": "CLICK"},
                    "objectCollection": {"stateImages": [{"name": "loginButton"}]}
                }
            ]
        }
    """

    steps: list["ActionStep"] = field(default_factory=list)
    """Ordered list of action steps to execute."""

    @classmethod
    def from_dict(cls, data: dict) -> "TaskSequence":
        """Create TaskSequence from dictionary.

        Args:
            data: Dictionary with task sequence data

        Returns:
            TaskSequence instance
        """
        steps = []
        if "steps" in data:
            steps = [ActionStep.from_dict(step) for step in data["steps"]]

        return cls(steps=steps)

    def to_dict(self) -> dict:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation
        """
        return {"steps": [step.to_dict() for step in self.steps]}


# Forward reference
class ActionStep:
    """Placeholder for ActionStep class."""

    @classmethod
    def from_dict(cls, data: dict) -> "ActionStep":
        """Create ActionStep from dictionary.

        Args:
            data: Dictionary with action step data

        Returns:
            ActionStep instance
        """
        # Placeholder implementation
        return cls()

    def to_dict(self) -> dict:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation
        """
        # Placeholder implementation
        return {}
