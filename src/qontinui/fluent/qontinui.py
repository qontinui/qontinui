"""Entry point for the Qontinui fluent API - ported from Qontinui framework.

This module provides factory methods to start building automation sequences
using a fluent, chainable interface.
"""


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .action_sequence_builder import ActionSequenceBuilder


class Qontinui:
    """Entry point for the Qontinui fluent API.

    Port of Brobot from Qontinui framework class.

    This class provides static factory methods to start building
    automation sequences using a fluent, chainable interface.

    Example usage:
        # Assuming user_field, password_field, submit_button are StateImage objects
        # and username, password are StateString objects
        login_sequence = Qontinui.build_sequence()\\
            .with_name("login")\\
            .find(user_field)\\
            .then_click()\\
            .then_type(username)\\
            .find(password_field)\\
            .then_click()\\
            .then_type(password)\\
            .find(submit_button)\\
            .then_click()\\
            .build()
    """

    @staticmethod
    def build_sequence() -> ActionSequenceBuilder:
        """Start building a new action sequence.

        Returns:
            A new ActionSequenceBuilder instance
        """
        from .action_sequence_builder import ActionSequenceBuilder

        return ActionSequenceBuilder()

    def __init__(self) -> None:
        """Private constructor to prevent instantiation."""
        raise RuntimeError("Qontinui is a utility class - not meant to be instantiated")
