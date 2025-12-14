"""Exit codes for Qontinui CLI.

Standard exit codes used across all CLI commands for consistent CI/CD integration.
"""


class ExitCode:
    """Exit code constants for Qontinui CLI."""

    SUCCESS = 0
    """All tests/operations passed successfully."""

    TEST_FAILURE = 1
    """One or more tests failed during execution."""

    CONFIG_ERROR = 2
    """Configuration file is invalid or cannot be loaded."""

    EXECUTION_ERROR = 3
    """Runtime execution error (e.g., timeout, unexpected exception)."""
