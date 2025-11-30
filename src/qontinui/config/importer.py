"""
JSON loading and parsing utilities for action configurations.

This module provides utilities for loading action configurations from JSON files,
parsing them into validated Pydantic models, and handling common import scenarios.
"""

import json
from pathlib import Path
from typing import Any

from .schema import Action
from .validator import ActionValidationError, ActionValidator


class ImportError(Exception):
    """Raised when importing action configurations fails."""

    pass


class ActionImporter:
    """Importer for action configurations from JSON."""

    def __init__(self, validate: bool = True, strict: bool = False) -> None:
        """
        Initialize the importer.

        Args:
            validate: Whether to validate actions after loading
            strict: Whether to raise errors on warnings (requires validate=True)
        """
        self.validate = validate
        self.strict = strict
        self.validator = ActionValidator() if validate else None

    def load_from_file(self, file_path: str | Path) -> list[Action]:
        """
        Load actions from a JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            List of validated Action models

        Raises:
            ImportError: If loading or validation fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ImportError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise ImportError(f"Not a file: {file_path}")

        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ImportError(f"Invalid JSON in {file_path}: {e}") from e
        except Exception as e:
            raise ImportError(f"Error reading {file_path}: {e}") from e

        return self.load_from_dict(data, source=str(file_path))

    def load_from_string(self, json_string: str) -> list[Action]:
        """
        Load actions from a JSON string.

        Args:
            json_string: JSON string containing action data

        Returns:
            List of validated Action models

        Raises:
            ImportError: If parsing or validation fails
        """
        try:
            data = json.loads(json_string)
        except json.JSONDecodeError as e:
            raise ImportError(f"Invalid JSON string: {e}") from e

        return self.load_from_dict(data, source="<string>")

    def load_from_dict(
        self, data: dict[str, Any] | list[dict[str, Any]], source: str = "<dict>"
    ) -> list[Action]:
        """
        Load actions from a dictionary or list of dictionaries.

        Args:
            data: Dictionary or list of dictionaries containing action data
            source: Source identifier for error messages

        Returns:
            List of validated Action models

        Raises:
            ImportError: If validation fails
        """
        # Handle both single action and list of actions
        if isinstance(data, dict):
            actions_data = [data]
        elif isinstance(data, list):
            actions_data = data
        else:
            raise ImportError(
                f"Invalid data type in {source}: expected dict or list, got {type(data).__name__}"
            )

        if not self.validate:
            # Create Action models without validation
            try:
                return [Action.model_validate(action_data) for action_data in actions_data]
            except Exception as e:
                raise ImportError(f"Error creating Action models from {source}: {e}") from e

        # Validate actions
        try:
            if self.validator is not None:
                actions = self.validator.validate_actions(actions_data)
        except ActionValidationError as e:
            raise ImportError(f"Validation failed for {source}: {e}") from e

        # Check for sequence warnings
        warnings = (
            self.validator.validate_action_sequence(actions) if self.validator is not None else []
        )
        if warnings:
            warning_text = "\n".join(f"  - {w}" for w in warnings)
            if self.strict:
                raise ImportError(
                    f"Action sequence validation failed for {source}:\n{warning_text}"
                )
            else:
                print(f"Warning: Issues found in {source}:\n{warning_text}")

        return actions

    def load_action(self, data: dict[str, Any] | str | Path) -> Action:
        """
        Load a single action from various sources.

        Args:
            data: Action data (dict, JSON string, or file path)

        Returns:
            Validated Action model

        Raises:
            ImportError: If loading or validation fails
        """
        if isinstance(data, dict):
            actions = self.load_from_dict(data, source="<dict>")
        elif isinstance(data, str | Path):
            path = Path(data)
            if path.exists():
                actions = self.load_from_file(path)
            else:
                # Try to parse as JSON string
                try:
                    actions = self.load_from_string(data)  # type: ignore[arg-type]
                except ImportError:
                    raise ImportError(f"Not a valid file path or JSON string: {data}") from None
        else:
            raise ImportError(
                f"Invalid data type: expected dict, str, or Path, got {type(data).__name__}"
            )

        if len(actions) != 1:
            raise ImportError(f"Expected single action, got {len(actions)} actions")

        return actions[0]

    def load_from_directory(
        self, directory: str | Path, pattern: str = "*.json", recursive: bool = False
    ) -> dict[str, list[Action]]:
        """
        Load all JSON files from a directory.

        Args:
            directory: Directory path
            pattern: Glob pattern for matching files
            recursive: Whether to search recursively

        Returns:
            Dictionary mapping file paths to lists of actions

        Raises:
            ImportError: If directory doesn't exist or errors occur
        """
        directory = Path(directory)

        if not directory.exists():
            raise ImportError(f"Directory not found: {directory}")

        if not directory.is_dir():
            raise ImportError(f"Not a directory: {directory}")

        results = {}
        errors = []

        # Find matching files
        if recursive:
            files = directory.rglob(pattern)
        else:
            files = directory.glob(pattern)

        for file_path in files:
            try:
                actions = self.load_from_file(file_path)
                results[str(file_path)] = actions
            except ImportError as e:
                errors.append(f"{file_path}: {e}")

        if errors and self.strict:
            raise ImportError(f"Errors loading from {directory}:\n" + "\n".join(errors))
        elif errors:
            print(f"Warning: Errors loading some files from {directory}:\n" + "\n".join(errors))

        return results


# ============================================================================
# Convenience Functions
# ============================================================================


def load_actions_from_file(
    file_path: str | Path, validate: bool = True, strict: bool = False
) -> list[Action]:
    """
    Load actions from a JSON file.

    Args:
        file_path: Path to JSON file
        validate: Whether to validate actions
        strict: Whether to raise errors on warnings

    Returns:
        List of validated Action models

    Raises:
        ImportError: If loading or validation fails
    """
    importer = ActionImporter(validate=validate, strict=strict)
    return importer.load_from_file(file_path)


def load_actions_from_string(
    json_string: str, validate: bool = True, strict: bool = False
) -> list[Action]:
    """
    Load actions from a JSON string.

    Args:
        json_string: JSON string containing action data
        validate: Whether to validate actions
        strict: Whether to raise errors on warnings

    Returns:
        List of validated Action models

    Raises:
        ImportError: If parsing or validation fails
    """
    importer = ActionImporter(validate=validate, strict=strict)
    return importer.load_from_string(json_string)


def load_actions_from_dict(
    data: dict[str, Any] | list[dict[str, Any]], validate: bool = True, strict: bool = False
) -> list[Action]:
    """
    Load actions from a dictionary or list of dictionaries.

    Args:
        data: Dictionary or list of dictionaries containing action data
        validate: Whether to validate actions
        strict: Whether to raise errors on warnings

    Returns:
        List of validated Action models

    Raises:
        ImportError: If validation fails
    """
    importer = ActionImporter(validate=validate, strict=strict)
    return importer.load_from_dict(data)


def load_action(
    data: dict[str, Any] | str | Path, validate: bool = True, strict: bool = False
) -> Action:
    """
    Load a single action from various sources.

    Args:
        data: Action data (dict, JSON string, or file path)
        validate: Whether to validate action
        strict: Whether to raise errors on warnings

    Returns:
        Validated Action model

    Raises:
        ImportError: If loading or validation fails
    """
    importer = ActionImporter(validate=validate, strict=strict)
    return importer.load_action(data)


def load_actions_from_directory(
    directory: str | Path,
    pattern: str = "*.json",
    recursive: bool = False,
    validate: bool = True,
    strict: bool = False,
) -> dict[str, list[Action]]:
    """
    Load all JSON files from a directory.

    Args:
        directory: Directory path
        pattern: Glob pattern for matching files
        recursive: Whether to search recursively
        validate: Whether to validate actions
        strict: Whether to raise errors on warnings

    Returns:
        Dictionary mapping file paths to lists of actions

    Raises:
        ImportError: If directory doesn't exist or errors occur
    """
    importer = ActionImporter(validate=validate, strict=strict)
    return importer.load_from_directory(directory, pattern, recursive)
