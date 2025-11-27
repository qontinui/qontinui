"""Utility functions for variable management and persistence.

This module provides helper functions for:
- Loading/saving variables from/to JSON files
- Merging variable scopes with proper precedence
- Validating variable values for JSON serialization
- Variable interpolation and resolution
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_variables_from_json(file_path: Path | str) -> dict[str, Any]:
    """Load variables from a JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Dictionary of variables loaded from file

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file contains invalid JSON or wrong format

    Example:
        >>> vars = load_variables_from_json("workflow_vars.json")
        >>> print(vars["user_id"])
        '12345'
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Variables file not found: {file_path}")

    try:
        with open(file_path, encoding="utf-8") as f:
            variables = json.load(f)

        if not isinstance(variables, dict):
            raise ValueError(
                f"Invalid format in {file_path}: expected dict, " f"got {type(variables).__name__}"
            )

        logger.info(f"Loaded {len(variables)} variables from {file_path}")
        return variables

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from {file_path}: {e}") from e
    except OSError as e:
        raise OSError(f"Failed to read variables from {file_path}: {e}") from e


def save_variables_to_json(
    variables: dict[str, Any],
    file_path: Path | str,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """Save variables to a JSON file.

    Args:
        variables: Dictionary of variables to save
        file_path: Path to JSON file
        indent: JSON indentation (default: 2)
        ensure_ascii: If True, escape non-ASCII characters (default: False)

    Raises:
        ValueError: If variables are not JSON-serializable
        OSError: If file write fails

    Example:
        >>> vars = {"user_id": "123", "count": 42}
        >>> save_variables_to_json(vars, "workflow_vars.json")
    """
    file_path = Path(file_path)

    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(variables, f, indent=indent, ensure_ascii=ensure_ascii)

        logger.info(f"Saved {len(variables)} variables to {file_path}")

    except (TypeError, ValueError) as e:
        raise ValueError(f"Failed to serialize variables to JSON: {e}") from e
    except OSError as e:
        raise OSError(f"Failed to write variables to {file_path}: {e}") from e


def merge_variable_scopes(
    *scopes: dict[str, Any],
    precedence: str = "last",
) -> dict[str, Any]:
    """Merge multiple variable scopes with specified precedence.

    Args:
        *scopes: Variable dictionaries to merge
        precedence: Merge strategy:
            - "last": Later scopes override earlier (default)
            - "first": Earlier scopes take precedence

    Returns:
        Merged dictionary

    Example:
        >>> global_vars = {"x": 1, "y": 2}
        >>> workflow_vars = {"y": 20, "z": 3}
        >>> execution_vars = {"z": 30}
        >>> merged = merge_variable_scopes(
        ...     global_vars, workflow_vars, execution_vars
        ... )
        >>> merged
        {'x': 1, 'y': 20, 'z': 30}
    """
    if precedence not in ("last", "first"):
        raise ValueError(f"Invalid precedence '{precedence}'. Must be 'last' or 'first'")

    merged: dict[str, Any] = {}

    if precedence == "last":
        # Later scopes override earlier
        for scope in scopes:
            merged.update(scope)
    else:  # first
        # Earlier scopes take precedence
        for scope in reversed(scopes):
            merged.update(scope)

    return merged


def is_json_serializable(value: Any) -> bool:
    """Check if a value is JSON-serializable.

    Args:
        value: Value to check

    Returns:
        True if value can be serialized to JSON, False otherwise

    Example:
        >>> is_json_serializable({"x": 1, "y": "text"})
        True
        >>> is_json_serializable(lambda x: x)
        False
    """
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError, OverflowError):
        return False


def validate_variable_name(name: str) -> bool:
    """Validate that a variable name follows Python identifier rules.

    Args:
        name: Variable name to validate

    Returns:
        True if valid, False otherwise

    Example:
        >>> validate_variable_name("user_id")
        True
        >>> validate_variable_name("123invalid")
        False
        >>> validate_variable_name("valid-kebab")  # Hyphens not allowed
        False
    """
    if not name:
        return False

    # Must start with letter or underscore
    # Can contain letters, digits, underscores
    pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*$"
    return bool(re.match(pattern, name))


def interpolate_variables(
    text: str,
    variables: dict[str, Any],
    pattern: str = r"\$\{([^}]+)\}",
) -> str:
    """Interpolate variable references in text.

    Replaces ${var_name} patterns with variable values from the dict.

    Args:
        text: Text containing variable references
        variables: Dictionary of variable values
        pattern: Regex pattern for variable references (default: ${...})

    Returns:
        Text with variables interpolated

    Example:
        >>> vars = {"user": "alice", "count": 42}
        >>> interpolate_variables("Hello ${user}, count=${count}", vars)
        'Hello alice, count=42'
        >>> interpolate_variables("Missing ${missing}", vars)
        'Missing ${missing}'
    """

    def replace_var(match: re.Match) -> str:
        var_name = match.group(1)
        if var_name in variables:
            return str(variables[var_name])
        else:
            # Keep original if variable not found
            return match.group(0)

    return re.sub(pattern, replace_var, text)


def resolve_variable_reference(
    value: Any,
    variables: dict[str, Any],
) -> Any:
    """Resolve variable reference if value is a reference string.

    If value is a string like "${var_name}", returns the variable value.
    Otherwise returns the value unchanged.

    Args:
        value: Value that may be a variable reference
        variables: Dictionary of variable values

    Returns:
        Resolved value

    Example:
        >>> vars = {"user_id": "12345"}
        >>> resolve_variable_reference("${user_id}", vars)
        '12345'
        >>> resolve_variable_reference("literal", vars)
        'literal'
        >>> resolve_variable_reference(42, vars)
        42
    """
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        var_name = value[2:-1]
        return variables.get(var_name, value)
    return value


def get_nested_variable(
    variables: dict[str, Any],
    path: str,
    separator: str = ".",
    default: Any = None,
) -> Any:
    """Get nested variable using dot notation path.

    Args:
        variables: Dictionary of variables
        path: Dot-separated path (e.g., "user.address.city")
        separator: Path separator (default: ".")
        default: Default value if path not found

    Returns:
        Value at path or default if not found

    Example:
        >>> vars = {
        ...     "user": {
        ...         "name": "alice",
        ...         "address": {"city": "NYC"}
        ...     }
        ... }
        >>> get_nested_variable(vars, "user.address.city")
        'NYC'
        >>> get_nested_variable(vars, "user.missing", default="N/A")
        'N/A'
    """
    parts = path.split(separator)
    current = variables

    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default

    return current


def set_nested_variable(
    variables: dict[str, Any],
    path: str,
    value: Any,
    separator: str = ".",
    create_missing: bool = True,
) -> bool:
    """Set nested variable using dot notation path.

    Args:
        variables: Dictionary of variables (modified in place)
        path: Dot-separated path (e.g., "user.address.city")
        value: Value to set
        separator: Path separator (default: ".")
        create_missing: If True, create missing intermediate dicts

    Returns:
        True if value was set, False if path invalid

    Example:
        >>> vars = {}
        >>> set_nested_variable(vars, "user.address.city", "NYC")
        True
        >>> vars
        {'user': {'address': {'city': 'NYC'}}}
    """
    parts = path.split(separator)
    current = variables

    # Navigate to parent
    for part in parts[:-1]:
        if part not in current:
            if create_missing:
                current[part] = {}
            else:
                return False
        elif not isinstance(current[part], dict):
            # Path blocked by non-dict value
            return False
        current = current[part]

    # Set final value
    current[parts[-1]] = value
    return True


def filter_variables_by_prefix(
    variables: dict[str, Any],
    prefix: str,
    strip_prefix: bool = False,
) -> dict[str, Any]:
    """Filter variables by name prefix.

    Args:
        variables: Dictionary of variables
        prefix: Prefix to filter by
        strip_prefix: If True, remove prefix from keys in result

    Returns:
        Filtered dictionary

    Example:
        >>> vars = {"app_name": "test", "app_version": "1.0", "user_id": "123"}
        >>> filter_variables_by_prefix(vars, "app_")
        {'app_name': 'test', 'app_version': '1.0'}
        >>> filter_variables_by_prefix(vars, "app_", strip_prefix=True)
        {'name': 'test', 'version': '1.0'}
    """
    filtered = {}

    for key, value in variables.items():
        if key.startswith(prefix):
            if strip_prefix:
                new_key = key[len(prefix) :]
                filtered[new_key] = value
            else:
                filtered[key] = value

    return filtered


def sanitize_for_persistence(
    variables: dict[str, Any],
    max_value_size: int | None = None,
    skip_non_serializable: bool = True,
) -> dict[str, Any]:
    """Sanitize variables for JSON persistence.

    Removes or converts non-serializable values.

    Args:
        variables: Dictionary of variables
        max_value_size: Optional max size for string values (chars)
        skip_non_serializable: If True, skip non-serializable values;
                               if False, convert to string

    Returns:
        Sanitized dictionary safe for JSON serialization

    Example:
        >>> vars = {
        ...     "valid": "text",
        ...     "func": lambda x: x,
        ...     "large": "x" * 10000
        ... }
        >>> sanitized = sanitize_for_persistence(
        ...     vars, max_value_size=100, skip_non_serializable=True
        ... )
        >>> "func" in sanitized
        False
        >>> len(sanitized["large"])
        100
    """
    sanitized = {}

    for key, value in variables.items():
        # Check serializability
        if not is_json_serializable(value):
            if skip_non_serializable:
                logger.warning(
                    f"Skipping non-serializable variable '{key}': " f"{type(value).__name__}"
                )
                continue
            else:
                # Convert to string representation
                value = str(value)
                logger.warning(f"Converted non-serializable variable '{key}' to string")

        # Truncate large strings
        if max_value_size and isinstance(value, str) and len(value) > max_value_size:
            value = value[:max_value_size]
            logger.warning(f"Truncated variable '{key}' to {max_value_size} chars")

        sanitized[key] = value

    return sanitized


def create_variable_snapshot(
    execution_vars: dict[str, Any],
    workflow_vars: dict[str, Any],
    global_vars: dict[str, Any],
) -> dict[str, Any]:
    """Create a snapshot of all variable scopes.

    Args:
        execution_vars: Execution scope variables
        workflow_vars: Workflow scope variables
        global_vars: Global scope variables

    Returns:
        Snapshot dictionary with separated scopes

    Example:
        >>> snapshot = create_variable_snapshot(
        ...     {"temp": 1},
        ...     {"user_id": "123"},
        ...     {"api_key": "secret"}
        ... )
        >>> snapshot["scopes"]["workflow"]
        {'user_id': '123'}
    """
    import time

    return {
        "timestamp": time.time(),
        "scopes": {
            "execution": dict(execution_vars),
            "workflow": dict(workflow_vars),
            "global": dict(global_vars),
        },
        "merged": merge_variable_scopes(global_vars, workflow_vars, execution_vars),
    }


def restore_variable_snapshot(
    snapshot: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Restore variable scopes from snapshot.

    Args:
        snapshot: Snapshot created by create_variable_snapshot

    Returns:
        Tuple of (execution_vars, workflow_vars, global_vars)

    Raises:
        ValueError: If snapshot format is invalid

    Example:
        >>> snapshot = {
        ...     "scopes": {
        ...         "execution": {"temp": 1},
        ...         "workflow": {"user_id": "123"},
        ...         "global": {"api_key": "secret"}
        ...     }
        ... }
        >>> exec_vars, wf_vars, global_vars = restore_variable_snapshot(snapshot)
    """
    if "scopes" not in snapshot:
        raise ValueError("Invalid snapshot: missing 'scopes' key")

    scopes = snapshot["scopes"]

    execution_vars = dict(scopes.get("execution", {}))
    workflow_vars = dict(scopes.get("workflow", {}))
    global_vars = dict(scopes.get("global", {}))

    return execution_vars, workflow_vars, global_vars
