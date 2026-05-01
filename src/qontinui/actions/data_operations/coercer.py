"""Type coercion utilities for data operations.

This module provides the TypeCoercer class for converting values between
different types (string, number, boolean, array, object).
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class TypeCoercer:
    """Handles type coercion for variable values.

    Provides static methods to convert values between common types:
    - string: Convert to string representation
    - number: Convert to int or float
    - boolean: Convert to boolean (handles string representations)
    - array: Convert to list (parses JSON if needed)
    - object: Convert to dict (parses JSON if needed)

    Example:
        >>> TypeCoercer.coerce("42", "number")
        42
        >>> TypeCoercer.coerce("true", "boolean")
        True
        >>> TypeCoercer.coerce('["a", "b"]', "array")
        ['a', 'b']
    """

    @staticmethod
    def coerce(value: Any, target_type: str | None) -> Any:
        """Coerce a value to the specified type.

        Attempts intelligent conversion based on the target type and current
        value type. Returns value unchanged if target_type is None.

        Args:
            value: Value to coerce
            target_type: Target type (string, number, boolean, array, object).
                         Case-insensitive. None means no coercion.

        Returns:
            Coerced value

        Raises:
            ValueError: If coercion fails or target type is invalid

        Example:
            >>> TypeCoercer.coerce(42, "string")
            '42'
            >>> TypeCoercer.coerce("3.14", "number")
            3.14
            >>> TypeCoercer.coerce("yes", "boolean")
            True
            >>> TypeCoercer.coerce('{"x": 1}', "object")
            {'x': 1}
        """
        if target_type is None or value is None:
            return value

        target_type = target_type.lower()

        try:
            if target_type == "string":
                return str(value)

            elif target_type == "number":
                # Try int first, then float
                if isinstance(value, str):
                    if "." in value:
                        return float(value)
                    return int(value)
                return float(value)

            elif target_type == "boolean":
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    return value.lower() in ("true", "yes", "1", "on")
                return bool(value)

            elif target_type == "array":
                if isinstance(value, list | tuple):
                    return list(value)
                if isinstance(value, str):
                    return json.loads(value)
                return [value]

            elif target_type == "object":
                if isinstance(value, dict):
                    return value
                if isinstance(value, str):
                    return json.loads(value)
                raise ValueError(f"Cannot coerce {type(value).__name__} to object")

            else:
                logger.warning(
                    f"Unknown target type '{target_type}', returning value as-is"
                )
                return value

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed during type coercion: {e}")
            raise ValueError(
                f"Failed to parse JSON while coercing to {target_type}: {e}"
            ) from e
        except (ValueError, TypeError) as e:
            logger.error(f"Type coercion failed: {e}")
            raise ValueError(f"Failed to coerce value to {target_type}: {e}") from e
