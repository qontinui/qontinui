"""Validates configuration schema and format."""

from typing import Any


class SchemaValidator:
    """Validates Qontinui configuration format and schema.

    SchemaValidator ensures configuration data matches required format before
    parsing. It rejects legacy v1.0.0 format and validates v2.0.0 structure.

    The validator checks for:
    - Legacy "processes" field (v1.0.0 format no longer supported)
    - Legacy workflow format (missing connections field)
    - Required v2.0.0 graph-based workflow structure

    Example:
        >>> validator = SchemaValidator()
        >>> try:
        ...     validator.validate(config_data)
        ... except ValueError as e:
        ...     print(f"Invalid config: {e}")
    """

    def validate(self, data: dict[str, Any]) -> None:
        """Validate configuration format.

        Checks configuration data for correct format and rejects legacy v1.0.0.

        Args:
            data: Configuration dictionary to validate.

        Raises:
            ValueError: If configuration uses legacy v1.0.0 format or is invalid.

        Example:
            >>> validator.validate({"version": "2.0.0", "workflows": [...]})
            >>> # Passes validation
            >>>
            >>> validator.validate({"version": "1.0.0", "processes": [...]})
            ValueError: Configuration format v1.0.0 is no longer supported.
        """
        # Check for legacy "processes" field
        if "processes" in data:
            raise ValueError(
                "Configuration format v1.0.0 is no longer supported. "
                "Please use v2.0.0 format with 'workflows' instead of 'processes'."
            )

        # Check for legacy workflow format (missing connections field)
        if "workflows" in data:
            for workflow in data["workflows"]:
                if isinstance(workflow, dict) and "connections" not in workflow:
                    raise ValueError(
                        "Configuration format v1.0.0 is no longer supported. "
                        "Please use v2.0.0 format with graph-based workflows."
                    )

    def validate_workflows_field(self, data: Any) -> Any:
        """Validate that workflows field is used (not legacy processes field).

        Used as Pydantic model validator to ensure transitions reference workflows.

        Args:
            data: Transition data to validate.

        Returns:
            Validated data unchanged if valid.

        Raises:
            ValueError: If data contains legacy 'processes' field.
        """
        if isinstance(data, dict):
            if "processes" in data:
                raise ValueError(
                    "Configuration format v1.0.0 is no longer supported. "
                    "Please use 'workflows' instead of 'processes'."
                )
        return data
