"""Validates configuration schema and format."""

from typing import Any


class SchemaValidator:
    """Validates Qontinui configuration format and schema.

    SchemaValidator ensures configuration data matches required format before
    parsing. It rejects legacy v1.0.0 format and validates v2.0.0 structure.

    The validator checks for:
    - Legacy "processes" field (v1.0.0 format no longer supported)
    - Required v2.0.0 workflow structure (either graph-based with connections or sequential with actions)

    v2.0.0 workflows support two formats:
    - Graph-based: workflows with "connections" field for complex branching
    - Sequential: workflows with "actions" field for linear execution

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

        # Check for legacy workflow format (missing both connections and actions)
        # v2.0.0 workflows can use either:
        # - Graph format: with "connections" field
        # - Sequential format: with "actions" field (list of actions executed in order)
        if "workflows" in data:
            for workflow in data["workflows"]:
                if isinstance(workflow, dict):
                    has_connections = "connections" in workflow
                    has_actions = "actions" in workflow and isinstance(workflow["actions"], list)

                    # Reject if workflow has neither connections nor actions
                    if not has_connections and not has_actions:
                        raise ValueError(
                            "Configuration format v1.0.0 is no longer supported. "
                            "Please use v2.0.0 format with graph-based workflows (connections) or sequential workflows (actions)."
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
