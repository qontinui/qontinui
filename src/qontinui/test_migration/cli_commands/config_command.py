"""Config command handler."""

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

from cli_utils.config_loader import ConfigLoader
from config import TestMigrationConfig
from cli_commands.base_command import BaseCommand, CommandResult


class ConfigCommand(BaseCommand):
    """
    Command to manage migration configuration.

    Supports creating and validating configuration files.
    """

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add config command arguments."""
        parser.add_argument("--create", action="store_true", help="Create a new configuration file")

        parser.add_argument(
            "--validate", action="store_true", help="Validate an existing configuration file"
        )

        parser.add_argument("--output", type=Path, help="Output file for configuration")

        parser.add_argument("--input", type=Path, help="Input configuration file to validate")

    def execute(self, args: Namespace) -> CommandResult:
        """Execute the config command."""
        if args.create:
            return self._create_config_file(args)
        elif args.validate:
            return self._validate_config_file(args)
        else:
            return CommandResult(exit_code=1, message="Error: Must specify --create or --validate")

    def _create_config_file(self, args: Namespace) -> CommandResult:
        """Create a new configuration file."""
        try:
            # Create default configuration
            config = TestMigrationConfig.create_default_config([], Path("tests/migrated"))

            # Convert to dictionary for serialization
            config_dict = {
                "source_directories": [str(d) for d in config.source_directories],
                "target_directory": str(config.target_directory),
                "preserve_structure": config.preserve_structure,
                "enable_mock_migration": config.enable_mock_migration,
                "diagnostic_level": config.diagnostic_level,
                "parallel_execution": config.parallel_execution,
                "comparison_mode": config.comparison_mode,
                "java_test_patterns": config.java_test_patterns,
                "exclude_patterns": config.exclude_patterns,
            }

            # Determine output file
            output_file = args.output or Path("migration_config.json")

            # Save configuration
            with open(output_file, "w") as f:
                json.dump(config_dict, f, indent=2)

            return CommandResult(exit_code=0, message=f"Configuration file created: {output_file}")

        except Exception as e:
            return CommandResult(exit_code=1, message=f"Failed to create configuration: {str(e)}")

    def _validate_config_file(self, args: Namespace) -> CommandResult:
        """Validate an existing configuration file."""
        config_file = args.input or args.config

        if not config_file:
            return CommandResult(exit_code=1, message="Error: No configuration file specified")

        is_valid, error_message = ConfigLoader.validate_config_file(config_file)

        if is_valid:
            return CommandResult(exit_code=0, message="Configuration file is valid")
        else:
            return CommandResult(exit_code=1, message=f"Error: {error_message}")
