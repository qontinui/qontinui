"""Configuration loading utilities."""

import json
from argparse import Namespace
from pathlib import Path

from core.models import MigrationConfig


class ConfigLoader:
    """
    Handles loading and creating migration configurations.

    Supports loading from:
    - Configuration files (JSON)
    - Command line arguments
    """

    @staticmethod
    def load_or_create(args: Namespace) -> MigrationConfig:
        """
        Load configuration from file or create from command line arguments.

        Args:
            args: Parsed command line arguments

        Returns:
            MigrationConfig instance
        """
        if hasattr(args, "config") and args.config and args.config.exists():
            return ConfigLoader._load_from_file(args.config)
        else:
            return ConfigLoader._create_from_args(args)

    @staticmethod
    def _load_from_file(config_file: Path) -> MigrationConfig:
        """Load configuration from a JSON file."""
        with open(config_file) as f:
            config_data = json.load(f)

        return MigrationConfig(
            source_directories=[Path(d) for d in config_data["source_directories"]],
            target_directory=Path(config_data["target_directory"]),
            preserve_structure=config_data.get("preserve_structure", True),
            enable_mock_migration=config_data.get("enable_mock_migration", True),
            diagnostic_level=config_data.get("diagnostic_level", "detailed"),
            parallel_execution=config_data.get("parallel_execution", True),
            comparison_mode=config_data.get("comparison_mode", "behavioral"),
            java_test_patterns=config_data.get(
                "java_test_patterns", ["*Test.java", "*Tests.java"]
            ),
            exclude_patterns=config_data.get("exclude_patterns", []),
        )

    @staticmethod
    def _create_from_args(args: Namespace) -> MigrationConfig:
        """Create configuration from command line arguments."""
        source_dirs = [args.source] if hasattr(args, "source") else []
        target_dir = args.target if hasattr(args, "target") else Path("tests/migrated")

        return MigrationConfig(
            source_directories=source_dirs,
            target_directory=target_dir,
            preserve_structure=getattr(args, "preserve_structure", True),
            enable_mock_migration=getattr(args, "enable_mocks", True),
            diagnostic_level=(
                "detailed"
                if args.verbose > 1
                else "normal" if args.verbose > 0 else "minimal"
            ),
            parallel_execution=getattr(args, "parallel", True),
            comparison_mode="behavioral",
        )

    @staticmethod
    def validate_config_file(config_file: Path) -> tuple[bool, str | None]:
        """
        Validate an existing configuration file.

        Args:
            config_file: Path to configuration file

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not config_file.exists():
            return False, "Configuration file not found"

        try:
            with open(config_file) as f:
                config_data = json.load(f)

            required_fields = [
                "source_directories",
                "target_directory",
                "preserve_structure",
                "enable_mock_migration",
                "diagnostic_level",
                "parallel_execution",
            ]

            missing_fields = [
                field for field in required_fields if field not in config_data
            ]

            if missing_fields:
                return False, f"Missing required fields: {missing_fields}"

            return True, None

        except json.JSONDecodeError as e:
            return False, f"Invalid JSON in configuration file: {str(e)}"
        except Exception as e:
            return False, f"Error validating configuration: {str(e)}"
