#!/usr/bin/env python3
"""Command-line interface for running Qontinui JSON configurations."""

import argparse
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from qontinui.json_executor import JSONRunner


def main():
    parser = argparse.ArgumentParser(
        description="Run Qontinui automation from JSON configuration files"
    )

    parser.add_argument("config", help="Path to JSON configuration file")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without executing",
    )

    args = parser.parse_args()

    # Check if file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    # Create runner
    runner = JSONRunner()

    # Load configuration
    if not runner.load_configuration(str(config_path)):
        print("Failed to load configuration")
        sys.exit(1)

    if args.dry_run:
        print("\nDry run complete - configuration is valid")
        print("\nSummary:")
        summary = runner.get_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")
        sys.exit(0)

    # Run automation
    print("\nRunning automation in state machine mode...")
    print("Press Ctrl+C to stop\n")

    success = runner.run()

    if success:
        print("\nAutomation completed successfully")
    else:
        print("\nAutomation failed or was interrupted")
        sys.exit(1)


if __name__ == "__main__":
    main()
