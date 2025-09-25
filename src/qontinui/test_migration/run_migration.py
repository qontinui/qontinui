#!/usr/bin/env python3
"""
Package runner for the test migration CLI.
This avoids all relative import issues by running as a proper package.
"""

import sys
from pathlib import Path

# Add the qontinui source directory to Python path
migration_dir = Path(__file__).parent
qontinui_src = migration_dir.parent.parent
sys.path.insert(0, str(qontinui_src))

# Now imports work correctly as a package
try:
    from qontinui.test_migration.cli import TestMigrationCLI

    def main():
        """Main entry point."""
        cli = TestMigrationCLI()
        exit_code = cli.run()
        sys.exit(exit_code)

    if __name__ == "__main__":
        main()

except ImportError as e:
    print(f"Import error: {e}")
    print("Falling back to standalone CLI...")

    # Fallback to standalone version
    sys.path.insert(0, str(migration_dir))
    from cli_standalone import StandaloneTestMigrationCLI

    def main():
        """Fallback main entry point."""
        cli = StandaloneTestMigrationCLI()
        exit_code = cli.run()
        sys.exit(exit_code)

    if __name__ == "__main__":
        main()
