#!/usr/bin/env python3
"""
Entry point for the Brobot Test Migration CLI.

This script provides a clean entry point that handles Python path setup
and launches the migration CLI with proper import resolution.
"""

import sys
from pathlib import Path

def setup_python_path():
    """Set up Python path for proper module imports."""
    # Add the test_migration directory to Python path
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

def main():
    """Main entry point for the migration CLI."""
    setup_python_path()
    
    try:
        from cli import TestMigrationCLI
        
        print("🚀 Brobot Test Migration System")
        print("=" * 50)
        
        cli = TestMigrationCLI()
        cli.run()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()