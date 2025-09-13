"""
Setup script for the test migration CLI tool.
"""

import sys
from pathlib import Path

def setup_cli():
    """Set up the CLI tool for easy execution."""
    
    # Get the current directory
    current_dir = Path(__file__).parent
    
    # Create a simple launcher script
    launcher_content = f"""#!/usr/bin/env python3
import sys
import os

# Add the test migration module to Python path
sys.path.insert(0, r"{current_dir}")

from cli import main

if __name__ == "__main__":
    main()
"""
    
    # Write launcher script
    launcher_path = current_dir / "qontinui-test-migration"
    launcher_path.write_text(launcher_content)
    
    # Make it executable on Unix systems
    if sys.platform != "win32":
        import stat
        launcher_path.chmod(launcher_path.stat().st_mode | stat.S_IEXEC)
    
    print(f"CLI launcher created at: {launcher_path}")
    print(f"You can now run: python {launcher_path} --help")
    
    # Also create a Windows batch file
    if sys.platform == "win32":
        batch_content = f"""@echo off
python "{current_dir / 'cli.py'}" %*
"""
        batch_path = current_dir / "qontinui-test-migration.bat"
        batch_path.write_text(batch_content)
        print(f"Windows batch file created at: {batch_path}")

if __name__ == "__main__":
    setup_cli()