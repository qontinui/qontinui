#!/usr/bin/env python3
"""Standalone test for JSON executor without main Qontinui imports."""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import only the json_executor module directly
from qontinui.json_executor.json_runner import JSONRunner
from qontinui.json_executor.config_parser import ConfigParser

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_json_executor.py <config.json>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    # Test the runner
    print(f"Testing JSON executor with: {config_file}")
    print("-" * 50)
    
    runner = JSONRunner()
    
    if runner.load_configuration(config_file):
        print("\n✓ Configuration loaded successfully")
        
        summary = runner.get_summary()
        print("\nConfiguration Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        if '--dry-run' in sys.argv:
            print("\nDry run complete - configuration is valid")
        else:
            print("\nWould you like to run the automation? (y/n): ", end='')
            response = input().strip().lower()
            if response == 'y':
                print("\nRunning automation...")
                runner.run()
    else:
        print("\n✗ Failed to load configuration")
        sys.exit(1)

if __name__ == '__main__':
    main()