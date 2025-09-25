"""
Simple test to verify core components work.
"""

import sys
import tempfile
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))


def test_basic_orchestrator():
    """Test basic orchestrator functionality without complex dependencies."""
    try:
        from core.models import MigrationConfig

        # Create a simple config
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_dir = temp_path / "source"
            target_dir = temp_path / "target"
            source_dir.mkdir()
            target_dir.mkdir()

            config = MigrationConfig(
                source_directories=[source_dir],
                target_directory=target_dir,
                preserve_structure=True,
                enable_mock_migration=True,
                diagnostic_level="detailed",
                parallel_execution=False,
            )

            print("✓ Configuration created successfully")
            print(f"  - Source directories: {config.source_directories}")
            print(f"  - Target directory: {config.target_directory}")

            return True

    except Exception as e:
        print(f"✗ Failed to create config: {e}")
        return False


def test_scanner():
    """Test the scanner component."""
    try:
        from discovery.scanner import BrobotTestScanner

        scanner = BrobotTestScanner()
        print("✓ Scanner created successfully")

        # Test with empty directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            results = scanner.scan_directory(temp_path)
            print(f"  - Scanned empty directory: {len(results)} files found")

        return True

    except Exception as e:
        print(f"✗ Failed to create scanner: {e}")
        return False


def test_pytest_runner():
    """Test the pytest runner component."""
    try:
        from execution.pytest_runner import PytestRunner

        runner = PytestRunner()
        print("✓ Pytest runner created successfully")

        # Test environment validation
        errors = runner.validate_test_environment()
        print(f"  - Environment validation: {len(errors)} errors")

        return True

    except Exception as e:
        print(f"✗ Failed to create pytest runner: {e}")
        return False


def main():
    """Run basic component tests."""
    print("Testing Basic Migration Components")
    print("=" * 40)

    tests = [test_basic_orchestrator, test_scanner, test_pytest_runner]

    results = []
    for test in tests:
        results.append(test())
        print()

    # Summary
    passed = sum(results)
    total = len(results)

    print("Summary:")
    print(f"  Passed: {passed}/{total}")

    if passed == total:
        print("✓ Basic components working correctly!")
        return 0
    else:
        print("✗ Some components failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
