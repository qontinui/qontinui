#!/usr/bin/env python3
"""
Verification script for TypeScript parser installation.

This script checks that all required files are present and that
the parser can be imported and initialized correctly.
"""

import sys
from pathlib import Path


def verify_files():
    """Check that all required files exist."""
    print("Checking files...")

    required_files = [
        "package.json",
        "parser.js",
        "parser.py",
        "__init__.py",
        "README.md",
        "QUICKSTART.md",
        "IMPLEMENTATION_SUMMARY.md",
        "install.sh",
        "example_test.py",
        "integration_example.py",
        ".gitignore",
    ]

    base_dir = Path(__file__).parent
    missing_files = []

    for filename in required_files:
        filepath = base_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} (MISSING)")
            missing_files.append(filename)

    if missing_files:
        print(f"\nERROR: {len(missing_files)} file(s) missing!")
        return False

    print("\nAll files present ✓")
    return True


def verify_node_dependencies():
    """Check if Node.js dependencies are installed."""
    print("\nChecking Node.js dependencies...")

    node_modules = Path(__file__).parent / "node_modules"

    if not node_modules.exists():
        print("  ✗ node_modules directory not found")
        print("\n  Run './install.sh' to install dependencies")
        return False

    required_modules = ["typescript", "@babel/parser", "@babel/traverse"]
    missing_modules = []

    for module in required_modules:
        module_path = node_modules / module
        # Handle scoped packages
        if module.startswith("@"):
            scope, name = module.split("/")
            module_path = node_modules / scope / name

        if module_path.exists():
            print(f"  ✓ {module}")
        else:
            print(f"  ✗ {module} (MISSING)")
            missing_modules.append(module)

    if missing_modules:
        print(f"\nWARNING: {len(missing_modules)} module(s) missing!")
        print("Run './install.sh' to install dependencies")
        return False

    print("\nAll Node.js dependencies installed ✓")
    return True


def verify_python_import():
    """Check if the parser can be imported."""
    print("\nChecking Python imports...")

    try:
        from .parser import (  # noqa: F401
            ComponentInfo,
            ConditionalRenderInfo,
            EventHandlerInfo,
            FileParseResult,
            ParseResult,
            StateVariableInfo,
            TypeScriptParser,
        )

        print("  ✓ All classes can be imported")
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False

    try:
        TypeScriptParser()
        print("  ✓ TypeScriptParser can be instantiated")
    except Exception as e:
        print(f"  ✗ Instantiation error: {e}")
        return False

    print("\nPython imports successful ✓")
    return True


def verify_node_availability():
    """Check if Node.js is available."""
    print("\nChecking Node.js availability...")

    import subprocess

    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        version = result.stdout.strip()
        print(f"  ✓ Node.js {version}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ✗ Node.js not found")
        print("\n  Install Node.js from https://nodejs.org/")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("TypeScript Parser Installation Verification")
    print("=" * 60)

    checks = [
        ("Files", verify_files),
        ("Node.js", verify_node_availability),
        ("Node.js Dependencies", verify_node_dependencies),
        ("Python Imports", verify_python_import),
    ]

    results = {}
    for name, check in checks:
        try:
            results[name] = check()
        except Exception as e:
            print(f"\n✗ {name} check failed with error: {e}")
            results[name] = False

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {name}: {status}")

    all_passed = all(results.values())

    print("=" * 60)

    if all_passed:
        print("\n✓ All checks passed! The parser is ready to use.\n")
        print("Try running:")
        print("  python example_test.py")
        print("  python integration_example.py /path/to/react/project")
        return 0
    else:
        print("\n✗ Some checks failed. Please address the issues above.\n")
        if not results.get("Node.js Dependencies", True):
            print("To install Node.js dependencies:")
            print("  ./install.sh")
        return 1


if __name__ == "__main__":
    sys.exit(main())
