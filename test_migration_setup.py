#!/usr/bin/env python3
"""
Simple test script to verify the test migration system setup.
This script tests the core components without requiring full qontinui dependencies.
"""

import sys
from pathlib import Path

# Add src to path and the test_migration module specifically
src_path = Path(__file__).parent / "src"
test_migration_path = src_path / "qontinui" / "test_migration"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(test_migration_path))

def test_core_models():
    """Test that core models can be imported and used."""
    try:
        from core.models import (
            TestType, 
            Dependency, 
            GuiModel, 
            MockUsage, 
            TestFile, 
            TestMethod,
            MigrationConfig
        )
        
        print("✓ Core models imported successfully")
        
        # Test basic model creation
        dep = Dependency(
            java_import="org.junit.jupiter.api.Test",
            python_equivalent="pytest",
            requires_adaptation=True
        )
        assert dep.java_import == "org.junit.jupiter.api.Test"
        print("✓ Dependency model works correctly")
        
        gui_model = GuiModel(
            model_name="TestWindow",
            elements={"button1": {"type": "button"}},
            actions=["click"]
        )
        assert gui_model.model_name == "TestWindow"
        print("✓ GuiModel works correctly")
        
        test_file = TestFile(
            path=Path("SampleTest.java"),
            test_type=TestType.UNIT,
            class_name="SampleTest"
        )
        assert test_file.test_type == TestType.UNIT
        print("✓ TestFile model works correctly")
        
        config = MigrationConfig(
            source_directories=[Path("java/tests")],
            target_directory=Path("python/tests")
        )
        assert config.preserve_structure is True
        print("✓ MigrationConfig works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Core models test failed: {e}")
        return False


def test_core_interfaces():
    """Test that core interfaces can be imported."""
    try:
        from core.interfaces import (
            TestScanner,
            TestTranslator,
            MockAnalyzer,
            MockGenerator,
            TestRunner,
            FailureAnalyzer,
            BehaviorComparator,
            DiagnosticReporter,
            MigrationOrchestrator
        )
        
        print("✓ Core interfaces imported successfully")
        
        # Verify they are abstract base classes
        import inspect
        assert inspect.isabstract(TestScanner)
        assert inspect.isabstract(TestTranslator)
        assert inspect.isabstract(MockAnalyzer)
        print("✓ Interfaces are properly abstract")
        
        return True
        
    except Exception as e:
        print(f"✗ Core interfaces test failed: {e}")
        return False


def test_config_module():
    """Test the configuration module."""
    try:
        from config import TestMigrationConfig
        
        print("✓ Configuration module imported successfully")
        
        # Test default config creation
        config = TestMigrationConfig.create_default_config(
            source_directories=[Path("java/tests")],
            target_directory=Path("python/tests")
        )
        assert config.preserve_structure is True
        print("✓ Default config creation works")
        
        # Test dependency mappings
        mappings = TestMigrationConfig.get_dependency_mapping()
        assert "org.junit.jupiter.api.Test" in mappings
        print("✓ Dependency mappings work")
        
        # Test Brobot mock mappings
        mock_mappings = TestMigrationConfig.get_brobot_mock_mappings()
        assert "io.github.jspinak.brobot.mock.Mock" in mock_mappings
        print("✓ Brobot mock mappings work")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration module test failed: {e}")
        return False


def test_directory_structure():
    """Test that the directory structure is correctly created."""
    base_path = Path(__file__).parent / "src" / "qontinui" / "test_migration"
    
    required_dirs = [
        "core",
        "discovery", 
        "translation",
        "mocks",
        "execution",
        "validation",
        "tests"
    ]
    
    for dir_name in required_dirs:
        dir_path = base_path / dir_name
        if not dir_path.exists():
            print(f"✗ Missing directory: {dir_path}")
            return False
        
        init_file = dir_path / "__init__.py"
        if not init_file.exists():
            print(f"✗ Missing __init__.py in: {dir_path}")
            return False
    
    print("✓ Directory structure is correct")
    return True


def main():
    """Run all tests."""
    print("Testing Brobot Test Migration System Setup")
    print("=" * 50)
    
    tests = [
        test_directory_structure,
        test_core_models,
        test_core_interfaces,
        test_config_module,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        if test():
            passed += 1
        else:
            print(f"Test {test.__name__} failed!")
    
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Test migration system setup is complete.")
        return 0
    else:
        print("❌ Some tests failed. Please check the setup.")
        return 1


if __name__ == "__main__":
    sys.exit(main())