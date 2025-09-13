"""
Test the minimal orchestrator functionality.
"""

import sys
import tempfile
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_minimal_orchestrator():
    """Test the minimal orchestrator."""
    try:
        from minimal_orchestrator import MinimalMigrationOrchestrator
        from core.models import MigrationConfig
        
        # Create a simple config
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_dir = temp_path / "source"
            target_dir = temp_path / "target"
            source_dir.mkdir()
            target_dir.mkdir()
            
            # Create a sample Java test file
            java_test = source_dir / "SampleTest.java"
            java_test.write_text("""
package com.example;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assertions;

public class SampleTest {
    
    @Test
    public void testSample() {
        Assertions.assertEquals(1, 1);
    }
}
""")
            
            config = MigrationConfig(
                source_directories=[source_dir],
                target_directory=target_dir,
                preserve_structure=False,
                enable_mock_migration=True,
                diagnostic_level="detailed",
                parallel_execution=False
            )
            
            # Create orchestrator
            orchestrator = MinimalMigrationOrchestrator(config)
            
            print("✓ Minimal orchestrator created successfully")
            
            # Test discovery
            discovered_tests = orchestrator.discover_tests(source_dir)
            print(f"  - Discovered {len(discovered_tests)} test files")
            
            # Test progress tracking
            progress = orchestrator.get_migration_progress()
            print(f"  - Progress: {progress}")
            
            # Create a simple Python test for validation
            python_test = target_dir / "test_sample.py"
            python_test.write_text("""
def test_sample():
    assert 1 == 1
""")
            
            # Test validation
            results = orchestrator.validate_migration(target_dir)
            print(f"  - Validation results: {results.total_tests} tests")
            
            return True
            
    except Exception as e:
        print(f"✗ Failed to test minimal orchestrator: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run minimal orchestrator test."""
    print("Testing Minimal Migration Orchestrator")
    print("=" * 40)
    
    success = test_minimal_orchestrator()
    
    if success:
        print("\n✓ Minimal orchestrator test passed!")
        return 0
    else:
        print("\n✗ Minimal orchestrator test failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)