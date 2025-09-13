"""
Simple test script to verify migration components work.
"""

import sys
import tempfile
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_orchestrator_creation():
    """Test that we can create a migration orchestrator."""
    try:
        from config import TestMigrationConfig
        from core.models import MigrationConfig
        from orchestrator import TestMigrationOrchestrator
        
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
                parallel_execution=False
            )
            
            # Create orchestrator
            orchestrator = TestMigrationOrchestrator(config)
            
            print("✓ Migration orchestrator created successfully")
            print(f"  - Config: {orchestrator.config}")
            print(f"  - Components initialized: {orchestrator.scanner is not None}")
            
            return True
            
    except Exception as e:
        print(f"✗ Failed to create orchestrator: {e}")
        return False

def test_cli_creation():
    """Test that we can create the CLI."""
    try:
        from cli import TestMigrationCLI
        
        cli = TestMigrationCLI()
        print("✓ CLI created successfully")
        print(f"  - Parser created: {cli.parser is not None}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to create CLI: {e}")
        return False

def test_reporting_dashboard():
    """Test that we can create the reporting dashboard."""
    try:
        from reporting.dashboard import MigrationReportingDashboard
        
        dashboard = MigrationReportingDashboard()
        print("✓ Reporting dashboard created successfully")
        
        # Test report generation with empty directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            report_data = dashboard.generate_comprehensive_report(
                temp_path,
                include_coverage=False,
                include_diagnostics=False
            )
            
            print(f"  - Report generated: {len(report_data)} sections")
            
        return True
        
    except Exception as e:
        print(f"✗ Failed to create reporting dashboard: {e}")
        return False

def main():
    """Run all component tests."""
    print("Testing Migration Components")
    print("=" * 40)
    
    tests = [
        test_orchestrator_creation,
        test_cli_creation,
        test_reporting_dashboard
    ]
    
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
        print("✓ All components working correctly!")
        return 0
    else:
        print("✗ Some components failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)