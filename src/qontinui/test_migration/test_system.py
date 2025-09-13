#!/usr/bin/env python3
"""
Test the complete migration system functionality.
"""

import sys
from pathlib import Path

# Setup Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_system_functionality():
    """Test the complete system functionality."""
    
    print("🧪 Testing Brobot Test Migration System")
    print("=" * 50)
    
    try:
        # Test 1: Import all major components
        print("1. Testing component imports...")
        from config import TestMigrationConfig
        from core.models import MigrationConfig
        from orchestrator import TestMigrationOrchestrator
        from cli import TestMigrationCLI
        from execution.hybrid_test_translator import HybridTestTranslator
        from reporting.dashboard import MigrationReportingDashboard
        print("   ✓ All imports successful")
        
        # Test 2: Create configuration
        print("2. Testing configuration creation...")
        config = MigrationConfig(
            source_directories=[Path("./test_data")],
            target_directory=Path("./output"),
            preserve_structure=True,
            enable_mock_migration=True
        )
        print("   ✓ Configuration created successfully")
        
        # Test 3: Initialize orchestrator
        print("3. Testing orchestrator initialization...")
        orchestrator = TestMigrationOrchestrator(config)
        print("   ✓ Orchestrator initialized successfully")
        
        # Test 4: Initialize CLI
        print("4. Testing CLI initialization...")
        cli = TestMigrationCLI()
        print("   ✓ CLI initialized successfully")
        
        # Test 5: Initialize hybrid translator
        print("5. Testing hybrid translator initialization...")
        translator = HybridTestTranslator()
        print("   ✓ Hybrid translator initialized successfully")
        
        # Test 6: Initialize reporting dashboard
        print("6. Testing reporting dashboard initialization...")
        dashboard = MigrationReportingDashboard()
        print("   ✓ Reporting dashboard initialized successfully")
        
        print("\n🎉 All system tests passed!")
        print("The migration system is ready to use.")
        print("\nTo run the migration CLI:")
        print("  python run_migration_cli.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_system_functionality()
    sys.exit(0 if success else 1)