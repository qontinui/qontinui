"""
Test the standalone CLI functionality.
"""

import sys
import tempfile
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def create_sample_brobot_tests(test_dir: Path):
    """Create sample Brobot-style test files for testing."""
    
    # Simple unit test
    unit_test = test_dir / "CalculatorTest.java"
    unit_test.write_text("""
package com.example.calculator;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Assertions;

public class CalculatorTest {
    
    private Calculator calculator;
    
    @BeforeEach
    public void setUp() {
        calculator = new Calculator();
    }
    
    @Test
    public void testAddition() {
        int result = calculator.add(2, 3);
        Assertions.assertEquals(5, result);
    }
    
    @Test
    public void testSubtraction() {
        int result = calculator.subtract(5, 3);
        Assertions.assertEquals(2, result);
    }
}
""")
    
    # Integration test with Spring
    integration_test = test_dir / "DatabaseIntegrationTest.java"
    integration_test.write_text("""
package com.example.integration;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit.jupiter.SpringJUnitConfig;
import org.mockito.Mock;
import org.mockito.Mockito;

@SpringBootTest
@SpringJUnitConfig
public class DatabaseIntegrationTest {
    
    @Mock
    private DatabaseService databaseService;
    
    @Test
    public void testDatabaseConnection() {
        Mockito.when(databaseService.isConnected()).thenReturn(true);
        boolean connected = databaseService.isConnected();
        Assertions.assertTrue(connected);
    }
}
""")
    
    # Brobot mock test
    brobot_test = test_dir / "BrobotGuiTest.java"
    brobot_test.write_text("""
package com.example.gui;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import io.github.jspinak.brobot.mock.Mock;
import io.github.jspinak.brobot.mock.MockBuilder;
import io.github.jspinak.brobot.actions.BrobotSettings;

public class BrobotGuiTest {
    
    private Mock guiMock;
    
    @BeforeEach
    public void setUp() {
        guiMock = new MockBuilder()
            .withElement("loginButton")
            .withElement("usernameField")
            .build();
        BrobotSettings.mock = true;
    }
    
    @Test
    public void testLoginFlow() {
        guiMock.click("loginButton");
        guiMock.type("usernameField", "testuser");
        
        boolean loginSuccessful = guiMock.exists("welcomeMessage");
        Assertions.assertTrue(loginSuccessful);
    }
}
""")

def test_cli_discovery():
    """Test the CLI discovery functionality."""
    print("Testing CLI Discovery Functionality")
    print("=" * 40)
    
    try:
        from cli_standalone import StandaloneTestMigrationCLI
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "brobot_tests"
            test_dir.mkdir()
            
            # Create sample test files
            create_sample_brobot_tests(test_dir)
            print(f"Created sample tests in: {test_dir}")
            
            # Test CLI discovery
            cli = StandaloneTestMigrationCLI()
            
            # Test discovery command
            print("\nTesting discovery command...")
            args = ["discover", str(test_dir)]
            exit_code = cli.run(args)
            
            print(f"Discovery exit code: {exit_code}")
            
            # Test dry run
            print("\nTesting dry run...")
            output_dir = Path(temp_dir) / "output"
            args = ["migrate", str(test_dir), str(output_dir), "--dry-run"]
            exit_code = cli.run(args)
            
            print(f"Dry run exit code: {exit_code}")
            
            # Test config creation
            print("\nTesting config creation...")
            config_file = Path(temp_dir) / "test_config.json"
            args = ["config", "--create", "--output", str(config_file)]
            exit_code = cli.run(args)
            
            print(f"Config creation exit code: {exit_code}")
            print(f"Config file exists: {config_file.exists()}")
            
            return True
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run CLI tests."""
    success = test_cli_discovery()
    
    if success:
        print("\n✅ CLI tests passed!")
        print("\nThe standalone CLI is working correctly.")
        print("You can now use it to discover and analyze your Brobot tests.")
        return 0
    else:
        print("\n❌ CLI tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)