# Migration Examples - Step-by-Step Walkthroughs

This document provides practical examples of migrating different types of Brobot tests to Qontinui.

## Example 1: Simple Unit Test Migration

### Original Java Test (Brobot)
```java
// File: brobot/library/src/test/java/com/example/CalculatorTest.java
package com.example;

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
    public void testDivisionByZero() {
        Assertions.assertThrows(ArithmeticException.class, () -> {
            calculator.divide(1, 0);
        });
    }
}
```

### Migration Command
```bash
cd qontinui/src/qontinui/test_migration
python cli.py migrate brobot/library/src/test qontinui/tests/migrated --dry-run
```

### Expected Python Output
```python
# File: qontinui/tests/migrated/test_calculator.py
import pytest

class TestCalculator:
    
    def setup_method(self):
        # TODO: Replace with actual Qontinui Calculator implementation
        self.calculator = Calculator()
    
    def test_addition(self):
        result = self.calculator.add(2, 3)
        assert result == 5
    
    def test_division_by_zero(self):
        with pytest.raises(ArithmeticException):
            self.calculator.divide(1, 0)
```

### Migration Steps
1. **Preview the migration:**
   ```bash
   python cli.py migrate brobot/library/src/test qontinui/tests/migrated --dry-run
   ```

2. **Execute the migration:**
   ```bash
   python cli.py migrate brobot/library/src/test qontinui/tests/migrated
   ```

3. **Validate the results:**
   ```bash
   python cli.py validate qontinui/tests/migrated
   ```

## Example 2: Brobot Mock Test Migration

### Original Java Test with Brobot Mocks
```java
// File: brobot/library/src/test/java/com/example/GuiAutomationTest.java
package com.example;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import io.github.jspinak.brobot.mock.Mock;
import io.github.jspinak.brobot.mock.MockBuilder;
import io.github.jspinak.brobot.actions.BrobotSettings;

public class GuiAutomationTest {
    
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
```

### Migration Command
```bash
python cli.py migrate brobot/library/src/test qontinui/tests/migrated --enable-mocks
```

### Expected Python Output
```python
# File: qontinui/tests/migrated/test_gui_automation.py
import pytest
from qontinui.test_migration.mocks import QontinuiMock, QontinuiMockBuilder
from qontinui.core.settings import QontinuiSettings

class TestGuiAutomation:
    
    def setup_method(self):
        self.gui_mock = QontinuiMockBuilder() \
            .with_element("loginButton") \
            .with_element("usernameField") \
            .build()
        QontinuiSettings.mock = True
    
    def test_login_flow(self):
        self.gui_mock.click("loginButton")
        self.gui_mock.type("usernameField", "testuser")
        
        login_successful = self.gui_mock.exists("welcomeMessage")
        assert login_successful is True
```

## Example 3: Spring Integration Test Migration

### Original Java Integration Test
```java
// File: brobot/library-test/src/test/java/com/example/DatabaseIntegrationTest.java
package com.example.integration;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit.jupiter.SpringJUnitConfig;
import org.springframework.beans.factory.annotation.Autowired;
import org.mockito.Mock;
import org.mockito.Mockito;

@SpringBootTest
@SpringJUnitConfig
public class DatabaseIntegrationTest {
    
    @Autowired
    private DatabaseService databaseService;
    
    @Mock
    private ExternalApiService externalApiService;
    
    @Test
    public void testDataPersistence() {
        // Mock external dependency
        Mockito.when(externalApiService.fetchData("key"))
               .thenReturn("test data");
        
        // Test the service
        String result = databaseService.processData("key");
        Assertions.assertEquals("processed: test data", result);
    }
}
```

### Migration Command
```bash
python cli.py migrate brobot/library-test/src/test/java qontinui/tests/integration --preserve-structure
```

### Expected Python Output
```python
# File: qontinui/tests/integration/test_database_integration.py
import pytest
from unittest.mock import Mock, patch

@pytest.mark.integration
class TestDatabaseIntegration:
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        # TODO: Set up Qontinui application context equivalent
        self.database_service = DatabaseService()
        self.external_api_service = Mock()
    
    def test_data_persistence(self):
        # Mock external dependency
        self.external_api_service.fetch_data.return_value = "test data"
        
        # Test the service
        with patch('external_api_service', self.external_api_service):
            result = self.database_service.process_data("key")
            assert result == "processed: test data"
```

## Example 4: Complex Test Suite Migration

### Scenario: Migrating Multiple Test Files

Let's say you have this structure:
```
brobot/library/src/test/java/
├── com/example/unit/
│   ├── CalculatorTest.java
│   ├── StringUtilsTest.java
│   └── ValidationTest.java
├── com/example/integration/
│   ├── DatabaseTest.java
│   └── ApiTest.java
└── com/example/gui/
    ├── LoginTest.java
    └── NavigationTest.java
```

### Step-by-Step Migration

1. **Create configuration file:**
   ```bash
   python cli.py config --create --output brobot_migration.json
   ```

2. **Edit configuration:**
   ```json
   {
     "source_directories": ["brobot/library/src/test/java"],
     "target_directory": "qontinui/tests/migrated",
     "preserve_structure": true,
     "enable_mock_migration": true,
     "diagnostic_level": "detailed",
     "parallel_execution": false,
     "java_test_patterns": ["*Test.java", "*Tests.java"],
     "exclude_patterns": ["*/target/*", "*/build/*"]
   }
   ```

3. **Preview the migration:**
   ```bash
   python cli.py migrate --config brobot_migration.json --dry-run -v
   ```

   Expected output:
   ```
   Found 7 test files:
     com/example/unit/CalculatorTest.java -> test_calculator.py (unit)
     com/example/unit/StringUtilsTest.java -> test_string_utils.py (unit)
     com/example/unit/ValidationTest.java -> test_validation.py (unit)
     com/example/integration/DatabaseTest.java -> test_database.py (integration)
     com/example/integration/ApiTest.java -> test_api.py (integration)
     com/example/gui/LoginTest.java -> test_login.py (unit, with mocks)
     com/example/gui/NavigationTest.java -> test_navigation.py (unit, with mocks)
   ```

4. **Execute migration:**
   ```bash
   python cli.py migrate --config brobot_migration.json
   ```

5. **Validate results:**
   ```bash
   python cli.py validate qontinui/tests/migrated --report-file validation_report.json
   ```

6. **Generate comprehensive report:**
   ```bash
   python cli.py report qontinui/tests/migrated --format html --output migration_report.html \
     --include-coverage --include-diagnostics
   ```

### Expected Output Structure
```
qontinui/tests/migrated/
├── com/example/unit/
│   ├── test_calculator.py
│   ├── test_string_utils.py
│   └── test_validation.py
├── com/example/integration/
│   ├── test_database.py
│   └── test_api.py
└── com/example/gui/
    ├── test_login.py
    └── test_navigation.py
```

## Example 5: Handling Migration Issues

### Common Issue: Complex Assertion Migration

**Java Code:**
```java
@Test
public void testComplexAssertion() {
    List<String> expected = Arrays.asList("a", "b", "c");
    List<String> actual = service.getItems();
    
    Assertions.assertAll(
        () -> Assertions.assertEquals(3, actual.size()),
        () -> Assertions.assertTrue(actual.containsAll(expected)),
        () -> Assertions.assertEquals("a", actual.get(0))
    );
}
```

**Initial Migration (may need manual adjustment):**
```python
def test_complex_assertion(self):
    expected = ["a", "b", "c"]
    actual = self.service.get_items()
    
    # TODO: Review complex assertion migration
    assert len(actual) == 3
    assert all(item in actual for item in expected)
    assert actual[0] == "a"
```

**Manual Refinement:**
```python
def test_complex_assertion(self):
    expected = ["a", "b", "c"]
    actual = self.service.get_items()
    
    # Multiple assertions for clear failure messages
    assert len(actual) == 3, f"Expected 3 items, got {len(actual)}"
    assert set(expected).issubset(set(actual)), f"Missing items: {set(expected) - set(actual)}"
    assert actual[0] == "a", f"First item should be 'a', got '{actual[0]}'"
```

### Workflow for Complex Cases

1. **Run migration with detailed diagnostics:**
   ```bash
   python cli.py migrate source target -vv --report-file detailed_report.json
   ```

2. **Review the report for issues:**
   ```bash
   # Check JSON report for translation warnings
   cat detailed_report.json | jq '.failed_migrations'
   ```

3. **Manual review and adjustment:**
   - Open generated Python files
   - Look for `# TODO:` comments
   - Adjust complex patterns manually

4. **Validate after manual changes:**
   ```bash
   python cli.py validate qontinui/tests/migrated
   ```

## Example 6: Incremental Migration Strategy

For large codebases, use an incremental approach:

### Phase 1: Unit Tests Only
```bash
# Migrate only unit tests first
python cli.py migrate brobot/library/src/test qontinui/tests/unit \
  --java-test-patterns "*Test.java" \
  --exclude-patterns "*Integration*" "*IT.java"
```

### Phase 2: Integration Tests
```bash
# Migrate integration tests separately
python cli.py migrate brobot/library-test/src/test/java qontinui/tests/integration
```

### Phase 3: GUI/Mock Tests
```bash
# Migrate GUI tests with mock support
python cli.py migrate brobot/gui-tests/src/test qontinui/tests/gui --enable-mocks
```

### Validation After Each Phase
```bash
# Validate each phase separately
python cli.py validate qontinui/tests/unit --report-file unit_validation.json
python cli.py validate qontinui/tests/integration --report-file integration_validation.json
python cli.py validate qontinui/tests/gui --report-file gui_validation.json
```

## Best Practices from Examples

1. **Always use dry-run first** to preview migrations
2. **Start with simple unit tests** to verify the system works
3. **Use configuration files** for complex migration scenarios
4. **Review generated TODO comments** for manual adjustments needed
5. **Validate frequently** during the migration process
6. **Generate reports** to track progress and identify issues
7. **Migrate incrementally** for large codebases
8. **Manual review** is often needed for complex test patterns

## Troubleshooting Examples

### Issue: Test Not Discovered
```bash
# Check file patterns
ls -la brobot/library/src/test/java/ | grep -i test

# Use custom patterns if needed
python cli.py migrate source target --java-test-patterns "*IT.java" "*TestCase.java"
```

### Issue: Mock Migration Failed
```bash
# Check mock-specific errors
python cli.py migrate source target --enable-mocks -vv 2>&1 | grep -i mock

# Disable mocks if needed and handle manually
python cli.py migrate source target --no-mocks
```

### Issue: Complex Spring Configuration
```bash
# Generate report to see Spring-specific issues
python cli.py migrate source target --report-file spring_issues.json

# Review Spring patterns in the report
cat spring_issues.json | jq '.migration_statistics.spring_patterns'
```

These examples should give you a solid foundation for migrating your Brobot tests to Qontinui. Remember that complex tests may require manual review and adjustment after the automated migration.