# Qontinui Test Migration - Quick Reference

## Essential Commands

### 🚀 Quick Start
```bash
# Navigate to migration tool
cd qontinui/src/qontinui/test_migration

# Preview migration (safe to run)
python cli.py migrate /path/to/brobot/tests /path/to/qontinui/tests --dry-run

# Perform migration
python cli.py migrate /path/to/brobot/tests /path/to/qontinui/tests

# Validate results
python cli.py validate /path/to/qontinui/tests
```

### 📁 Common Paths for Brobot
```bash
# Typical Brobot test locations
brobot/library/src/test/java                    # Unit tests
brobot/library-test/src/test/java              # Integration tests

# Example migration commands
python cli.py migrate brobot/library/src/test qontinui/tests/unit --dry-run
python cli.py migrate brobot/library-test/src/test/java qontinui/tests/integration --dry-run
```

## Command Cheat Sheet

| Command | Purpose | Example |
|---------|---------|---------|
| `migrate` | Convert Java tests to Python | `python cli.py migrate src target` |
| `validate` | Run and check migrated tests | `python cli.py validate target` |
| `report` | Generate migration report | `python cli.py report target --format html` |
| `config` | Manage configuration | `python cli.py config --create` |

## Useful Options

| Option | Purpose | Example |
|--------|---------|---------|
| `--dry-run` | Preview without changes | `migrate src target --dry-run` |
| `-v, -vv, -vvv` | Increase verbosity | `migrate src target -vv` |
| `--no-preserve-structure` | Flat output structure | `migrate src target --no-preserve-structure` |
| `--report-file` | Save detailed report | `migrate src target --report-file report.json` |

## File Patterns

### Java Test Detection
The system automatically finds files matching:
- `*Test.java` (e.g., `CalculatorTest.java`)
- `*Tests.java` (e.g., `DatabaseTests.java`)
- `Test*.java` (e.g., `TestRunner.java`)

### Python Test Output
Generated Python files follow pytest conventions:
- `test_*.py` (e.g., `test_calculator.py`)
- `*_test.py` (e.g., `calculator_test.py`)

## Migration Examples

### Example 1: Basic Unit Test Migration
```bash
# Source: brobot/library/src/test/java/com/example/CalculatorTest.java
# Target: qontinui/tests/migrated/test_calculator.py

python cli.py migrate brobot/library/src/test qontinui/tests/migrated
```

### Example 2: Integration Test Migration
```bash
# Source: brobot/library-test/src/test/java/
# Target: qontinui/tests/integration/

python cli.py migrate brobot/library-test/src/test/java qontinui/tests/integration \
  --preserve-structure --enable-mocks
```

### Example 3: Selective Migration
```bash
# Only migrate specific test files
python cli.py migrate brobot/library/src/test qontinui/tests/migrated \
  --java-test-patterns "*ServiceTest.java"
```

## Validation Workflow

```bash
# 1. Migrate tests
python cli.py migrate brobot/library/src/test qontinui/tests/migrated

# 2. Validate migration
python cli.py validate qontinui/tests/migrated --report-file validation.json

# 3. Generate comprehensive report
python cli.py report qontinui/tests/migrated --format html --output report.html \
  --include-coverage --include-diagnostics
```

## Configuration Quick Setup

### Create Default Config
```bash
python cli.py config --create --output migration_config.json
```

### Sample Configuration
```json
{
  "source_directories": ["brobot/library/src/test", "brobot/library-test/src/test/java"],
  "target_directory": "qontinui/tests/migrated",
  "preserve_structure": true,
  "enable_mock_migration": true,
  "diagnostic_level": "detailed"
}
```

### Use Configuration
```bash
python cli.py migrate --config migration_config.json
```

## Troubleshooting Quick Fixes

### Problem: No tests found
```bash
# Check if files exist
ls -la /path/to/brobot/tests/

# Use dry-run to see what's detected
python cli.py migrate /path/to/brobot/tests /path/to/output --dry-run -v
```

### Problem: Import errors
```bash
# Ensure you're in the right directory
cd qontinui/src/qontinui/test_migration

# Use absolute paths
python cli.py migrate /absolute/path/to/source /absolute/path/to/target
```

### Problem: Translation failures
```bash
# Use verbose mode to see details
python cli.py migrate source target -vv

# Check specific error in report
python cli.py migrate source target --report-file errors.json
```

## Output Formats

| Format | Use Case | Command |
|--------|----------|---------|
| `text` | Console output | `--output-format text` |
| `json` | Programmatic access | `--output-format json` |
| `html` | Visual reports | `--format html` |
| `yaml` | Human-readable data | `--format yaml` |

## Migration Patterns

### Java → Python Conversions

| Java Pattern | Python Equivalent |
|--------------|-------------------|
| `@Test` | `def test_*():` |
| `@BeforeEach` | `def setup_method():` |
| `@AfterEach` | `def teardown_method():` |
| `Assertions.assertEquals(a, b)` | `assert a == b` |
| `Assertions.assertTrue(x)` | `assert x` |
| `@Mock` | `@pytest.fixture` or `Mock()` |

### Brobot → Qontinui Mocks

| Brobot | Qontinui |
|--------|----------|
| `io.github.jspinak.brobot.mock.Mock` | `qontinui.test_migration.mocks.QontinuiMock` |
| `BrobotSettings` | `QontinuiSettings` |
| `AllStatesInProject` | `StateManager` |

## Performance Tips

- Use `--parallel` for large test suites
- Use `--no-preserve-structure` for simpler output
- Migrate in batches for very large codebases
- Use `--dry-run` first to estimate scope

## Quick Health Check

```bash
# Test the system is working
python cli.py --help

# Test with minimal example
mkdir -p /tmp/test_source /tmp/test_target
echo 'public class TestExample { @Test public void test() {} }' > /tmp/test_source/TestExample.java
python cli.py migrate /tmp/test_source /tmp/test_target --dry-run
```

This should show the test being discovered and ready for migration.