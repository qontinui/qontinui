# Qontinui Test Migration System

## 🎯 Overview

This system helps you migrate Java test suites from Brobot to equivalent Python tests in Qontinui, preserving the model-based GUI automation approach while adapting to Python syntax and testing frameworks.

## ✅ Current Status: Working CLI Available

The **standalone CLI** (`cli_standalone.py`) is fully functional and ready to use for discovering and analyzing your Brobot tests.

## 🚀 Quick Start

### 1. Verify Installation
```bash
cd qontinui/src/qontinui/test_migration
python cli_standalone.py --help
```

### 2. Discover Your Brobot Tests
```bash
# Analyze your Brobot test structure
python cli_standalone.py discover /path/to/brobot/library/src/test

# Save results to file
python cli_standalone.py discover /path/to/brobot/library/src/test --output-file discovery.json
```

### 3. Preview Migration
```bash
# See what would be migrated (safe to run)
python cli_standalone.py migrate /path/to/brobot/tests /path/to/output --dry-run
```

### 4. Create Configuration
```bash
# Create a configuration file for your project
python cli_standalone.py config --create --output my_migration_config.json
```

## 📋 Available Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `discover` | Find and analyze Java tests | `python cli_standalone.py discover /path/to/tests` |
| `migrate` | Preview migration (dry-run) | `python cli_standalone.py migrate src target --dry-run` |
| `validate` | Test Python files with pytest | `python cli_standalone.py validate /path/to/python/tests` |
| `config` | Manage configuration files | `python cli_standalone.py config --create` |

## 🔍 What the Discovery Shows You

The discovery command analyzes your Brobot tests and shows:

- **Test Classification**: Unit vs Integration tests
- **Package Structure**: Java package organization
- **Dependencies**: JUnit, Spring, Brobot, and Mockito imports
- **Mock Usage**: Brobot mock patterns detected
- **File Mapping**: How Java files would map to Python files

### Example Output
```
Discovered 3 test files:
==================================================
1. BrobotGuiTest.java
   Type: integration
   Package: com.example.gui
   Dependencies: 5
   Key dependencies:
     - org.junit.jupiter.api.Test
     - io.github.jspinak.brobot.mock.Mock
     - io.github.jspinak.brobot.actions.BrobotSettings

2. CalculatorTest.java
   Type: unit
   Package: com.example.calculator
   Dependencies: 3
   Key dependencies:
     - org.junit.jupiter.api.Test
     - org.junit.jupiter.api.Assertions
```

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **SETUP_INSTRUCTIONS.md** | Quick setup and troubleshooting |
| **USER_GUIDE.md** | Comprehensive user manual |
| **QUICK_REFERENCE.md** | Command cheat sheet |
| **MIGRATION_EXAMPLES.md** | Step-by-step migration examples |
| **INSTALLATION_GUIDE.md** | Detailed installation instructions |

## 🛠️ What Works Now

✅ **Test Discovery**: Finds and analyzes Java test files  
✅ **Classification**: Identifies unit vs integration tests  
✅ **Dependency Analysis**: Maps Java imports to Python equivalents  
✅ **Configuration Management**: Create and validate config files  
✅ **Test Validation**: Run pytest on existing Python test files  
✅ **Dry Run Preview**: Shows what would be migrated  
✅ **Brobot Pattern Detection**: Identifies Brobot-specific code patterns  
✅ **Spring Integration Detection**: Finds SpringBoot test annotations  

## 🔄 Recommended Migration Workflow

### Phase 1: Analysis
```bash
# 1. Discover your test structure
python cli_standalone.py discover brobot/library/src/test --output-file unit_tests.json
python cli_standalone.py discover brobot/library-test/src/test/java --output-file integration_tests.json

# 2. Review the results to understand your test landscape
```

### Phase 2: Manual Migration
Based on the discovery results, manually migrate tests using the patterns from `MIGRATION_EXAMPLES.md`:

1. **Start with simple unit tests** (no mocks, basic assertions)
2. **Progress to integration tests** (Spring patterns)
3. **Handle Brobot mock tests** (GUI automation patterns)

### Phase 3: Validation
```bash
# Validate your migrated Python tests
python cli_standalone.py validate qontinui/tests/migrated --report-file validation.json
```

## 📊 Migration Patterns

The system recognizes these common patterns:

### Java → Python Conversions
| Java Pattern | Python Equivalent |
|--------------|-------------------|
| `@Test` | `def test_*():` |
| `@BeforeEach` | `def setup_method():` |
| `Assertions.assertEquals(a, b)` | `assert a == b` |
| `@Mock` | `@pytest.fixture` or `Mock()` |

### Brobot → Qontinui Patterns
| Brobot | Qontinui Equivalent |
|--------|---------------------|
| `io.github.jspinak.brobot.mock.Mock` | `qontinui.test_migration.mocks.QontinuiMock` |
| `BrobotSettings` | `QontinuiSettings` |
| `AllStatesInProject` | `StateManager` |

## 🎯 Your Next Steps

1. **Run discovery** on your Brobot test directories
2. **Review the analysis** to understand your test structure
3. **Start manual migration** with simple unit tests
4. **Use the examples** in MIGRATION_EXAMPLES.md as templates
5. **Validate** your Python tests as you create them

## 🔧 Configuration Example

Create `my_config.json`:
```json
{
  "source_directories": [
    "/path/to/brobot/library/src/test/java",
    "/path/to/brobot/library-test/src/test/java"
  ],
  "target_directory": "/path/to/qontinui/tests/migrated",
  "preserve_structure": true,
  "enable_mock_migration": true,
  "diagnostic_level": "detailed",
  "java_test_patterns": ["*Test.java", "*Tests.java"],
  "exclude_patterns": ["*/target/*", "*/build/*"]
}
```

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**: Use `cli_standalone.py` instead of `cli.py`
2. **No Tests Found**: Check file patterns and paths
3. **Permission Issues**: Ensure read access to Brobot directories

### Getting Help

- Check **SETUP_INSTRUCTIONS.md** for quick fixes
- Review **USER_GUIDE.md** for detailed troubleshooting
- Run with verbose output: `python cli_standalone.py discover /path/to/tests -v`

## 🎉 Success Story

The CLI successfully:
- ✅ Discovered 3 test files in the demo
- ✅ Correctly classified unit vs integration tests
- ✅ Identified Brobot mock usage patterns
- ✅ Mapped Java dependencies to Python equivalents
- ✅ Generated preview of migration targets

You now have a working tool to analyze your Brobot test structure and guide your manual migration process!

---

**Ready to start?** Run the discovery command on your Brobot tests and see what you're working with!