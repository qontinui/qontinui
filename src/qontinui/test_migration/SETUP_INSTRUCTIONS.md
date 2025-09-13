# Quick Setup Instructions

## Issue Resolution

The original CLI had import dependency issues. I've created a **standalone version** that works without complex dependencies.

## Working CLI

Use `cli_standalone.py` instead of `cli.py`:

```bash
cd qontinui/src/qontinui/test_migration
python cli_standalone.py --help
```

## Quick Test

1. **Verify it works:**
   ```bash
   python cli_standalone.py --help
   ```

2. **Create a configuration file:**
   ```bash
   python cli_standalone.py config --create --output my_config.json
   ```

3. **Test discovery with your Brobot tests:**
   ```bash
   python cli_standalone.py discover /path/to/your/brobot/tests -v
   ```

4. **Preview migration (safe to run):**
   ```bash
   python cli_standalone.py migrate /path/to/brobot/tests /path/to/output --dry-run -v
   ```

## Available Commands

### `discover` - Find Java Tests
```bash
# Basic discovery
python cli_standalone.py discover /path/to/brobot/tests

# With detailed output
python cli_standalone.py discover /path/to/brobot/tests -v --output-format json

# Save results to file
python cli_standalone.py discover /path/to/brobot/tests --output-file discovery.json
```

### `migrate` - Convert Tests (Limited)
```bash
# Dry run (preview only)
python cli_standalone.py migrate /path/to/brobot/tests /path/to/output --dry-run

# Note: Full migration requires additional components
# This version focuses on discovery and analysis
```

### `validate` - Test Python Files
```bash
# Validate existing Python tests
python cli_standalone.py validate /path/to/python/tests

# With report
python cli_standalone.py validate /path/to/python/tests --report-file validation.json
```

### `config` - Manage Settings
```bash
# Create default config
python cli_standalone.py config --create --output config.json

# Validate config
python cli_standalone.py config --validate --input config.json
```

## What Works Now

✅ **Test Discovery**: Finds and analyzes Java test files  
✅ **Classification**: Identifies unit vs integration tests  
✅ **Dependency Analysis**: Maps Java imports  
✅ **Configuration Management**: Create and validate config files  
✅ **Test Validation**: Run pytest on Python test files  
✅ **Dry Run**: Preview what would be migrated  

## What Needs Full System

❌ **Complete Migration**: Java → Python code translation  
❌ **Mock Conversion**: Brobot → Qontinui mock migration  
❌ **Spring Integration**: SpringBoot test patterns  
❌ **Advanced Reporting**: HTML/PDF reports  

## Recommended Workflow

1. **Start with Discovery:**
   ```bash
   python cli_standalone.py discover /path/to/brobot/library/src/test -v
   ```

2. **Analyze the Results:**
   - See what tests are found
   - Check test types (unit vs integration)
   - Review dependencies

3. **Use Dry Run:**
   ```bash
   python cli_standalone.py migrate /path/to/brobot/tests /path/to/output --dry-run
   ```

4. **Manual Migration:**
   - Use the discovery results to understand your test structure
   - Manually migrate simple tests first
   - Use the patterns shown in MIGRATION_EXAMPLES.md

## Example Configuration

Edit `my_config.json` to match your project:

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
  "parallel_execution": false,
  "comparison_mode": "behavioral",
  "java_test_patterns": [
    "*Test.java",
    "*Tests.java",
    "Test*.java"
  ],
  "exclude_patterns": [
    "*/target/*",
    "*/build/*",
    "*/.git/*"
  ]
}
```

## Next Steps

1. **Use the standalone CLI** to discover and analyze your Brobot tests
2. **Review the discovery results** to understand your test structure
3. **Start manual migration** of simple tests using the patterns in the documentation
4. **Use the validation command** to test your migrated Python files

The standalone version gives you the core functionality to analyze your Brobot tests and start the migration process manually with full understanding of what needs to be converted.