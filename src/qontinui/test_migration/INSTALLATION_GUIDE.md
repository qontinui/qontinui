# Installation and Setup Guide

This guide will help you set up and configure the Qontinui Test Migration System in your development environment.

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: At least 4GB RAM (8GB recommended for large test suites)
- **Disk Space**: Sufficient space for both source and target test files

### Required Software
- **Python** with pip package manager
- **Git** (for version control)
- **Text Editor/IDE** (VS Code, PyCharm, etc.)

### Python Dependencies
The migration system uses mostly built-in Python libraries:
- `pathlib` (built-in)
- `argparse` (built-in) 
- `json` (built-in)
- `logging` (built-in)
- `subprocess` (built-in)
- `tempfile` (built-in)
- `dataclasses` (built-in in Python 3.7+)

Optional dependencies for enhanced functionality:
- `pytest` (for running migrated tests)
- `PyYAML` (for YAML report output)
- `reportlab` (for PDF report generation)

## Installation Steps

### Step 1: Verify Python Installation

```bash
# Check Python version
python --version
# or
python3 --version

# Should show Python 3.8 or higher
```

If Python is not installed or version is too old:
- **Windows**: Download from [python.org](https://python.org)
- **macOS**: Use Homebrew: `brew install python3`
- **Linux**: Use package manager: `sudo apt install python3 python3-pip`

### Step 2: Navigate to Migration Tool

```bash
# Navigate to your Qontinui project
cd /path/to/your/qontinui_parent_directory/qontinui

# Go to the migration tool directory
cd src/qontinui/test_migration
```

### Step 3: Verify Installation

```bash
# Test that the CLI works
python cli.py --help
```

You should see the help output with available commands.

### Step 4: Install Optional Dependencies

```bash
# Install pytest for running migrated tests
pip install pytest

# Install PyYAML for YAML output (optional)
pip install PyYAML

# Install reportlab for PDF reports (optional)
pip install reportlab
```

### Step 5: Set Up CLI Launcher (Optional)

For easier access, create launcher scripts:

```bash
# Run the setup script
python setup_cli.py
```

This creates:
- **Linux/macOS**: `qontinui-test-migration` executable script
- **Windows**: `qontinui-test-migration.bat` batch file

## Configuration

### Create Default Configuration

```bash
# Create a default configuration file
python cli.py config --create --output migration_config.json
```

### Edit Configuration for Your Environment

Edit `migration_config.json` to match your project structure:

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
  "parallel_execution": true,
  "comparison_mode": "behavioral",
  "java_test_patterns": [
    "*Test.java",
    "*Tests.java",
    "Test*.java"
  ],
  "exclude_patterns": [
    "*/target/*",
    "*/build/*",
    "*/.git/*",
    "*/node_modules/*"
  ]
}
```

### Validate Configuration

```bash
# Check that your configuration is valid
python cli.py config --validate --input migration_config.json
```

## Environment Setup

### Directory Structure

Organize your directories for efficient migration:

```
your-project/
├── brobot/                          # Original Brobot project
│   ├── library/src/test/java/       # Unit tests
│   └── library-test/src/test/java/  # Integration tests
├── qontinui/                        # Qontinui project
│   ├── src/qontinui/test_migration/ # Migration tool (this directory)
│   └── tests/                       # Target for migrated tests
│       ├── migrated/                # Migrated tests
│       ├── unit/                    # Unit tests
│       └── integration/             # Integration tests
└── migration_reports/               # Migration reports and logs
```

### Create Target Directories

```bash
# Create directories for migrated tests
mkdir -p qontinui/tests/migrated
mkdir -p qontinui/tests/unit  
mkdir -p qontinui/tests/integration
mkdir -p migration_reports
```

## Verification

### Test with Sample Data

Create a simple test to verify everything works:

```bash
# Create a temporary test directory
mkdir -p /tmp/test_migration_sample
cd /tmp/test_migration_sample

# Create a sample Java test
cat > SampleTest.java << 'EOF'
package com.example;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assertions;

public class SampleTest {
    @Test
    public void testSample() {
        Assertions.assertEquals(1, 1);
    }
}
EOF

# Create target directory
mkdir -p python_tests

# Go back to migration tool
cd /path/to/qontinui/src/qontinui/test_migration

# Test migration (dry run)
python cli.py migrate /tmp/test_migration_sample python_tests --dry-run

# If successful, you should see output like:
# Found 1 test files:
#   SampleTest.java -> test_sample.py (unit)
```

### Run System Health Check

```bash
# Run the demo workflow to verify all components work
python demo_workflow.py
```

This should complete successfully and show:
```
Workflow demonstration completed successfully!
The migration system is ready for use.
```

## IDE Integration

### VS Code Setup

If you use VS Code, create a workspace configuration:

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "/usr/bin/python3",
    "python.terminal.activateEnvironment": true,
    "files.associations": {
        "*.md": "markdown"
    },
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true
}
```

### PyCharm Setup

1. Open PyCharm
2. File → Open → Select your qontinui directory
3. Configure Python interpreter: File → Settings → Project → Python Interpreter
4. Add the migration tool directory to Python path

## Troubleshooting Installation

### Common Issues

#### 1. Python Not Found
```bash
# Error: 'python' is not recognized
# Solution: Use python3 instead
python3 cli.py --help

# Or add alias to your shell profile
echo "alias python=python3" >> ~/.bashrc
source ~/.bashrc
```

#### 2. Permission Denied (Linux/macOS)
```bash
# Error: Permission denied
# Solution: Make script executable
chmod +x qontinui-test-migration

# Or run with python explicitly
python cli.py --help
```

#### 3. Import Errors
```bash
# Error: ModuleNotFoundError
# Solution: Ensure you're in the correct directory
cd qontinui/src/qontinui/test_migration
python cli.py --help

# Or use absolute paths
python /full/path/to/qontinui/src/qontinui/test_migration/cli.py --help
```

#### 4. Path Issues on Windows
```bash
# Use forward slashes or escape backslashes
python cli.py migrate C:/path/to/brobot/tests C:/path/to/qontinui/tests

# Or use raw strings in configuration files
"source_directories": ["C:\\path\\to\\brobot\\tests"]
```

### Debugging Installation

```bash
# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Check current directory
pwd

# List files in migration directory
ls -la qontinui/src/qontinui/test_migration/

# Test basic imports
python -c "import pathlib, json, argparse; print('Basic imports work')"
```

## Performance Optimization

### For Large Test Suites

```bash
# Enable parallel processing
python cli.py migrate source target --parallel

# Use configuration file for complex setups
python cli.py migrate --config migration_config.json
```

### Memory Management

For very large test suites (1000+ files):

1. **Increase Python memory limit** (if needed):
   ```bash
   export PYTHONHASHSEED=0
   ulimit -v 8388608  # 8GB virtual memory limit
   ```

2. **Process in batches**:
   ```bash
   # Migrate unit tests first
   python cli.py migrate brobot/unit qontinui/tests/unit
   
   # Then integration tests
   python cli.py migrate brobot/integration qontinui/tests/integration
   ```

## Backup and Safety

### Before Migration

```bash
# Create backup of existing tests
cp -r qontinui/tests qontinui/tests.backup.$(date +%Y%m%d)

# Use version control
cd qontinui
git add tests/
git commit -m "Backup before migration"
```

### During Migration

```bash
# Always use dry-run first
python cli.py migrate source target --dry-run

# Save migration reports
python cli.py migrate source target --report-file migration_$(date +%Y%m%d).json
```

## Next Steps

After successful installation:

1. **Read the User Guide**: `USER_GUIDE.md`
2. **Try the Quick Reference**: `QUICK_REFERENCE.md`  
3. **Follow Migration Examples**: `MIGRATION_EXAMPLES.md`
4. **Start with a small test migration** to familiarize yourself with the system
5. **Create your project-specific configuration file**

## Support

If you encounter issues during installation:

1. **Check the logs** for detailed error messages
2. **Verify file paths** and permissions
3. **Test with the sample data** provided above
4. **Review the troubleshooting section** in the User Guide

The migration system is designed to be self-contained and should work with a standard Python installation. Most issues are related to file paths or Python environment configuration.