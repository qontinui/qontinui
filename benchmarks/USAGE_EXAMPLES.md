# Performance Benchmarks Usage Examples

This guide provides practical examples for using the Qontinui performance benchmarks.

## Quick Start

### 1. Run All Benchmarks

The easiest way to get started:

```bash
cd /mnt/c/Users/jspin/Documents/qontinui_parent/qontinui
python -m benchmarks.run_all_benchmarks
```

**Output:**
- Console output with real-time progress
- JSON report: `/tmp/benchmark_results.json`
- HTML report: `/tmp/benchmark_report.html`

### 2. View Results

Open the HTML report in your browser:

```bash
# On WSL/Linux
xdg-open /tmp/benchmark_report.html

# On Windows
start /tmp/benchmark_report.html
```

Or view the JSON results:

```bash
cat /tmp/benchmark_results.json | python -m json.tool
```

## Individual Benchmarks

### Configuration Loading Benchmark

Test how fast configurations are loaded and parsed:

```bash
python -m benchmarks.benchmark_config_loading
```

**Example Output:**
```
============================================================
Benchmarking small config: 10 workflows, 5 actions
============================================================
File size:           12.3 KB
Parse time:          87.2 ms
Target time:         100.0 ms
Actions/second:      574
Memory used:         2.34 MB
Peak memory:         3.12 MB
Grade:               Excellent
```

**What to Look For:**
- Parse time should be < 1 second for typical configs
- Memory usage should scale linearly with config size
- Grade should be "Excellent" or "Good"

### Workflow Execution Benchmark

Test workflow orchestration overhead:

```bash
python -m benchmarks.benchmark_workflow_execution
```

**Example Output:**
```
============================================================
Benchmarking simple workflow: 5 actions
============================================================
Execution time:      412.3 ms
Expected time:       5.0 ms
Overhead:            407.3 ms (8146.0%)
Overhead/action:     81.46 ms [Excellent]
Target time:         500.0 ms
Grade:               Excellent
```

**What to Look For:**
- Overhead per action should be < 100ms
- Grade should be "Excellent" or "Good"
- Parallel workflows should show speedup

### Image Finding Benchmark

Test template matching performance:

```bash
python -m benchmarks.benchmark_image_finding
```

**Example Output:**
```
============================================================
Benchmarking best_case: will_find=True, timeout=3.0s
============================================================
Found:               True
Execution time:      156.2 ms
Confidence:          0.95
Target time:         500.0 ms
Grade:               Excellent
```

**What to Look For:**
- Best case find should be < 200ms
- Pattern matching FPS should be > 10
- Worst case timeout should be close to timeout setting

### Export/Import Benchmark

Test configuration serialization:

```bash
python -m benchmarks.benchmark_export_import
```

**Example Output:**
```
============================================================
Benchmarking small config export: 10 workflows, 5 actions
============================================================
Export time:         54.2 ms
File size:           12.3 KB
Memory used:         1.23 MB
Peak memory:         2.34 MB
Target time:         100.0 ms
Grade:               Excellent
```

**What to Look For:**
- Export/import should be < 2 seconds for large configs
- Round-trip should preserve data accurately
- Memory usage should be reasonable

### Threading Benchmark

Test thread safety overhead:

```bash
python -m benchmarks.benchmark_threading
```

**Example Output:**
```
============================================================
Benchmarking lock overhead: 100000 operations
============================================================
Without lock:        45.2 ms
With lock:           52.3 ms
Overhead:            7.1 ms (15.7%)
Overhead/operation:  0.071 us
Grade:               Good
```

**What to Look For:**
- Threading overhead should be < 10%
- Lock acquisition should be < 1ms
- No race conditions or deadlocks

## Regression Testing

### 1. Create Baseline

Run benchmarks and save as baseline:

```bash
python -m benchmarks.run_all_benchmarks
cp /tmp/benchmark_results.json /tmp/benchmark_results_baseline.json
```

### 2. Make Changes

Make your code changes, then run benchmarks again:

```bash
python -m benchmarks.run_all_benchmarks
```

### 3. Check for Regressions

Compare current results against baseline:

```bash
python -m benchmarks.check_regression
```

**Example Output:**
```
================================================================================
                         PERFORMANCE REGRESSION CHECK
================================================================================

REGRESSIONS DETECTED:
--------------------------------------------------------------------------------
Suite                Test                 Metric               Regression
--------------------------------------------------------------------------------
configuration_loading large               parse_time_ms        +15.3%
workflow_execution   complex             execution_time_ms    +12.7%

Total regressions: 2

IMPROVEMENTS DETECTED:
--------------------------------------------------------------------------------
Suite                Test                 Metric               Improvement
--------------------------------------------------------------------------------
image_finding        best_case           execution_time_ms    -8.2%

Total improvements: 1

================================================================================
```

**Return Codes:**
- `0` - No regressions detected
- `1` - Regressions detected (fails CI/CD)

## Integration Examples

### CI/CD Pipeline

Add to `.github/workflows/benchmarks.yml`:

```yaml
name: Performance Benchmarks

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run benchmarks
        run: |
          cd qontinui
          python -m benchmarks.run_all_benchmarks

      - name: Check for regressions
        run: |
          cd qontinui
          python -m benchmarks.check_regression

      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: |
            /tmp/benchmark_results.json
            /tmp/benchmark_report.html
```

### Pre-commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash

echo "Running performance benchmarks..."

cd qontinui
python -m benchmarks.run_all_benchmarks

if [ $? -ne 0 ]; then
    echo "ERROR: Benchmarks failed!"
    exit 1
fi

echo "Checking for performance regressions..."
python -m benchmarks.check_regression

if [ $? -ne 0 ]; then
    echo "WARNING: Performance regressions detected!"
    echo "Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

echo "Benchmarks passed!"
```

### Python Script Integration

Use benchmarks in your own scripts:

```python
from benchmarks import benchmark_config_loading

# Run specific benchmark
results = benchmark_config_loading.benchmark_config_loading(
    config_size="large",
    num_workflows=100,
    num_actions=10
)

# Check results
if results['grade'] in ['Excellent', 'Good']:
    print("Performance is acceptable")
else:
    print(f"WARNING: Performance is {results['grade']}")
    print(f"Parse time: {results['parse_time_ms']:.1f}ms")
```

## Interpreting Results

### Grade Meanings

- **Excellent**: Within target performance (100%)
- **Good**: Within 1.5x target (67-100% of target)
- **Fair**: Within 2x target (50-67% of target)
- **Poor**: Within 5x target (20-50% of target)
- **Critical**: Exceeds 5x target (< 20% of target)

### Common Issues

#### "Poor" or "Critical" Grades

**Possible Causes:**
- System under heavy load
- Running on slow hardware
- Disk I/O bottleneck
- Memory pressure

**Solutions:**
- Run benchmarks when system is idle
- Close unnecessary applications
- Use SSD instead of HDD
- Increase available RAM

#### High Variation Between Runs

**Possible Causes:**
- Background processes
- CPU throttling
- Thermal throttling
- Cache effects

**Solutions:**
- Run multiple times and average results
- Disable background processes
- Ensure adequate cooling
- Warm up before benchmarking

#### Memory Usage Growing Over Time

**Possible Causes:**
- Memory leak
- Improper cleanup
- Cache not being cleared

**Solutions:**
- Check for memory leaks with tracemalloc
- Ensure proper cleanup in finally blocks
- Clear caches between operations

## Advanced Usage

### Custom Configurations

Test with your own configuration:

```python
from benchmarks.benchmark_config_loading import ConfigParser

# Parse your own config
parser = ConfigParser()
config = parser.parse_file("/path/to/your/config.json")

# Measure performance
import time
start = time.time()
config = parser.parse_file("/path/to/your/config.json")
elapsed = time.time() - start

print(f"Parse time: {elapsed * 1000:.1f}ms")
```

### Profiling Hot Spots

Use cProfile to find bottlenecks:

```bash
python -m cProfile -o profile_output.prof -m benchmarks.run_all_benchmarks

# Analyze results
python -m pstats profile_output.prof
>>> sort cumulative
>>> stats 20
```

### Memory Profiling

Use memory_profiler to track memory usage:

```bash
pip install memory_profiler

python -m memory_profiler benchmarks/benchmark_config_loading.py
```

## Tips and Best Practices

### 1. Run Benchmarks Regularly

- Before and after major changes
- As part of CI/CD pipeline
- Weekly to track trends

### 2. Keep Baseline Updated

- Update baseline after intentional performance changes
- Track baseline history in version control
- Document performance targets

### 3. Focus on Critical Paths

- Prioritize benchmarks for frequently-used features
- Optimize hot paths first
- Consider user experience

### 4. Measure Real-World Scenarios

- Test with realistic configurations
- Use actual workflow patterns
- Consider edge cases

### 5. Document Performance Changes

- Record why performance changed
- Note intentional vs unintentional changes
- Track performance over time

## Troubleshooting

### Import Errors

```
ModuleNotFoundError: No module named 'benchmarks'
```

**Solution:**
```bash
# Make sure you're in the right directory
cd /mnt/c/Users/jspin/Documents/qontinui_parent/qontinui

# Run as module
python -m benchmarks.run_all_benchmarks
```

### Permission Errors

```
PermissionError: [Errno 13] Permission denied: '/tmp/benchmark_results.json'
```

**Solution:**
```bash
# Check permissions
ls -l /tmp/benchmark_results.json

# Remove old file
rm /tmp/benchmark_results.json

# Try again
python -m benchmarks.run_all_benchmarks
```

### Out of Memory

```
MemoryError: Unable to allocate array
```

**Solution:**
- Reduce config size in benchmarks
- Close other applications
- Run benchmarks individually
- Increase swap space

## Getting Help

If you encounter issues:

1. Check the README: `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/benchmarks/README.md`
2. Check main documentation: `/mnt/c/Users/jspin/Documents/qontinui_parent/PERFORMANCE_BENCHMARKS.md`
3. Review benchmark source code for implementation details
4. Check existing performance tests: `tests/integration/test_performance_regression.py`

## Version History

- 2025-10-29: Initial usage examples created
