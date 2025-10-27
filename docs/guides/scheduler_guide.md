# State-Aware Scheduler Guide

## Overview

The Qontinui State-Aware Scheduler is a powerful scheduling system adapted from Brobot's `StateAwareScheduler`. It allows you to schedule automation processes with state validation, ensuring that scheduled tasks only execute when the application is in the correct state.

## Key Features

- **State Validation**: Ensures required states are active before execution
- **Flexible Checking Modes**: Check all states or only inactive ones
- **Auto-Recovery**: Automatically rebuild states when requirements aren't met
- **Multiple Trigger Types**: Time-based (cron), interval-based, state-based, and manual
- **Iteration Limits**: Control how many times a schedule executes
- **Priority-Based Execution**: Higher priority schedules execute first

## Schedule Configuration

### Basic Structure

```json
{
  "schedules": [
    {
      "id": "schedule-id",
      "name": "Schedule Name",
      "description": "What this schedule does",
      "processId": "process-to-execute",
      "triggerType": "time_based|interval|state_based|manual",
      "scheduleType": "fixed_rate|fixed_delay",

      "requiredStates": ["state1", "state2"],
      "forbiddenStates": ["error-state"],
      "checkMode": "CHECK_ALL|CHECK_INACTIVE_ONLY",
      "rebuildOnMismatch": true,
      "skipIfStatesMissing": false,
      "maxIterations": -1,
      "enabled": true,
      "priority": 10
    }
  ]
}
```

### Trigger Types

#### Time-Based (Cron)
Executes at specific times using cron expressions:

```json
{
  "triggerType": "time_based",
  "cronExpression": "0 9 * * *",
  "startTime": "2025-01-13T09:00:00",
  "endTime": "2025-12-31T23:59:59"
}
```

Common cron expressions:
- `0 9 * * *` - Daily at 9 AM
- `*/30 * * * *` - Every 30 minutes
- `0 0 * * 0` - Weekly on Sunday at midnight
- `0 0 1 * *` - Monthly on the 1st at midnight

#### Interval-Based
Executes repeatedly at fixed intervals:

```json
{
  "triggerType": "interval",
  "intervalSeconds": 300,
  "initialDelaySeconds": 60,
  "scheduleType": "fixed_rate"
}
```

**Schedule Types:**
- `fixed_rate`: Execute every N seconds (scheduleAtFixedRate)
- `fixed_delay`: Wait N seconds between executions (scheduleWithFixedDelay)

#### Manual
Only executes when explicitly triggered:

```json
{
  "triggerType": "manual"
}
```

### State Requirements

#### Required States
States that must be active for the schedule to execute:

```json
{
  "requiredStates": ["main-menu", "logged-in"],
  "checkMode": "CHECK_INACTIVE_ONLY",
  "rebuildOnMismatch": true
}
```

**Check Modes:**
- `CHECK_ALL`: Check all required states every time (thorough but slower)
- `CHECK_INACTIVE_ONLY`: Only check states that aren't currently active (efficient)

**Rebuild on Mismatch:**
- `true`: Automatically rebuild active states if requirements aren't met
- `false`: Fail immediately if requirements aren't met

#### Forbidden States
States that must NOT be active:

```json
{
  "forbiddenStates": ["error-dialog", "loading-screen"]
}
```

If any forbidden state is active, the schedule will not execute.

#### Skip if States Missing
Control behavior when states can't be satisfied:

```json
{
  "skipIfStatesMissing": true
}
```

- `true`: Log warning and skip execution
- `false`: Log error and mark schedule as failed

### Execution Limits

#### Max Iterations
Control how many times a schedule executes:

```json
{
  "maxIterations": 10
}
```

- `-1`: Unlimited (default)
- `> 0`: Execute up to N times, then stop automatically

#### Timeout
Maximum execution time for each run:

```json
{
  "timeoutSeconds": 600
}
```

#### Retry Configuration

```json
{
  "maxRetries": 3,
  "retryDelaySeconds": 5
}
```

### Priority

Schedules with lower priority values execute first:

```json
{
  "priority": 5
}
```

- `0-4`: High priority
- `5-10`: Normal priority
- `11+`: Low priority

## Python API

### Loading Schedules

```python
from qontinui.json_executor import JSONRunner

# Load configuration with schedules
runner = JSONRunner()
runner.load_configuration("config_with_schedules.json")

# Scheduler is automatically initialized if schedules exist
if runner.scheduler_executor:
    print("Scheduler ready")
```

### Starting/Stopping the Scheduler

```python
# Start all enabled schedules
runner.start_scheduler()

# Stop all schedules
runner.stop_scheduler()

# Get scheduler statistics
stats = runner.get_scheduler_statistics()
print(f"Active schedules: {stats['running_schedules']}")
print(f"Total executions: {stats['total_executions']}")
```

### Manual Schedule Execution

```python
# Execute a specific schedule manually
schedule_id = "schedule-daily-check"
success = runner.scheduler_executor.execute_schedule(schedule_id)
```

### Accessing Execution History

```python
# Get all execution records
records = runner.scheduler_executor.get_execution_history()

# Get records for specific schedule
schedule_records = runner.scheduler_executor.get_execution_history("schedule-id")

# Examine a record
for record in records:
    print(f"Schedule: {record.schedule_id}")
    print(f"Status: {record.status}")
    print(f"Duration: {record.duration_seconds()}s")

    # View state checks
    for check in record.state_checks:
        print(f"  Required: {check.required_states}")
        print(f"  Active: {check.active_states}")
        print(f"  Passed: {check.check_passed}")
        if check.states_rebuilt:
            print(f"  Rebuilt: {check.rebuild_success}")
```

### Managing Schedules at Runtime

```python
# Enable/disable schedules
runner.scheduler_executor.enable_schedule("schedule-id")
runner.scheduler_executor.disable_schedule("schedule-id")

# Get schedule details
schedule = runner.scheduler_executor.get_schedule("schedule-id")
print(f"Schedule: {schedule.name}")
print(f"Enabled: {schedule.enabled}")
print(f"Next execution: {schedule.next_execution}")

# Get all schedules
all_schedules = runner.scheduler_executor.get_all_schedules()
```

## Best Practices

### 1. Use CHECK_INACTIVE_ONLY for Efficiency

```json
{
  "checkMode": "CHECK_INACTIVE_ONLY"
}
```

This mode only checks states that aren't already active, improving performance.

### 2. Set Realistic Timeouts

```json
{
  "timeoutSeconds": 300,
  "intervalSeconds": 600
}
```

Ensure timeout is less than interval to prevent overlapping executions.

### 3. Use Forbidden States to Prevent Conflicts

```json
{
  "forbiddenStates": ["backup-in-progress", "update-installing"]
}
```

Prevent schedules from running when certain states are active.

### 4. Limit Long-Running Schedules

```json
{
  "maxIterations": 100,
  "intervalSeconds": 60
}
```

Use `maxIterations` for schedules that shouldn't run forever.

### 5. Handle State Mismatches Gracefully

```json
{
  "skipIfStatesMissing": true,
  "rebuildOnMismatch": true
}
```

Allow schedules to skip execution when states aren't available.

### 6. Use Priority for Critical Tasks

```json
{
  "name": "Critical Health Check",
  "priority": 0,
  "requiredStates": ["main-menu"]
}
```

Higher priority schedules (lower numbers) execute first.

## Common Patterns

### Daily Maintenance Task

```json
{
  "id": "daily-maintenance",
  "name": "Daily Maintenance",
  "processId": "maintenance-process",
  "triggerType": "time_based",
  "cronExpression": "0 2 * * *",
  "requiredStates": ["idle"],
  "forbiddenStates": ["user-active"],
  "maxIterations": 1,
  "priority": 10
}
```

### Continuous Monitoring

```json
{
  "id": "monitor-errors",
  "name": "Error Monitor",
  "processId": "check-errors",
  "triggerType": "interval",
  "intervalSeconds": 60,
  "requiredStates": ["main-menu"],
  "checkMode": "CHECK_INACTIVE_ONLY",
  "maxIterations": -1,
  "priority": 5
}
```

### Limited Retry Pattern

```json
{
  "id": "sync-data",
  "name": "Data Sync",
  "processId": "sync-process",
  "triggerType": "interval",
  "intervalSeconds": 300,
  "maxIterations": 5,
  "skipIfStatesMissing": true,
  "priority": 8
}
```

## Troubleshooting

### Schedule Not Executing

1. **Check if schedule is enabled:**
   ```python
   schedule = runner.scheduler_executor.get_schedule("schedule-id")
   print(schedule.enabled)
   ```

2. **Verify state requirements:**
   ```python
   stats = runner.get_scheduler_statistics()
   print(f"Active states: {stats['active_states']}")
   ```

3. **Check execution history:**
   ```python
   records = runner.scheduler_executor.get_execution_history("schedule-id")
   for record in records[-5:]:  # Last 5 executions
       print(f"Status: {record.status}, Error: {record.error_message}")
   ```

### States Not Being Detected

Enable state rebuilding:
```json
{
  "rebuildOnMismatch": true,
  "checkMode": "CHECK_ALL"
}
```

### Overlapping Executions

Use `fixed_delay` instead of `fixed_rate`:
```json
{
  "scheduleType": "fixed_delay",
  "intervalSeconds": 300
}
```

### Schedule Running Too Many Times

Set iteration limit:
```json
{
  "maxIterations": 10
}
```

## Advanced Features

### Custom State Detection

The scheduler integrates with StateExecutor's state detection:

```python
# State detection happens automatically
# But you can customize thresholds in recognition settings
{
  "settings": {
    "recognition": {
      "defaultThreshold": 0.90,
      "multiScaleSearch": true
    }
  }
}
```

### Execution Statistics

```python
stats = runner.get_scheduler_statistics()
print(f"Total schedules: {stats['total_schedules']}")
print(f"Running: {stats['running_schedules']}")
print(f"Enabled: {stats['enabled_schedules']}")
print(f"Disabled: {stats['disabled_schedules']}")
print(f"Total executions: {stats['total_executions']}")
print(f"Successful: {stats['successful_executions']}")
print(f"Failed: {stats['failed_executions']}")
```

### Cleanup

```python
# Scheduler is automatically shut down on cleanup
runner.cleanup()

# Or explicitly:
runner.stop_scheduler()
runner.scheduler_executor.shutdown()
```

## See Also

- [Brobot StateAwareScheduler](https://github.com/jspinak/brobot) - Original Java implementation
- [Process Configuration](process_guide.md) - How to create processes
- [State Machine Guide](state_machine_guide.md) - Understanding states and transitions
- [Cron Expression Syntax](https://en.wikipedia.org/wiki/Cron) - Cron format reference
