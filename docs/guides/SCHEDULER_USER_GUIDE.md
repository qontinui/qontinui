# State-Aware Scheduler User Guide

## Overview

The State-Aware Scheduler is a powerful automation feature in qontinui that allows you to schedule processes to run automatically based on time, intervals, or state conditions. This guide will help you understand and use the scheduler effectively.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Creating Schedules](#creating-schedules)
3. [Trigger Types](#trigger-types)
4. [Check Modes](#check-modes)
5. [Schedule Types](#schedule-types)
6. [Managing Schedules](#managing-schedules)
7. [Monitoring Execution](#monitoring-execution)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Examples](#examples)

## Core Concepts

### What is a Schedule?

A schedule is a configuration that tells qontinui when and how to run a specific process automatically. Each schedule consists of:

- **Trigger**: When the schedule should run (time, interval, or state-based)
- **Process**: What process to execute
- **Check Mode**: How to verify states before execution
- **Execution Limits**: How many times to run the process
- **Failure Handling**: What to do when execution fails

### State-Aware Execution

Unlike simple schedulers that just run tasks at set times, qontinui's scheduler is **state-aware**. This means:

- Schedules can check if specific UI states are active before running
- Processes can wait for the right states to appear
- Failed executions can trigger state rebuilding
- State changes can trigger schedule execution

## Creating Schedules

### Via Frontend UI

1. Open the **Automation Builder** in qontinui-web
2. Navigate to the **Scheduler** tab
3. Click **Create Schedule**
4. Fill in the schedule details:
   - **Name**: Descriptive name for the schedule
   - **Description**: Optional details about what it does
   - **Process**: Select which process to run
   - **Trigger Type**: Choose when it should run
   - **Check Mode**: Choose how to verify states
   - Configure trigger-specific settings
5. Click **Save**

### Via JSON Configuration

Add a schedule object to your configuration's `schedules` array:

```json
{
  "schedules": [
    {
      "id": "my-schedule",
      "name": "My Automated Task",
      "description": "Runs daily at 9 AM",
      "processId": "my-process",
      "triggerType": "TIME",
      "cronExpression": "0 9 * * *",
      "checkMode": "CHECK_ALL",
      "scheduleType": "FIXED_RATE",
      "maxIterations": 5,
      "stateCheckDelaySeconds": 2.0,
      "stateRebuildDelaySeconds": 1.0,
      "failureThreshold": 3,
      "enabled": true
    }
  ]
}
```

## Trigger Types

### 1. TIME (Cron-Based)

Run at specific times using cron expressions.

**Use Cases:**
- Daily reports
- Nightly backups
- Weekly maintenance
- Monthly tasks

**Configuration:**
```json
{
  "triggerType": "TIME",
  "cronExpression": "0 9 * * *"
}
```

**Cron Expression Examples:**
- `0 9 * * *` - Every day at 9:00 AM
- `0 */6 * * *` - Every 6 hours
- `0 9 * * 1` - Every Monday at 9:00 AM
- `0 0 1 * *` - First day of every month at midnight
- `0 9-17 * * 1-5` - Every hour from 9 AM to 5 PM on weekdays

**Cron Format:**
```
┌───────────── minute (0 - 59)
│ ┌───────────── hour (0 - 23)
│ │ ┌───────────── day of month (1 - 31)
│ │ │ ┌───────────── month (1 - 12)
│ │ │ │ ┌───────────── day of week (0 - 6) (Sunday to Saturday)
│ │ │ │ │
* * * * *
```

### 2. INTERVAL

Run repeatedly at fixed intervals.

**Use Cases:**
- Polling status
- Health checks
- Periodic monitoring
- Data synchronization

**Configuration:**
```json
{
  "triggerType": "INTERVAL",
  "intervalSeconds": 300
}
```

**Examples:**
- `60` - Every minute
- `300` - Every 5 minutes
- `3600` - Every hour
- `86400` - Every 24 hours

### 3. STATE

Run when a specific state becomes active.

**Use Cases:**
- Error handling
- Reactive automation
- Event-driven workflows
- Conditional processing

**Configuration:**
```json
{
  "triggerType": "STATE",
  "triggerState": "error-dialog"
}
```

**How it Works:**
1. Scheduler continuously monitors for the trigger state
2. When the state becomes active, the process is executed
3. After execution, the scheduler continues monitoring

### 4. MANUAL

Run only when manually triggered.

**Use Cases:**
- One-time tasks
- User-initiated automation
- Testing and debugging
- On-demand operations

**Configuration:**
```json
{
  "triggerType": "MANUAL"
}
```

## Check Modes

Check modes determine how the scheduler verifies states before execution.

### CHECK_ALL

Checks all states in the state structure before every execution.

**When to Use:**
- You want comprehensive state validation
- State accuracy is critical
- Performance is not a concern

**Behavior:**
- Rebuilds entire state structure
- Checks all configured states
- Most accurate but slowest

### CHECK_INACTIVE_ONLY

Only checks states that aren't currently active.

**When to Use:**
- You want better performance
- Active states are likely still valid
- You have many states to check

**Behavior:**
- Only rebuilds inactive states
- Faster than CHECK_ALL
- Good balance of speed and accuracy

## Schedule Types

### FIXED_RATE

Executes at fixed intervals, regardless of execution time.

**Timing:**
```
Start → Execute (2s) → Wait (8s) → Execute (2s) → Wait (8s) → ...
[Every 10s from start time]
```

**Use When:**
- You need precise timing
- Execution time is predictable
- Overlap is acceptable

### FIXED_DELAY

Waits for a fixed delay after each execution completes.

**Timing:**
```
Start → Execute (2s) → Wait (10s) → Execute (2s) → Wait (10s) → ...
[10s delay between executions]
```

**Use When:**
- You want to avoid overlap
- Execution time varies
- Resource contention is a concern

## Managing Schedules

### Enabling/Disabling Schedules

**Via UI:**
1. Open ScheduleManager
2. Toggle the enable switch next to the schedule

**Via JSON:**
```json
{
  "enabled": true  // or false
}
```

**When Disabled:**
- Schedule will not trigger
- Existing executions continue
- Statistics are retained

### Editing Schedules

**Via UI:**
1. Open ScheduleManager
2. Click edit button on the schedule
3. Modify settings
4. Click Save

**Important:**
- Changes take effect immediately
- Running executions are not affected
- Trigger recalculation happens automatically

### Deleting Schedules

**Via UI:**
1. Open ScheduleManager
2. Click delete button on the schedule
3. Confirm deletion

**Warning:**
- Deletion is permanent
- Execution history is preserved
- Running executions are stopped

## Monitoring Execution

### Execution Statistics

View scheduler statistics in the dashboard:

- **Total Schedules**: Number of configured schedules
- **Active Schedules**: Number of enabled schedules
- **Total Executions**: Total execution count
- **Success Rate**: Percentage of successful executions

### Execution History

View detailed execution records:

**Information Available:**
- Start and end times
- Success/failure status
- Number of iterations
- Error messages
- Custom metadata

**Filtering Options:**
- By schedule
- By status (success/failed)
- By date range (in future versions)

### Schedule Details

View per-schedule analytics:

- Recent execution history
- Success rate for this schedule
- Average iteration count
- Total execution count

## Best Practices

### 1. Naming Conventions

Use descriptive names that indicate:
- What the schedule does
- When it runs
- Why it exists

**Good Examples:**
- "Daily Morning Health Check"
- "Error Dialog Auto-Recovery"
- "Hourly Status Polling"

**Bad Examples:**
- "Schedule 1"
- "Test"
- "Automation"

### 2. Setting Iteration Limits

**Recommended Values:**
- **Quick checks**: 1-3 iterations
- **Moderate tasks**: 5-10 iterations
- **Complex workflows**: 10-20 iterations
- **Continuous monitoring**: null (unlimited)

**Why Limit Iterations:**
- Prevents infinite loops
- Controls execution time
- Manages resource usage

### 3. Failure Thresholds

Set appropriate failure thresholds:
- **Critical schedules**: 1-2 failures
- **Regular schedules**: 3-5 failures
- **Polling schedules**: 5-10 failures

### 4. State Check Delays

Configure delays based on your application:
- **Fast UI**: 1-2 seconds
- **Normal UI**: 2-3 seconds
- **Slow UI**: 3-5 seconds

### 5. Scheduling Strategy

**Don't Over-Schedule:**
- Avoid too many schedules running simultaneously
- Use appropriate intervals
- Consider system resources

**Use Appropriate Triggers:**
- Time-based for predictable tasks
- Interval-based for continuous monitoring
- State-based for reactive automation
- Manual for one-time operations

### 6. Testing Schedules

Before enabling in production:
1. Test with MANUAL trigger first
2. Verify process execution
3. Check execution records
4. Monitor resource usage
5. Enable automatic triggering

## Troubleshooting

### Schedule Not Triggering

**Possible Causes:**
1. Schedule is disabled
2. Cron expression is invalid
3. Scheduler not started
4. Process doesn't exist

**Solutions:**
- Check enabled status
- Validate cron expression
- Verify scheduler is running via runner
- Confirm process ID is correct

### Execution Fails Immediately

**Possible Causes:**
1. Required states not present
2. Process has errors
3. Images not found
4. Timeout too short

**Solutions:**
- Check state configuration
- Test process manually
- Verify all images exist
- Increase timeout values

### Schedule Executes Too Often

**Possible Causes:**
1. Interval too short
2. FIXED_RATE vs FIXED_DELAY confusion
3. Multiple schedules for same process

**Solutions:**
- Increase interval
- Use FIXED_DELAY for spacing
- Consolidate duplicate schedules

### High Resource Usage

**Possible Causes:**
1. Too many active schedules
2. CHECK_ALL mode on all schedules
3. No iteration limits
4. State check delays too short

**Solutions:**
- Reduce number of active schedules
- Use CHECK_INACTIVE_ONLY where possible
- Set reasonable iteration limits
- Increase state check delays

### Execution Records Not Appearing

**Possible Causes:**
1. Execution not completing
2. Configuration not saved
3. API connection issue

**Solutions:**
- Check execution status
- Save and export configuration
- Verify API connectivity

## Examples

### Example 1: Daily Morning Report

Generate a report every weekday at 9 AM:

```json
{
  "id": "morning-report",
  "name": "Daily Morning Report",
  "description": "Generates morning status report",
  "processId": "generate-report-process",
  "triggerType": "TIME",
  "cronExpression": "0 9 * * 1-5",
  "checkMode": "CHECK_ALL",
  "scheduleType": "FIXED_RATE",
  "maxIterations": 3,
  "stateCheckDelaySeconds": 2.0,
  "stateRebuildDelaySeconds": 1.0,
  "failureThreshold": 2,
  "enabled": true
}
```

### Example 2: Status Polling

Check application status every 5 minutes:

```json
{
  "id": "status-poll",
  "name": "Status Polling",
  "description": "Monitors application health",
  "processId": "health-check-process",
  "triggerType": "INTERVAL",
  "intervalSeconds": 300,
  "checkMode": "CHECK_INACTIVE_ONLY",
  "scheduleType": "FIXED_DELAY",
  "maxIterations": null,
  "stateCheckDelaySeconds": 2.0,
  "stateRebuildDelaySeconds": 1.0,
  "failureThreshold": 5,
  "enabled": true
}
```

### Example 3: Error Recovery

Automatically handle error dialogs when they appear:

```json
{
  "id": "error-recovery",
  "name": "Auto Error Recovery",
  "description": "Handles error dialogs automatically",
  "processId": "handle-error-process",
  "triggerType": "STATE",
  "triggerState": "error-dialog",
  "checkMode": "CHECK_ALL",
  "scheduleType": "FIXED_RATE",
  "maxIterations": 3,
  "stateCheckDelaySeconds": 1.0,
  "stateRebuildDelaySeconds": 0.5,
  "failureThreshold": 3,
  "enabled": true
}
```

### Example 4: Manual Backup

One-time backup that requires manual triggering:

```json
{
  "id": "manual-backup",
  "name": "Manual Database Backup",
  "description": "Backs up database on demand",
  "processId": "backup-process",
  "triggerType": "MANUAL",
  "checkMode": "CHECK_ALL",
  "scheduleType": "FIXED_RATE",
  "maxIterations": 1,
  "stateCheckDelaySeconds": 2.0,
  "stateRebuildDelaySeconds": 1.0,
  "failureThreshold": 1,
  "enabled": false
}
```

## Advanced Topics

### Execution Metadata

Add custom metadata to execution records for tracking:

```python
metadata = {
    "user": "admin",
    "trigger": "manual",
    "environment": "production",
    "version": "1.0.0"
}
```

### Programmatic Schedule Management

Create and manage schedules via code:

```python
from qontinui.scheduling import ScheduleConfig, TriggerType

schedule = ScheduleConfig(
    schedule_id="my-schedule",
    name="My Schedule",
    process_id="my-process",
    trigger_type=TriggerType.TIME,
    cron_expression="0 9 * * *",
    enabled=True
)
```

### Custom State Checks

Implement custom state verification logic:

```python
def custom_state_check(state_structure):
    # Custom validation logic
    return all_required_states_present()
```

## API Reference

### REST API Endpoints

**Get Statistics:**
```
GET /api/v1/scheduler/statistics/{project_id}
```

**Get Status:**
```
GET /api/v1/scheduler/status/{project_id}
```

**Get Execution History:**
```
GET /api/v1/scheduler/executions/{project_id}?schedule_id=X&limit=50
```

**Get Schedule Details:**
```
GET /api/v1/scheduler/schedule/{project_id}/{schedule_id}
```

### Runner Commands

**Start Scheduler:**
```json
{"type": "command", "command": "scheduler_start", "params": {}}
```

**Stop Scheduler:**
```json
{"type": "command", "command": "scheduler_stop", "params": {}}
```

**Get Status:**
```json
{"type": "command", "command": "scheduler_status", "params": {}}
```

**Get Statistics:**
```json
{"type": "command", "command": "scheduler_get_statistics", "params": {}}
```

## Frequently Asked Questions

### Q: Can I have multiple schedules for the same process?

**A:** Yes! You can create multiple schedules that run the same process with different triggers and settings.

### Q: What happens if a schedule triggers while the process is already running?

**A:** Behavior depends on schedule type:
- **FIXED_RATE**: New execution starts immediately (can overlap)
- **FIXED_DELAY**: New execution waits until previous completes

### Q: Can I change a schedule while it's running?

**A:** Yes, but changes only affect future executions, not currently running ones.

### Q: How do I temporarily pause a schedule?

**A:** Disable the schedule via the UI or by setting `"enabled": false` in the configuration.

### Q: What's the maximum number of schedules I can have?

**A:** There's no hard limit, but consider system resources. Monitor CPU and memory usage.

### Q: Can schedules run in parallel?

**A:** Yes, different schedules can run simultaneously. Same schedule instances depend on schedule type.

### Q: How long are execution records kept?

**A:** Execution records are kept indefinitely in the configuration. You can manually clear old records if needed.

## Support and Resources

- **Documentation**: `/docs/` directory in qontinui repository
- **Examples**: `/examples/scheduler_example.json`
- **Issue Tracker**: GitHub Issues
- **Community**: Discord/Forums (if available)

## Version History

- **v1.0.0** (2025-01-15): Initial scheduler implementation
  - Time, Interval, State, and Manual triggers
  - CHECK_ALL and CHECK_INACTIVE_ONLY modes
  - FIXED_RATE and FIXED_DELAY scheduling
  - Full frontend, backend, and runner integration
