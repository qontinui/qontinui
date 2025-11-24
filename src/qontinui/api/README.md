# API Module

## Overview

The `api` module provides REST and WebSocket APIs for workflow execution, state management, and external integration with the qontinui automation framework.

## Purpose

This module enables:

- REST API for workflow execution and control
- WebSocket connections for real-time status updates
- External integration with CI/CD systems
- Remote control of automation workflows
- Execution monitoring and management
- State operation facade for simplified access

## Key Components

### Execution Management

- **ExecutionManager**: Core execution lifecycle management
- **ExecutionOrchestrator**: Coordinates complex multi-step executions
- **ExecutionController**: Controls running executions (pause, resume, stop)
- **ExecutionRegistry**: Tracks active and historical executions
- **ExecutionEventBus**: Event-driven communication for execution events

### API Endpoints

- **execution_api.py**: FastAPI application setup and configuration
- **routers/**: REST endpoint implementations
- **models.py**: API request/response models

### Status & History

- **ExecutionStatusTracker**: Real-time execution status monitoring
- **ExecutionHistory**: Historical execution data and analytics

### Facades

- **StateOperationsFacade**: Simplified interface for state operations

## Usage Pattern

### Starting the API Server

```python
from qontinui.api import create_app

app = create_app()

# Run with uvicorn
# uvicorn qontinui.api.execution_api:app --host 0.0.0.0 --port 8000
```

### Using the Execution Manager

```python
from qontinui.api import ExecutionManager, ExecutionOptions

# Initialize manager
manager = ExecutionManager()

# Execute workflow
result = await manager.execute_workflow(
    workflow_id="login_flow",
    options=ExecutionOptions(
        timeout=30,
        retry_on_failure=True,
        capture_screenshots=True
    )
)

print(f"Execution {result.execution_id}: {result.status}")
```

### Monitoring Execution Events

```python
from qontinui.api import ExecutionManager, ExecutionEventType

manager = ExecutionManager()

# Subscribe to events
def on_event(event):
    if event.type == ExecutionEventType.STATE_CHANGED:
        print(f"State changed: {event.data}")

manager.event_bus.subscribe(on_event)
```

## REST API Endpoints

### Workflow Execution

- `POST /api/v1/workflows/{workflow_id}/execute` - Execute a workflow
- `GET /api/v1/executions/{execution_id}` - Get execution status
- `POST /api/v1/executions/{execution_id}/cancel` - Cancel execution
- `POST /api/v1/executions/{execution_id}/pause` - Pause execution
- `POST /api/v1/executions/{execution_id}/resume` - Resume execution

### State Operations

- `GET /api/v1/states` - List available states
- `GET /api/v1/states/{state_id}` - Get state details
- `POST /api/v1/states/detect` - Detect current state
- `POST /api/v1/states/navigate` - Navigate to a state

### History & Analytics

- `GET /api/v1/executions` - List execution history
- `GET /api/v1/executions/{execution_id}/events` - Get execution events
- `GET /api/v1/analytics/summary` - Get execution analytics

## WebSocket Endpoints

### Real-time Updates

- `WS /ws/executions/{execution_id}` - Subscribe to execution updates
- `WS /ws/events` - Subscribe to all execution events

## Configuration

API configuration via environment variables or config file:

```python
# config.yaml
api:
  host: "0.0.0.0"
  port: 8000
  cors_origins:
    - "http://localhost:3000"
    - "https://app.example.com"
  websocket_enabled: true
  max_connections: 100
  request_timeout: 300
```

## Security Considerations

- Authentication via API keys or JWT tokens
- CORS configuration for web clients
- Rate limiting for API endpoints
- WebSocket connection limits
- Execution permissions and access control

## Integration Examples

### CI/CD Integration

```bash
# Execute workflow via curl
curl -X POST "http://localhost:8000/api/v1/workflows/smoke_test/execute" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"timeout": 60, "capture_screenshots": true}'
```

### Python Client

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/v1/workflows/login/execute",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"timeout": 30}
    )
    execution = response.json()
    print(f"Execution started: {execution['execution_id']}")
```

## Future Enhancements

- GraphQL API support
- Batch execution endpoints
- Workflow scheduling API
- Test result reporting API
- Performance metrics API
- State discovery API integration

## Related Modules

- `execution`: Core workflow execution engine
- `state_management`: State lifecycle management
- `orchestration`: Workflow orchestration
- `discovery`: State detection and discovery
