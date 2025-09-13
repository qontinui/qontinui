# Qontinui JSON Executor

Execute automation configurations exported from the Qontinui web builder.

## Installation

1. Install Python dependencies:
```bash
cd /home/jspinak/qontinui_parent_directory/qontinui
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run a JSON configuration file:
```bash
python run_json.py path/to/config.json
```

### Command Line Options

```bash
python run_json.py config.json [options]

Options:
  --mode {state_machine,process,single_action}
                        Execution mode (default: state_machine)
  --verbose            Enable verbose output
  --dry-run           Validate configuration without executing
  -h, --help          Show help message
```

### Execution Modes

1. **state_machine** (default): Execute using state machine logic with transitions
2. **process**: Execute all processes sequentially
3. **single_action**: Execute all individual actions

### Examples

```bash
# Run with state machine (default)
python run_json.py examples/simple_automation.json

# Run processes only
python run_json.py my_config.json --mode process

# Validate configuration without running
python run_json.py my_config.json --dry-run

# Verbose output
python run_json.py my_config.json --verbose
```

## Python API

### Basic Usage

```python
from qontinui.json_executor import JSONRunner

# Create runner
runner = JSONRunner()

# Load configuration
runner.load_configuration('path/to/config.json')

# Run automation
runner.run(mode='state_machine')

# Clean up
runner.cleanup()
```

### Advanced Usage

```python
from qontinui.json_executor import ConfigParser, StateExecutor

# Parse configuration
parser = ConfigParser()
config = parser.parse_file('config.json')

# Create state executor
executor = StateExecutor(config)

# Initialize and run
executor.initialize()
executor.execute()

# Get results
print(f"Active states: {executor.get_active_states()}")
print(f"History: {executor.get_state_history()}")

# Clean up
parser.cleanup()
```

## Configuration Format

The JSON configuration follows the schema defined in the Qontinui web builder.

### Minimal Example

```json
{
  "version": "1.0.0",
  "metadata": {
    "name": "My Automation"
  },
  "images": [],
  "processes": [{
    "id": "p1",
    "name": "Click Process",
    "type": "sequence",
    "actions": [{
      "id": "a1",
      "type": "CLICK",
      "config": {
        "target": {
          "type": "coordinates",
          "coordinates": {"x": 100, "y": 200}
        }
      }
    }]
  }],
  "states": [{
    "id": "s1",
    "name": "Initial",
    "isInitial": true,
    "identifyingImages": [],
    "position": {"x": 0, "y": 0}
  }],
  "transitions": []
}
```

## Supported Actions

- **FIND**: Locate element on screen
- **CLICK**: Single click
- **DOUBLE_CLICK**: Double click
- **RIGHT_CLICK**: Right click
- **TYPE**: Type text
- **KEY_PRESS**: Press keyboard keys
- **DRAG**: Drag from one point to another
- **SCROLL**: Scroll in direction
- **WAIT**: Wait for duration
- **VANISH**: Wait for element to disappear
- **EXISTS**: Check if element exists
- **MOVE**: Move mouse to position
- **SCREENSHOT**: Capture screen region

## Image Recognition

Images are exported as base64 in the JSON and automatically extracted to temporary files during execution.

### Image Matching
- Uses OpenCV template matching
- Configurable similarity threshold (0.0 - 1.0)
- Multi-scale search support

## State Machine Execution

1. **Initialization**: Find and activate initial state
2. **State Verification**: Check identifying images
3. **Transition Execution**: Execute applicable transitions
4. **Process Execution**: Run processes in transitions
5. **State Updates**: Activate/deactivate states per transition rules

## Error Handling

- Configurable retry counts per action
- Timeout settings
- Failure strategies: stop, retry, continue
- Keyboard interrupt (Ctrl+C) to stop execution

## Troubleshooting

### Common Issues

1. **Image not found**: 
   - Check threshold settings
   - Ensure image quality matches
   - Verify screen resolution

2. **State not detected**:
   - Check identifying images
   - Verify state is visible on screen

3. **Action timeout**:
   - Increase timeout values
   - Check target element exists

### Debug Tips

- Use `--dry-run` to validate configuration
- Use `--verbose` for detailed output
- Check temporary image directory for extracted images
- Review action logs for failure points

## Limitations

- Currently uses PyAutoGUI for actions (will migrate to HAL)
- Image recognition requires exact resolution match
- No parallel action execution yet

## Next Steps

- Integration with Qontinui HAL for cross-platform support
- Advanced image recognition with ML models
- Parallel action execution
- Real-time progress reporting to web interface