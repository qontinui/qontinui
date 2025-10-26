# Qontinui JSON Schemas

This directory contains JSON Schema definitions for all Pydantic models in the Qontinui library. These schemas can be used by TypeScript/JavaScript frontends for type validation, code generation, and IDE support.

## Generated Schema File

**`qontinui-schemas.json`** - Contains all model definitions in JSON Schema format (draft-07)

## Usage

### TypeScript/JavaScript

#### 1. Using with TypeScript Type Generation

You can use tools like `json-schema-to-typescript` to generate TypeScript types:

```bash
npm install -g json-schema-to-typescript
json2ts schemas/qontinui-schemas.json -o src/types/qontinui.d.ts
```

#### 2. Using with Runtime Validation (Ajv)

```typescript
import Ajv from 'ajv';
import schemas from './schemas/qontinui-schemas.json';

const ajv = new Ajv();

// Validate an Action object
const actionSchema = schemas.definitions.Action;
const validate = ajv.compile(actionSchema);

const action = {
  id: 'action1',
  type: 'CLICK',
  name: 'Click Login Button',
  config: {
    target: {
      type: 'image',
      imageId: 'login-button'
    }
  }
};

if (validate(action)) {
  console.log('Action is valid!');
} else {
  console.error('Validation errors:', validate.errors);
}
```

#### 3. Using with Zod

Convert JSON Schema to Zod schemas using `json-schema-to-zod`:

```bash
npm install json-schema-to-zod
```

#### 4. Direct JSON Schema Validation

```javascript
import { Validator } from 'jsonschema';
import schemas from './schemas/qontinui-schemas.json';

const validator = new Validator();

// Validate a Workflow
const workflowSchema = schemas.definitions.Workflow;
const result = validator.validate(myWorkflow, workflowSchema);

if (result.valid) {
  console.log('Workflow is valid!');
} else {
  console.error('Errors:', result.errors);
}
```

## Available Schemas

The schemas include definitions for:

### Main Models
- **Action** - Individual action definition
- **Workflow** - Complete workflow with actions and connections

### Action Configuration Types
All action types have corresponding configuration schemas:

#### Mouse Actions
- ClickActionConfig
- MouseMoveActionConfig
- MouseDownActionConfig
- MouseUpActionConfig
- DragActionConfig
- ScrollActionConfig

#### Keyboard Actions
- TypeActionConfig
- KeyPressActionConfig
- KeyDownActionConfig
- KeyUpActionConfig
- HotkeyActionConfig

#### Find/Wait Actions
- FindActionConfig
- FindStateImageActionConfig
- VanishActionConfig
- ExistsActionConfig
- WaitActionConfig

#### Control Flow Actions
- IfActionConfig
- LoopActionConfig
- BreakActionConfig
- ContinueActionConfig
- SwitchActionConfig
- TryCatchActionConfig

#### Data Actions
- SetVariableActionConfig
- GetVariableActionConfig
- SortActionConfig
- FilterActionConfig
- MapActionConfig
- ReduceActionConfig
- StringOperationActionConfig
- MathOperationActionConfig

#### State Actions
- GoToStateActionConfig
- RunWorkflowActionConfig
- ScreenshotActionConfig

### Supporting Types

#### Targets
- ImageTarget
- RegionTarget
- TextTarget
- CoordinatesTarget
- StateStringTarget
- CurrentPositionTarget

#### Basic Types
- Region
- Coordinates
- LoggingOptions
- SearchOptions
- VerificationConfig
- BaseActionSettings
- ExecutionSettings

#### Workflow Types
- Connection
- Connections
- WorkflowMetadata
- Variables
- WorkflowSettings

## Regenerating Schemas

If the Pydantic models are updated, regenerate the schemas by running:

```bash
cd /mnt/c/Users/jspin/Documents/qontinui_parent/qontinui
python scripts/export_schemas.py
```

Or from the project root:

```bash
python -m scripts.export_schemas
```

## Schema Structure

The generated `qontinui-schemas.json` file has the following structure:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Qontinui Schema Definitions",
  "description": "JSON Schema definitions for Qontinui Pydantic models",
  "version": "1.0.0",
  "definitions": {
    "Action": { ... },
    "Workflow": { ... },
    "ClickActionConfig": { ... },
    ...
  }
}
```

Each definition in the `definitions` object is a complete JSON Schema that can be used independently or referenced by other schemas.

## Field Naming

The schemas respect the Pydantic field aliases, using camelCase naming (e.g., `mouseButton`, `clickTarget`) which is standard in JavaScript/TypeScript, while the Python models use snake_case (e.g., `mouse_button`, `click_target`).

## Support

For issues or questions about the schemas:
1. Check the Pydantic models in `src/qontinui/config/schema.py`
2. Refer to the main Qontinui documentation
3. File an issue in the repository

## License

These schemas are part of the Qontinui library and are subject to the same license.
