"""DSL JSON parser - ported from Qontinui framework.

Handles parsing of DSL JSON into Python objects.
"""

import json
from typing import Dict, Any, Optional
import logging

from ..dsl.instruction_set import InstructionSet
from ..dsl.business_task import BusinessTask
from ..dsl.model.parameter import Parameter
from ..dsl.model.task_sequence import TaskSequence
from ..dsl.model.action_step import ActionStep
from ..dsl.statements.statement import Statement
from ..dsl.expressions.expression import Expression


logger = logging.getLogger(__name__)


class DSLParser:
    """Parses DSL JSON into Python objects.
    
    Port of DSL from Qontinui framework parsing functionality.
    
    This parser handles the conversion of JSON-based DSL definitions into
    executable Python objects. It supports the full DSL including:
    - Automation functions (BusinessTask)
    - Statements (variable declarations, assignments, control flow, etc.)
    - Expressions (literals, variables, method calls, operations, builders)
    - Task sequences and action steps
    
    The parser ensures type safety and validates the structure against
    the expected DSL format.
    """
    
    def parse_json(self, json_string: str) -> InstructionSet:
        """Parse JSON string into InstructionSet.
        
        Args:
            json_string: JSON string containing DSL definition
            
        Returns:
            InstructionSet with parsed automation functions
            
        Raises:
            json.JSONDecodeError: If JSON is malformed
            ValueError: If DSL structure is invalid
        """
        try:
            data = json.loads(json_string)
            return self.parse_dict(data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise
    
    def parse_file(self, file_path: str) -> InstructionSet:
        """Parse JSON file into InstructionSet.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            InstructionSet with parsed automation functions
        """
        with open(file_path, 'r') as f:
            return self.parse_json(f.read())
    
    def parse_dict(self, data: Dict[str, Any]) -> InstructionSet:
        """Parse dictionary into InstructionSet.
        
        Args:
            data: Dictionary containing DSL definition
            
        Returns:
            InstructionSet with parsed automation functions
        """
        instruction_set = InstructionSet()
        
        if 'automationFunctions' in data:
            for func_data in data['automationFunctions']:
                instruction_set.automation_functions.append(
                    self.parse_business_task(func_data)
                )
        
        return instruction_set
    
    def parse_business_task(self, data: Dict[str, Any]) -> BusinessTask:
        """Parse dictionary into BusinessTask.
        
        Args:
            data: Dictionary containing function definition
            
        Returns:
            BusinessTask instance
        """
        task = BusinessTask()
        
        if 'id' in data:
            task.id = data['id']
        if 'name' in data:
            task.name = data['name']
        if 'description' in data:
            task.description = data['description']
        if 'returnType' in data:
            task.return_type = data['returnType']
        
        if 'parameters' in data:
            for param_data in data['parameters']:
                task.parameters.append(self.parse_parameter(param_data))
        
        if 'statements' in data:
            for stmt_data in data['statements']:
                task.statements.append(Statement.from_dict(stmt_data))
        
        return task
    
    def parse_parameter(self, data: Dict[str, Any]) -> Parameter:
        """Parse dictionary into Parameter.
        
        Args:
            data: Dictionary containing parameter definition
            
        Returns:
            Parameter instance
        """
        return Parameter(
            name=data.get('name', ''),
            type=data.get('type', '')
        )


class DSLValidator:
    """Validates DSL structures against schema.
    
    Port of DSL from Qontinui framework validation functionality.
    
    Ensures that DSL structures comply with the expected format and
    provides detailed error messages for malformed DSL.
    """
    
    def validate_instruction_set(self, instruction_set: InstructionSet) -> bool:
        """Validate an InstructionSet.
        
        Args:
            instruction_set: InstructionSet to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not instruction_set.automation_functions:
            logger.warning("InstructionSet has no automation functions")
            return False
        
        for func in instruction_set.automation_functions:
            if not self.validate_business_task(func):
                return False
        
        return True
    
    def validate_business_task(self, task: BusinessTask) -> bool:
        """Validate a BusinessTask.
        
        Args:
            task: BusinessTask to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not task.name:
            logger.error("BusinessTask missing name")
            return False
        
        if not task.return_type:
            logger.error(f"BusinessTask '{task.name}' missing return type")
            return False
        
        # Validate parameters
        param_names = set()
        for param in task.parameters:
            if param.name in param_names:
                logger.error(f"Duplicate parameter name '{param.name}' in task '{task.name}'")
                return False
            param_names.add(param.name)
            
            if not param.type:
                logger.error(f"Parameter '{param.name}' missing type in task '{task.name}'")
                return False
        
        return True