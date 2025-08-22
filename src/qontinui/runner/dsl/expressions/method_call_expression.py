"""Method call expression - ported from Qontinui framework.

Represents a method call expression in the DSL.
"""

from typing import List, Optional, Any
from dataclasses import dataclass, field

from .expression import Expression


@dataclass
class MethodCallExpression(Expression):
    """Represents a method call expression in the DSL.
    
    Port of MethodCallExpression from Qontinui framework class.
    
    A method call expression invokes a method and returns its result.
    This is used when the return value of the method is needed, unlike
    MethodCallStatement which discards the return value.
    
    Example in JSON:
        {
            "expressionType": "methodCall",
            "object": "calculator",
            "method": "add",
            "arguments": [
                {"expressionType": "literal", "valueType": "integer", "value": 5},
                {"expressionType": "literal", "valueType": "integer", "value": 3}
            ]
        }
    """
    
    object: Optional[str] = None
    """The name of the object on which to invoke the method.
    Can be None for static methods or global function calls."""
    
    method: str = ""
    """The name of the method or function to invoke."""
    
    arguments: List[Expression] = field(default_factory=list)
    """List of argument expressions to pass to the method."""
    
    def __init__(self, object: Optional[str] = None, method: str = "",
                 arguments: Optional[List[Expression]] = None):
        """Initialize method call expression.
        
        Args:
            object: Object to call method on
            method: Method name
            arguments: Method arguments
        """
        super().__init__("methodCall")
        self.object = object
        self.method = method
        self.arguments = arguments or []
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MethodCallExpression':
        """Create MethodCallExpression from dictionary.
        
        Args:
            data: Dictionary with expression data
            
        Returns:
            MethodCallExpression instance
        """
        arguments = []
        if 'arguments' in data:
            arguments = [Expression.from_dict(arg) for arg in data['arguments']]
        
        return cls(
            object=data.get('object'),
            method=data.get('method', ''),
            arguments=arguments
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        result = super().to_dict()
        if self.object:
            result['object'] = self.object
        result['method'] = self.method
        if self.arguments:
            result['arguments'] = [arg.to_dict() for arg in self.arguments]
        return result
    
    def evaluate(self, context: dict) -> Any:
        """Evaluate the method call expression.
        
        Args:
            context: Variable context for evaluation
            
        Returns:
            The return value of the method call
        """
        # Evaluate arguments
        arg_values = [arg.evaluate(context) for arg in self.arguments]
        
        # In a real implementation, this would look up the object/method
        # and invoke it with the arguments
        # For now, return a placeholder
        return f"MethodCall({self.object}.{self.method}({arg_values}))"