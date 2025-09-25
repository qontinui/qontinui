"""Statement base class - ported from Qontinui framework.

Abstract base class for all statements in the DSL.
"""


class Statement:
    """Abstract base class for all statements in the DSL.

    Port of Statement from Qontinui framework class.

    A statement represents an executable instruction in the DSL that performs
    an action but does not produce a value (unlike expressions). Statements
    form the body of automation functions and control the flow of execution.

    Supported statement types:
    - Variable declarations - Declare and optionally initialize variables
    - Assignments - Assign values to existing variables
    - Method calls - Invoke methods for their side effects
    - Control flow - If statements and forEach loops
    - Returns - Return values from functions

    This class uses polymorphic deserialization to support parsing different
    statement types from JSON based on the "statementType" discriminator field.
    """

    def __init__(self, statement_type: str):
        """Initialize statement with its type.

        Args:
            statement_type: The discriminator field used to determine the concrete type
                          Valid values: "variableDeclaration", "assignment", "if",
                          "forEach", "return", "methodCall"
        """
        self.statement_type = statement_type

    @classmethod
    def from_dict(cls, data: dict) -> "Statement":
        """Create Statement from dictionary representation.

        Args:
            data: Dictionary with statement data

        Returns:
            Appropriate Statement subclass instance
        """
        statement_type = data.get("statementType", "")

        if statement_type == "variableDeclaration":
            from .variable_declaration_statement import VariableDeclarationStatement

            return VariableDeclarationStatement.from_dict(data)
        elif statement_type == "assignment":
            from .assignment_statement import AssignmentStatement

            return AssignmentStatement.from_dict(data)
        elif statement_type == "methodCall":
            from .method_call_statement import MethodCallStatement

            return MethodCallStatement.from_dict(data)
        elif statement_type == "if":
            from .if_statement import IfStatement

            return IfStatement.from_dict(data)
        elif statement_type == "forEach":
            from .for_each_statement import ForEachStatement

            return ForEachStatement.from_dict(data)
        elif statement_type == "return":
            from .return_statement import ReturnStatement

            return ReturnStatement.from_dict(data)
        else:
            raise ValueError(f"Unknown statement type: {statement_type}")

    def to_dict(self) -> dict:
        """Convert Statement to dictionary representation.

        Returns:
            Dictionary representation
        """
        return {"statementType": self.statement_type}
