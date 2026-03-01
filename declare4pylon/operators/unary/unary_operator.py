from declare4pylon import LogicExpression
from declare4pylon.operators import Operator


class UnaryOperator(Operator):
    """Base class for unary operators.

    This class represents a unary operator that operates on a single operand.
    The operand must be an instance of LogicExpression.
    This class is intended to be subclassed by specific unary operator implementations.
    It is not meant to be instantiated directly.
    """

    def __init__(self, operand: LogicExpression) -> None:
        """Initialize a UnaryOperator.

        Args:
        ----
            operand (LogicExpression): The operand for the unary operator.

        """
        self.operand = operand

    def __repr__(self) -> str:
        """Return a string representation for debugging."""
        return f"{self.__class__.__name__}(operand={self.operand!r})"

    def __str__(self) -> str:
        """Return a string representation of the unary operator."""
        return f"{self.__class__.__name__}({self.operand})"

    def __eq__(self, other: object) -> bool:
        """Check equality with another unary operator based on the operand."""
        if not isinstance(other, self.__class__):
            return False
        return self.operand == other.operand

    def __hash__(self) -> int:
        """Return the hash based on the operator class and its operand."""
        return hash((self.__class__, self.operand))
