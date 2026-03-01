from declare4pylon import LogicExpression
from declare4pylon.operators import Operator


class BinaryOperator(Operator):
    """Abstract class for binary operators.

    This class represents a binary operator that operates on two operands.
    The operands must be instances of LogicExpression.
    This class is intended to be subclassed by specific binary operator implementations.
    It is not meant to be instantiated directly.
    """

    def __init__(self, left: LogicExpression, right: LogicExpression) -> None:
        """Initialize a binary operator.

        Args:
        ----
            left (LogicExpression): The left operand.
            right (LogicExpression): The right operand.

        """
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        """Return a string representation for debugging."""
        return f"{self.__class__.__name__}(left={self.left!r}, right={self.right!r})"

    def __str__(self) -> str:
        """Return a string representation of the binary operator."""
        return f"{self.__class__.__name__}({self.left}, {self.right})"

    def __eq__(self, other: object) -> bool:
        """Check equality with another binary operator based on the left and right operands."""
        if not isinstance(other, self.__class__):
            return False
        return self.left == other.left and self.right == other.right

    def __hash__(self) -> int:
        """Return the hash based on the operator class and its left and right operands."""
        return hash((self.__class__, self.left, self.right))
