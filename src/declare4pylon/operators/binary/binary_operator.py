from declare4pylon.logic_expression import LogicExpression
from declare4pylon.operators.operator import Operator


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
            left (LogicalExpression): The left operand.
            right (LogicalExpression): The right operand.

        """
        self.left = left
        self.right = right
