from declare4pylon import LogicExpression
from declare4pylon.operators.operator import Operator


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
