import torch
from jaxtyping import Bool, Int

from declare4pylon.operators.unary.unary_operator import UnaryOperator


class UnaryNot(UnaryOperator):
    """Unary operator to negate the result of a `LogicExpression`."""

    def _condition(self, traces: Int[torch.Tensor, "batch activities"]) -> Bool[torch.Tensor, " batch"]:
        return torch.logical_not(self.operand.evaluate(traces))
