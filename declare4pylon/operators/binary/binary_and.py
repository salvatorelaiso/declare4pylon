import torch
from jaxtyping import Bool, Int

from .binary_operator import BinaryOperator


class BinaryAnd(BinaryOperator):
    """Binary operator to combine two `LogicExpression` using logical AND."""

    def _condition(self, traces: Int[torch.Tensor, "batch activities"]) -> Bool[torch.Tensor, " batch"]:
        return torch.logical_and(
            self.left.evaluate(traces),
            self.right.evaluate(traces),
        )
