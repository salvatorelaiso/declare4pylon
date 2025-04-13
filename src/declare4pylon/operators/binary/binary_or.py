import torch
from jaxtyping import Bool, Int

from .binary_operator import BinaryOperator


class BinaryOr(BinaryOperator):
    """Binary operator to combine two `LogicExpression` using logical OR."""

    def _condition(self, traces: Int[torch.Tensor, "batch activities"]) -> Bool[torch.Tensor, " batch"]:
        return torch.logical_or(
            self.left.evaluate(traces),
            self.right.evaluate(traces),
        )
