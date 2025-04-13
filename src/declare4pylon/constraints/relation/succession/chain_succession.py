import torch
from jaxtyping import Bool, Int

from declare4pylon.constraints.relation import (
    ChainPrecedenceConstraint,
    ChainResponseConstraint,
    RelationTemplate,
)


class ChainSuccessionConstraint(RelationTemplate):
    """A and B occur in the process instance if and only if the latter immediately follows the former."""

    def _condition(self, traces: Int[torch.Tensor, "batch activities"]) -> Bool[torch.Tensor, " batch"]:
        return torch.logical_and(
            ChainResponseConstraint(activity_a=self.activity_a, activity_b=self.activity_b)._condition(traces),  # noqa: SLF001
            ChainPrecedenceConstraint(activity_a=self.activity_a, activity_b=self.activity_b)._condition(traces),  # noqa: SLF001
        )
