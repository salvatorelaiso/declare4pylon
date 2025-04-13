import torch
from jaxtyping import Bool, Int

from declare4pylon.constraints.relation import (
    PrecedenceConstraint,
    RelationTemplate,
    ResponseConstraint,
)


class SuccessionConstraint(RelationTemplate):
    """A occurs if and only if it is followed by B. B occurs if and only if it is preceded by A."""

    def _condition(self, traces: Int[torch.Tensor, "batch activities"]) -> Bool[torch.Tensor, " batch"]:
        return torch.logical_and(
            ResponseConstraint(activity_a=self.activity_a, activity_b=self.activity_b)._condition(traces),  # noqa: SLF001
            PrecedenceConstraint(activity_a=self.activity_a, activity_b=self.activity_b)._condition(traces),  # noqa: SLF001
        )
