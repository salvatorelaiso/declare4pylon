import torch
from jaxtyping import Bool, Int

from declare4pylon.constraints.relation import AlternatePrecedenceConstraint, AlternateResponseConstraint

from .template import RelationTemplate


class AlternateSuccessionConstraint(RelationTemplate):
    """A and B occur if and only if the latter follows the former, and they alternate each other in the trace."""

    def _condition(self, traces: Int[torch.Tensor, "batch activities"]) -> Bool[torch.Tensor, " batch"]:
        return torch.logical_and(
            AlternateResponseConstraint(activity_a=self.activity_a, activity_b=self.activity_b)._condition(traces),  # noqa: SLF001
            AlternatePrecedenceConstraint(activity_a=self.activity_a, activity_b=self.activity_b)._condition(traces),  # noqa: SLF001
        )
