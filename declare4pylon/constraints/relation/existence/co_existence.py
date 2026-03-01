import torch
from jaxtyping import Bool, Int

from declare4pylon.constraints.relation import RelationTemplate


class CoExistenceConstraint(RelationTemplate):
    """A and B occur either both or none in the process instance."""

    def _condition(self, traces: Int[torch.Tensor, "batch activities"]) -> Bool[torch.Tensor, " batch"]:
        activity_a_is_present = torch.any(traces == self.activity_a, dim=1)
        activity_b_is_present = torch.any(traces == self.activity_b, dim=1)
        return torch.logical_or(
            torch.logical_and(activity_a_is_present, activity_b_is_present),
            torch.logical_and(
                torch.logical_not(activity_a_is_present),
                torch.logical_not(activity_b_is_present),
            ),
        )
