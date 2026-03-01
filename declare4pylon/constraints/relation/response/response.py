import torch
from jaxtyping import Bool, Int

from declare4pylon.constraints.relation import RelationTemplate


class ResponseConstraint(RelationTemplate):
    """If A occurs in the process instance, then B occurs after A."""

    def _condition(self, traces: Int[torch.Tensor, "batch activities"]) -> Bool[torch.Tensor, " batch"]:
        a_positions = torch.where(traces == self.activity_a)
        b_positions = torch.where(traces == self.activity_b)

        mask = torch.ones_like(traces, dtype=torch.bool)
        mask[a_positions] = False

        for i in range(traces.shape[0]):
            b_indices = b_positions[1][b_positions[0] == i]
            if b_indices.numel() > 0:
                mask[i, : b_indices.max() + 1] = True

        return mask.all(dim=1)
