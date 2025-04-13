import torch
from jaxtyping import Bool, Int

from .template import RelationTemplate


class ChainResponseConstraint(RelationTemplate):
    """Each time A occurs in the process instance, then B occurs immediately afterwards."""

    def _condition(self, traces: Int[torch.Tensor, "batch activities"]) -> Bool[torch.Tensor, " batch"]:
        a_positions = torch.where(traces == self.activity_a)

        b_positions = torch.where(traces == self.activity_b)

        mask = torch.ones_like(traces, dtype=torch.bool)
        mask[a_positions] = False
        pre_b_positions = [b_positions[0], torch.add(b_positions[1], -1)]
        pre_b_positions[0] = pre_b_positions[0][pre_b_positions[1] >= 0]
        pre_b_positions[1] = pre_b_positions[1][pre_b_positions[1] >= 0]
        mask[pre_b_positions] = True

        return mask.all(dim=1)
