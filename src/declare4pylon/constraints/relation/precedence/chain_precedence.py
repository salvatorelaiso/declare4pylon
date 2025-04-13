import torch
from jaxtyping import Bool, Int

from declare4pylon.constraints.relation import RelationTemplate


class ChainPrecedenceConstraint(RelationTemplate):
    """Each time B occurs in the process instance, then A occurs immediately beforehand."""

    def _condition(self, traces: Int[torch.Tensor, "batch activities"]) -> Bool[torch.Tensor, " batch"]:
        a_positions = torch.where(traces == self.activity_a)
        b_positions = torch.where(traces == self.activity_b)

        mask = torch.ones_like(traces, dtype=torch.bool)
        mask[b_positions] = False
        post_a_positions = [a_positions[0], torch.add(a_positions[1], +1)]
        post_a_positions[0] = post_a_positions[0][post_a_positions[1] < traces.shape[1]]
        post_a_positions[1] = post_a_positions[1][post_a_positions[1] < traces.shape[1]]
        mask[post_a_positions] = True

        return mask.all(dim=1)
