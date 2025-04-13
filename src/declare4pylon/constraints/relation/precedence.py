import torch
from jaxtyping import Bool, Int

from .template import RelationTemplate


class PrecedenceConstraint(RelationTemplate):
    """B occurs in the process instance only if preceded by A."""

    def _condition(self, traces: Int[torch.Tensor, "batch activities"]) -> Bool[torch.Tensor, " batch"]:
        a_positions = torch.where(traces == self.activity_a)
        b_positions = torch.where(traces == self.activity_b)

        mask = torch.ones_like(traces, dtype=torch.bool)
        mask[b_positions] = False
        for i in range(traces.shape[0]):
            a_row_positions = a_positions[1][a_positions[0] == i]
            if a_row_positions.numel() > 0:
                mask[i, a_row_positions.min() :] = True

        return mask.all(dim=1)
