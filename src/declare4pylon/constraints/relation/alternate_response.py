import torch
from jaxtyping import Bool, Int

from .template import RelationTemplate


class AlternateResponseConstraint(RelationTemplate):
    """Each time A occurs in the process instance, then B occurs afterwards, before A recurs."""

    def _condition(self, traces: Int[torch.Tensor, "batch activities"]) -> Bool[torch.Tensor, " batch"]:
        rows = torch.ones(traces.shape[0], dtype=torch.bool)
        a_positions = torch.where(traces == self.activity_a)

        for i in range(a_positions[0].numel()):
            row, col = a_positions[0][i], a_positions[1][i]
            if not rows[row]:
                continue
            rows[row] = False

            for j in range(col + 1, traces.shape[1]):
                if traces[row, j] == self.activity_b:
                    rows[row] = True
                    break
                if traces[row, j] == self.activity_a:
                    break

        return rows
