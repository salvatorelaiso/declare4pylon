import torch
from jaxtyping import Bool, Int

from .template import RelationTemplate


class AlternatePrecedenceConstraint(RelationTemplate):
    """Each time B occurs in the process instance, it is preceded by A and no other B can recur in between."""

    def _condition(self, traces: Int[torch.Tensor, "batch activities"]) -> Bool[torch.Tensor, " batch"]:
        rows = torch.ones(traces.shape[0], dtype=torch.bool)

        b_positions = torch.where(traces == self.activity_b)

        for i in range(b_positions[0].numel()):
            row, col = b_positions[0][i], b_positions[1][i]
            if not rows[row]:
                continue
            rows[row] = False

            for j in reversed(range(col)):
                if traces[row, j] == self.activity_b:
                    break
                if traces[row, j] == self.activity_a:
                    rows[row] = True
                    break

        return rows
