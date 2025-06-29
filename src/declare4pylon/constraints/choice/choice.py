import torch
from jaxtyping import Bool, Int

from .template import ChoiceTemplate


class ChoiceConstraint(ChoiceTemplate):
    """Activity A or B eventually occur in the process instance."""

    def _condition(self, traces: Int[torch.Tensor, "batch activities"]) -> Bool[torch.Tensor, " batch"]:
        return torch.any(traces == self.activity_a, dim=1) | torch.any(traces == self.activity_b, dim=1)
