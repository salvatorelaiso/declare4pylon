import torch
from jaxtyping import Bool, Int

from .template import ExistenceTemplateWithCount


class ExistenceConstraint(ExistenceTemplateWithCount):
    """Constraint to check if an activity exists in traces.

    This constraint states that a certain activity must exist in the traces.
    It is defined by the activity to be checked and an optional count.
    It can also check if the activity exists with a specified count.
    The count specifies the minimum number of occurrences of the activity in the traces.
    If the count is not provided, it defaults to 1.

    Args:
    ----
        activity (int): The activity to be checked.
        count (int, optional): The minimum number of occurrences of the activity in the
            traces. Defaults to 1.

    """

    def _condition(
        self,
        traces: Int[torch.Tensor, "batch activities"],
    ) -> Bool[torch.Tensor, " batch"]:
        return (traces == self.activity).sum(dim=1) >= self.count
