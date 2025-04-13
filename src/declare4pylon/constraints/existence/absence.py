import torch
from jaxtyping import Bool, Int

from .template import ExistenceTemplateWithCount


class AbsenceConstraint(ExistenceTemplateWithCount):
    """A class used to represent the absence of an activity in traces.

    This constraint states that a certain activity must not occur in the traces.
    It is defined by the activity to be checked and a count.
    The count specifies the threshold at which the activity is considered absent.
    E.g. if the count is 1, the activity must not occur at all in the traces.
    If the count is not provided, it defaults to 1.

    Args:
    ----
        activity (int): The activity to be checked.
        count (int, optional): The threshold at which the activity is considered absent.
            Defaults to 1.

    """

    def _condition(
        self,
        traces: Int[torch.Tensor, "batch activities"],
    ) -> Bool[torch.Tensor, " batch"]:
        return (traces == self.activity).sum(dim=1) < self.count
