import torch
from pylon.solver import Solver

from declare4pylon import shape
from declare4pylon.constraint import DeclareConstraint
from declare4pylon.existence.settings import ExistenceConstraintSettings


def last(sampled: torch.IntTensor, *, activity: int) -> torch.BoolTensor:
    """Returns a boolean tensor indicating whether the given `activity` is the last activity in the trace for each row.

    Parameters
    ----------
    sampled : torch.IntTensor
        The samples returned by pylon
    activity : int
        The activity to search for last activity

    Returns
    -------
    torch.BoolTensor
        A boolean tensor indicating whether the given `activity` is the last activity in the trace for each row.
    """
    shape.check(sampled)

    nonzero_indices = torch.nonzero(sampled)
    row_indices, col_indices = nonzero_indices[:, 0], nonzero_indices[:, 1]

    last_nonzero_indices = torch.zeros(
        sampled.shape[0], dtype=torch.long
    ).scatter_reduce(0, row_indices, col_indices, reduce="amax", include_self=False)

    last_nonzero_indices[(sampled != 0).sum(dim=1) == 0] = (
        -1
    )  # Use -1 to indicate no nonzero elements

    result = torch.zeros(sampled.shape[0], dtype=torch.bool)
    for last_index, row in zip(last_nonzero_indices, sampled):
        if last_index == -1:
            continue
        result[row] = row[last_index] == activity

    return result


class LastConstraint(DeclareConstraint):
    _condition = staticmethod(last)

    def __init__(self, settings: ExistenceConstraintSettings, solver: Solver):
        super().__init__(settings, solver)

    def __call__(self, logits, prefixes: torch.IntTensor | None = None) -> torch.Tensor:
        return self._constraint(logits, **self._settings.dict())
