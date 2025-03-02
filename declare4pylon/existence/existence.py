import torch
from pylon.solver import Solver

from declare4pylon import shape
from declare4pylon.constraint import DeclareConstraint
from declare4pylon.existence.last import ExistenceConstraintSettings


def existence(
    sampled: torch.IntTensor,
    *,
    activity: int,
    count: int = 1,
    prefixes: torch.IntTensor | None = None
) -> torch.BoolTensor:
    """Returns a boolean tensor indicating whether the given `activity` is present at least `count` times in the trace for each row.

    Parameters
    ----------
    sampled : torch.IntTensor
        The samples returned by pylon
    activity : int
        The activity to search for existence
    count : int
        The minimum number of times the activity should be present in the trace (default is 1).
    prefixes : torch.IntTensor
        The prefixes of the traces (default is None).
        If None, the function will search for the activity in the traces without prefixes, otherwise it will stack the prefixes in front of the traces.

    Returns
    -------
    torch.BoolTensor
        A boolean tensor indicating whether the given `activity` is present at least `count` times in the trace for each row.
    """
    shape.check(sampled)
    shape.match(sampled, prefixes)
    traces = sampled if prefixes is None else torch.cat((prefixes, sampled), dim=1)
    return (traces == activity).sum(dim=1) >= count


class ExistenceConstraint(DeclareConstraint):
    _condition = existence

    def __init__(self, settings: ExistenceConstraintSettings, solver: Solver):
        super().__init__(settings, solver)
