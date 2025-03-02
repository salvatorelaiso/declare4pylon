import torch
from pylon.solver import Solver

from declare4pylon import shape
from declare4pylon.constraint import DeclareConstraint
from declare4pylon.existence.settings import ExistenceCountConstraintSettings


def exactly(
    sampled: torch.IntTensor,
    *,
    activity: int,
    count: int,
    prefixes: torch.IntTensor | None = None
) -> torch.BoolTensor:
    """Returns a boolean tensor indicating whether the given `activity` is present exactly `count` times in the trace for each row.

    Parameters
    ----------
    sampled : torch.IntTensor
        The samples returned by pylon
    activity : int
        The activity to search for existence
    count : int
        The exact number of times the activity should be present in the trace
    prefixes : torch.IntTensor
        The prefixes of the traces (default is None).
        If None, the function will search for the activity in the traces without prefixes, otherwise it will stack the prefixes in front of the traces.

    Returns
    -------
    torch.BoolTensor
        A boolean tensor indicating whether the given `activity` is present exactly `count` times in the trace for each row.
    """
    shape.check(sampled)
    shape.match(sampled, prefixes)
    traces = sampled if prefixes is None else torch.cat((prefixes, sampled), dim=1)
    return (traces == activity).sum(dim=1) == count


class ExactlyConstraint(DeclareConstraint):
    _condition = staticmethod(exactly)

    def __init__(self, settings: ExistenceCountConstraintSettings, solver: Solver):
        super().__init__(settings, solver)
