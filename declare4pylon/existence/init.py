import torch
from pylon.solver import Solver

from declare4pylon import shape
from declare4pylon.constraint import DeclareConstraint
from declare4pylon.existence.settings import ExistenceConstraintSettings


def init(
    sampled: torch.IntTensor, *, activity: int, prefixes: torch.IntTensor = None
) -> torch.BoolTensor:
    """Returns a boolean tensor indicating whether the given `activity` is the first activity in the trace for each row.

    Parameters
    ----------
    sampled : torch.IntTensor
        The samples returned by pylon
    activity : int
        The activity to search for initialization
    prefixes : torch.IntTensor
        The prefixes of the traces (default is None).
        If None, the function will search for the activity in the traces without prefixes, otherwise it will consider the first activity in the prefixes.

    Returns
    -------
    torch.BoolTensor
        A boolean tensor indicating whether the given `activity` is the first activity in the trace for each row.
    """
    shape.check(sampled)
    shape.match(sampled, prefixes)
    traces = sampled if prefixes is None else prefixes
    return traces[:, 0] == activity


class InitConstraint(DeclareConstraint):
    _condition = init

    def __init__(self, settings: ExistenceConstraintSettings, solver: Solver):
        super().__init__(settings, solver)
