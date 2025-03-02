import torch
from pylon.solver import Solver

import declare4pylon.shape as shape
from declare4pylon.choice.settings import ChoiceConstraintSettings
from declare4pylon.constraint import DeclareConstraint


def choice(
    sampled: torch.IntTensor, *, a: int, b: int, prefixes: torch.IntTensor = None
) -> torch.BoolTensor:
    """Returns a boolean tensor indicating whether the activities `a` or `b` are present in the trace for each row.

    Parameters
    ----------
    sampled : torch.IntTensor
        The samples returned by pylon
    a : int
        One of the two activity to search for existence
    b : int
        The other activity to search for existence
    prefixes : torch.IntTensor
        The prefixes of the traces (default is None).
        If None, the function will search for the activity in the traces without prefixes, otherwise it will stack the prefixes in front of the traces.

    Returns
    -------
    torch.BoolTensor
        A boolean tensor indicating whether the activities `a` or `b` are present in the trace for each row.
    """
    shape.check(sampled)
    shape.match(sampled, prefixes)
    traces = sampled if prefixes is None else torch.cat((prefixes, sampled), dim=1)
    return torch.any(traces == a, dim=1) | torch.any(traces == b, dim=1)


class ChoiceConstraint(DeclareConstraint):
    _condition = choice

    def __init__(self, settings: ChoiceConstraintSettings, solver: Solver):
        super().__init__(settings, solver)
