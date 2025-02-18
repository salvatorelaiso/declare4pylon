import torch
from pydantic.dataclasses import dataclass
from pylon.solver import Solver

from declare4pylon import shape
from declare4pylon.constraint import DeclareConstraint, DeclareConstraintSettings
from declare4pylon.existence.last import ExistenceConstraintSettings


def existence(
    sampled: torch.IntTensor, *, activity: int, prefixes: torch.IntTensor | None = None
) -> torch.BoolTensor:
    """Returns a boolean tensor indicating whether the given `activity` is present in the trace for each row.

    Parameters
    ----------
    sampled : torch.IntTensor
        The samples returned by pylon
    activity : int
        The activity to search for existence
    prefixes : torch.IntTensor
        The prefixes of the traces (default is None).
        If None, the function will search for the activity in the traces without prefixes, otherwise it will stack the prefixes in front of the traces.

    Returns
    -------
    torch.BoolTensor
        A boolean tensor indicating whether the given `activity` is present in the trace for each row.
    """
    shape.check(sampled)
    shape.match(sampled, prefixes)
    traces = sampled if prefixes is None else torch.cat((prefixes, sampled), dim=1)
    return torch.any(traces == activity, dim=1)


class ExistenceConstraint(DeclareConstraint):
    def __init__(self, settings: ExistenceConstraintSettings, solver: Solver):
        super().__init__(settings, solver)

    @staticmethod
    def _condition(
        sampled: torch.Tensor,
        kwargs: dict,
    ) -> callable:
        return existence(sampled, **kwargs)
