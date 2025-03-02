import torch
from pylon.solver import Solver

from declare4pylon import shape
from declare4pylon.constraint import DeclareConstraint
from declare4pylon.relation.settings import RelationConstraintSettings


def responded_existence(
    sampled: torch.IntTensor, *, a: int, b: int, prefixes: torch.IntTensor = None
) -> torch.BoolTensor:
    """If A occurs in the process instance, then B occurs as well.

    Parameters
    ----------
    sampled : torch.IntTensor
        The samples returned by pylon
    a : int
        Activity A
    b : int
        Activity B
    prefixes : torch.IntTensor
        The prefixes of the traces.
        If None, the function will search for the activity in the `sampled` traces without prefixes,
        otherwise it will stack the prefixes in front of the traces. (default is None).

    Returns
    -------
    torch.BoolTensor
        A boolean tensor indicating which rows of the traces satisfy the relation.
    """
    shape.check(sampled)
    shape.match(sampled, prefixes)
    traces = sampled if prefixes is None else torch.cat((prefixes, sampled), dim=1)
    a_is_present = torch.any(traces == a, dim=1)
    b_is_present = torch.any(traces == b, dim=1)
    return torch.logical_or(
        torch.logical_not(a_is_present), torch.logical_and(a_is_present, b_is_present)
    )


def co_existence(
    sampled: torch.IntTensor, *, a: int, b: int, prefixes: torch.IntTensor = None
) -> torch.BoolTensor:
    """A and B occur either both or none in the process instance.

    Parameters
    ----------
    sampled : torch.IntTensor
        The samples returned by pylon
    a : int
        Activity A
    b : int
        Activity B
    prefixes : torch.IntTensor
        The prefixes of the traces.
        If None, the function will search for the activity in the `sampled` traces without prefixes,
        otherwise it will stack the prefixes in front of the traces. (default is None).

    Returns
    -------
    torch.BoolTensor
        A boolean tensor indicating which rows of the traces satisfy the relation.
    """
    shape.check(sampled)
    shape.match(sampled, prefixes)
    traces = sampled if prefixes is None else torch.cat((prefixes, sampled), dim=1)
    a_is_present = torch.any(traces == a, dim=1)
    b_is_present = torch.any(traces == b, dim=1)
    return torch.logical_or(
        torch.logical_and(a_is_present, b_is_present),
        torch.logical_and(
            torch.logical_not(a_is_present), torch.logical_not(b_is_present)
        ),
    )


class RespondedExistenceConstraint(DeclareConstraint):
    _condition = responded_existence

    def __init__(self, settings: RelationConstraintSettings, solver: Solver):
        super().__init__(settings, solver)


class CoExistenceConstraint(DeclareConstraint):
    _condition = co_existence

    def __init__(self, settings: RelationConstraintSettings, solver: Solver):
        super().__init__(settings, solver)
