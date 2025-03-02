import torch
from pylon.solver import Solver

from declare4pylon import shape
from declare4pylon.constraint import DeclareConstraint
from declare4pylon.relation.precedence import (
    alternate_precedence,
    chain_precedence,
    precedence,
)
from declare4pylon.relation.response import alternate_response, chain_response, response
from declare4pylon.relation.settings import RelationConstraintSettings


def succession(
    sampled: torch.IntTensor, *, a: int, b: int, prefixes: torch.IntTensor = None
) -> torch.BoolTensor:
    """A occurs if and only if it is followed by B in the process instance.
    B occurs if and only if it is preceded by A.

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
    return torch.logical_and(
        response(traces, a=a, b=b),
        precedence(traces, a=a, b=b),
    )


def alternate_succession(
    sampled: torch.IntTensor, *, a: int, b: int, prefixes: torch.IntTensor = None
) -> torch.BoolTensor:
    """A and B occur if and only if the latter follows the former, and they alternate each other in the trace.

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
    return torch.logical_and(
        alternate_response(traces, a=a, b=b),
        alternate_precedence(traces, a=a, b=b),
    )


def chain_succession(
    sampled: torch.IntTensor, *, a: int, b: int, prefixes: torch.IntTensor = None
) -> torch.BoolTensor:
    """A and B occur in the process instance if and only if the latter follows the former.

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
    return torch.logical_and(
        chain_response(sampled, a=a, b=b, prefixes=prefixes),
        chain_precedence(sampled, a=a, b=b, prefixes=prefixes),
    )


class SuccessionConstraint(DeclareConstraint):
    _condition = staticmethod(succession)

    def __init__(self, settings: RelationConstraintSettings, solver: Solver):
        super().__init__(settings, solver)


class AlternateSuccessionConstraint(DeclareConstraint):
    _condition = staticmethod(alternate_succession)

    def __init__(self, settings: RelationConstraintSettings, solver: Solver):
        super().__init__(settings, solver)


class ChainSuccessionConstraint(DeclareConstraint):
    _condition = staticmethod(chain_succession)

    def __init__(self, settings: RelationConstraintSettings, solver: Solver):
        super().__init__(settings, solver)
