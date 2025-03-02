import torch
from pylon.solver import Solver

from declare4pylon import shape
from declare4pylon.constraint import DeclareConstraint
from declare4pylon.relation.precedence import precedence
from declare4pylon.relation.response import response
from declare4pylon.relation.settings import RelationConstraintSettings


def not_co_existence(
    sampled: torch.IntTensor, *, a: int, b: int, prefixes: torch.IntTensor = None
) -> torch.BoolTensor:
    """A and B never occur together in the process instance.

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
    return torch.logical_not(torch.logical_and(a_is_present, b_is_present))


def not_succession(
    sampled: torch.IntTensor, *, a: int, b: int, prefixes: torch.IntTensor = None
) -> torch.BoolTensor:
    """A can never occur before B in the process instance.

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
    return torch.logical_not(
        torch.logical_and(
            response(traces, a=a, b=b),
            precedence(traces, a=a, b=b),
        )
    )


def not_chain_succession(
    sampled: torch.IntTensor, *, a: int, b: int, prefixes: torch.IntTensor = None
) -> torch.BoolTensor:
    """A and B occur in the process instance if and only if the latter does not follows the former.

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

    b_positions = torch.where(traces == b)

    mask = torch.ones_like(traces, dtype=torch.bool)
    pre_b_positions = [b_positions[0], torch.add(b_positions[1], -1)]
    pre_b_positions[0] = pre_b_positions[0][pre_b_positions[1] >= 0]
    pre_b_positions[1] = pre_b_positions[1][pre_b_positions[1] >= 0]
    mask[pre_b_positions] = traces[pre_b_positions] != a

    return mask.all(dim=1)


class NotCoExistenceConstraint(DeclareConstraint):
    _condition = staticmethod(not_co_existence)

    def __init__(self, settings: RelationConstraintSettings, solver: Solver):
        super().__init__(settings, solver)


class NotSuccessionConstraint(DeclareConstraint):
    _condition = staticmethod(not_succession)

    def __init__(self, settings: RelationConstraintSettings, solver: Solver):
        super().__init__(settings, solver)


class NotChainSuccessionConstraint(DeclareConstraint):
    _condition = staticmethod(not_chain_succession)

    def __init__(self, settings: RelationConstraintSettings, solver: Solver):
        super().__init__(settings, solver)
