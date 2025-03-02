import torch
from pylon.solver import Solver

from declare4pylon import shape
from declare4pylon.constraint import DeclareConstraint
from declare4pylon.relation.settings import RelationConstraintSettings


def precedence(
    sampled: torch.IntTensor, *, a: int, b: int, prefixes: torch.IntTensor = None
) -> torch.BoolTensor:
    """B occurs in the process instance only if preceded by A.

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
    a_positions = torch.where(traces == a)
    b_positions = torch.where(traces == b)

    mask = torch.ones_like(traces, dtype=torch.bool)
    mask[b_positions] = False

    for i in range(traces.shape[0]):
        a_indices = a_positions[1][a_positions[0] == i]
        if a_indices.numel() > 0:
            mask[i, a_indices.max() :] = True

    return mask.all(dim=1)


def alternate_precedence(
    sampled: torch.IntTensor, *, a: int, b: int, prefixes: torch.IntTensor = None
) -> torch.BoolTensor:
    """Each time B occurs in the process instance, it is preceded by A and no other B can recur in between.

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

    rows = torch.ones(traces.shape[0], dtype=torch.bool)
    b_positions = torch.where(traces == b)

    for i in range(b_positions[0].numel()):
        row, col = b_positions[0][i], b_positions[1][i]
        if not rows[row]:
            continue
        rows[row] = False

        for j in reversed(range(0, col)):
            if traces[row, j] == b:
                break
            if traces[row, j] == a:
                rows[row] = True
                break

    return rows


def chain_precedence(
    sampled: torch.IntTensor, *, a: int, b: int, prefixes: torch.IntTensor = None
) -> torch.BoolTensor:
    """Each time B occurs in the process instance, then A occurs immediately beforehand.

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
    a_positions = torch.where(traces == a)
    b_positions = torch.where(traces == b)

    mask = torch.ones_like(traces, dtype=torch.bool)
    mask[b_positions] = False
    post_a_positions = [a_positions[0], torch.add(a_positions[1], +1)]
    post_a_positions[0] = post_a_positions[0][post_a_positions[1] < traces.shape[1]]
    post_a_positions[1] = post_a_positions[1][post_a_positions[1] < traces.shape[1]]
    mask[post_a_positions] = True

    return mask.all(dim=1)


class PrecedenceConstraint(DeclareConstraint):
    _condition = precedence

    def __init__(self, settings: RelationConstraintSettings, solver: Solver):
        super().__init__(settings, solver)


class AlternatePrecedenceConstraint(DeclareConstraint):
    _condition = alternate_precedence

    def __init__(self, settings: RelationConstraintSettings, solver: Solver):
        super().__init__(settings, solver)


class ChainPrecedenceConstraint(DeclareConstraint):
    _condition = chain_precedence

    def __init__(self, settings: RelationConstraintSettings, solver: Solver):
        super().__init__(settings, solver)
