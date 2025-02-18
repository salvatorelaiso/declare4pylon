import torch
from pylon.solver import Solver

from declare4pylon import shape
from declare4pylon.constraint import DeclareConstraint
from declare4pylon.relation.settings import RelationConstraintSettings


def response(
    sampled: torch.IntTensor, *, a: int, b: int, prefixes: torch.IntTensor = None
) -> torch.BoolTensor:
    """If A occurs in the process instance, then B occurs after A.

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
    traces = sampled if prefixes is None else torch.cat((prefixes, sampled), dim=1)
    shape.match(sampled, prefixes)
    a_positions = torch.where(traces == a)
    b_positions = torch.where(traces == b)

    mask = torch.ones_like(traces, dtype=torch.bool)
    mask[a_positions] = False

    for i in range(traces.shape[0]):
        b_indices = b_positions[1][b_positions[0] == i]
        if b_indices.numel() > 0:
            mask[i, : b_indices.max() + 1] = True

    return mask.all(dim=1)


def alternate_response(
    sampled: torch.IntTensor, *, a: int, b: int, prefixes: torch.IntTensor = None
) -> torch.BoolTensor:
    """Each time A occurs in the process instance, then B occurs afterwards, before A recurs.

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
    a_positions = torch.where(traces == a)

    for i in range(a_positions[0].numel()):
        row, col = a_positions[0][i], a_positions[1][i]
        if not rows[row]:
            continue
        rows[row] = False

        for j in range(col + 1, traces.shape[1]):
            if traces[row, j] == b:
                rows[row] = True
                break
            if traces[row, j] == a:
                break

    return rows


def chain_response(
    sampled: torch.IntTensor, *, a: int, b: int, prefixes: torch.IntTensor = None
) -> torch.BoolTensor:
    """Each time A occurs in the process instance, then B occurs immediately afterwards.

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
    mask[a_positions] = False
    pre_b_positions = [b_positions[0], torch.add(b_positions[1], -1)]
    pre_b_positions[0] = pre_b_positions[0][pre_b_positions[1] >= 0]
    pre_b_positions[1] = pre_b_positions[1][pre_b_positions[1] >= 0]
    mask[pre_b_positions] = True

    return mask.all(dim=1)


class ResponseConstraint(DeclareConstraint):
    def __init__(self, settings: RelationConstraintSettings, solver: Solver):
        super().__init__(settings, solver)

    @staticmethod
    def _condition(
        sampled: torch.Tensor,
        kwargs: dict,
    ) -> callable:
        return response(sampled, **kwargs)


class AlternateResponseConstraint(DeclareConstraint):
    def __init__(self, settings: RelationConstraintSettings, solver: Solver):
        super().__init__(settings, solver)

    @staticmethod
    def _condition(
        sampled: torch.Tensor,
        kwargs: dict,
    ) -> callable:
        return alternate_response(sampled, **kwargs)


class ChainResponseConstraint(DeclareConstraint):
    def __init__(self, settings: RelationConstraintSettings, solver: Solver):
        super().__init__(settings, solver)

    @staticmethod
    def _condition(
        sampled: torch.Tensor,
        kwargs: dict,
    ) -> callable:
        return chain_response(sampled, **kwargs)
