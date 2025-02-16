import torch

from declare4pylon import shape


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
