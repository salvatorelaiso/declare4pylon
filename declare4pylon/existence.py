import torch

import declare4pylon.shape as shape


def existence(
    sampled: torch.IntTensor, *, activity: int, prefixes: torch.IntTensor = None
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


def absence(
    sampled: torch.IntTensor, *, activity: int, prefixes: torch.IntTensor = None
) -> torch.BoolTensor:
    """Returns a boolean tensor indicating whether the given `activity` is absent in the trace for each row.

    Parameters
    ----------
    sampled : torch.IntTensor
        The samples returned by pylon
    activity : int
        The activity to search for absence
    prefixes : torch.IntTensor
        The prefixes of the traces (default is None).
        If None, the function will search for the activity in the traces without prefixes, otherwise it will stack the prefixes in front of the traces.

    Returns
    -------
    torch.BoolTensor
        A boolean tensor indicating whether the given `activity` is absent in the trace for each row.
    """
    shape.check(sampled)
    shape.match(sampled, prefixes)
    traces = sampled if prefixes is None else torch.cat((prefixes, sampled), dim=1)
    return torch.all(traces != activity, dim=1)


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


def last(sampled: torch.IntTensor, *, activity: int) -> torch.BoolTensor:
    """Returns a boolean tensor indicating whether the given `activity` is the last activity in the trace for each row.

    Parameters
    ----------
    sampled : torch.IntTensor
        The samples returned by pylon
    activity : int
        The activity to search for last activity

    Returns
    -------
    torch.BoolTensor
        A boolean tensor indicating whether the given `activity` is the last activity in the trace for each row.
    """
    shape.check(sampled)
    return sampled[:, -1] == activity
