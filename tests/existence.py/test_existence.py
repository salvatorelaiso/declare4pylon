from contextlib import nullcontext as does_not_raise

import pytest
import torch
from pylon.sampling_solver import WeightedSamplingSolver

from declare4pylon.existence.existence import (
    ExistenceConstraint,
    ExistenceConstraintSettings,
    existence,
)


@pytest.mark.parametrize(
    "traces, activity, prefixes, expected_result, expected_raise",
    [
        (
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
            1,
            None,
            torch.tensor([True, False, False]),
            does_not_raise(),
        ),
        (
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
            1,
            torch.tensor([[1], [1], [0]]),
            torch.tensor([True, True, False]),
            does_not_raise(),
        ),
        (
            torch.tensor([[1, 2, 3, 4]]),
            1,
            torch.tensor([[1]]),
            torch.tensor([True]),
            does_not_raise(),
        ),
        (
            torch.tensor([[1, 2, 3, 4]]),
            1,
            None,
            torch.tensor([True]),
            does_not_raise(),
        ),
        (
            torch.tensor([0]),
            1,
            None,
            None,
            pytest.raises(AssertionError),
        ),
        (
            torch.tensor([[]]),
            1,
            None,
            torch.tensor([False]),
            pytest.raises(AssertionError),
        ),
    ],
)
def test_existence_functions(
    traces, activity, prefixes, expected_result, expected_raise
):
    with expected_raise:
        assert torch.equal(
            existence(traces, activity=activity, prefixes=prefixes), expected_result
        )


@pytest.mark.parametrize(
    "traces, activity, prefixes, expected_str, expected_result",
    [
        (
            torch.tensor([[[-torch.inf, 1, -torch.inf, -torch.inf, -torch.inf]]]),
            1,
            None,
            "ExistenceConstraint(solver=WeightedSamplingSolver(num_samples=1), settings=ExistenceConstraintSettings(activity=1))",
            torch.tensor(0),
        ),
        (
            torch.tensor([[[-torch.inf, -torch.inf, 1, -torch.inf, -torch.inf]]]),
            1,
            None,
            "ExistenceConstraint(solver=WeightedSamplingSolver(num_samples=1), settings=ExistenceConstraintSettings(activity=1))",
            torch.tensor(torch.inf),
        ),
        (
            torch.tensor(
                [
                    [[-torch.inf, 1, -torch.inf, -torch.inf, -torch.inf]],
                    [[-torch.inf, -torch.inf, -torch.inf, -torch.inf, 1]],
                ]
            ),
            1,
            None,
            "ExistenceConstraint(solver=WeightedSamplingSolver(num_samples=1), settings=ExistenceConstraintSettings(activity=1))",
            torch.tensor(torch.inf),
        ),
        (
            torch.tensor(
                [
                    [[-torch.inf, 1, -torch.inf, -torch.inf, -torch.inf]],
                    [[-torch.inf, -torch.inf, -torch.inf, -torch.inf, 1]],
                ]
            ),
            1,
            torch.tensor([[9], [1]]),
            "ExistenceConstraint(solver=WeightedSamplingSolver(num_samples=1), settings=ExistenceConstraintSettings(activity=1))",
            torch.tensor(0),
        ),
    ],
)
def test_existence_constraint(
    traces, activity, prefixes, expected_str, expected_result
):
    settings = ExistenceConstraintSettings(activity=activity)
    existence_constraint = ExistenceConstraint(
        settings, solver=WeightedSamplingSolver(num_samples=1)
    )

    assert str(existence_constraint) == expected_str
    assert torch.equal(existence_constraint(traces, prefixes=prefixes), expected_result)
