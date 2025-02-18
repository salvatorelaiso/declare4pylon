from contextlib import nullcontext as does_not_raise

import pytest
import torch

from declare4pylon.existence.absence import absence


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
            absence(traces, activity=activity, prefixes=prefixes), ~expected_result
        )
