from contextlib import nullcontext as does_not_raise

import pytest
import torch

from declare4pylon.existence.absence import absence


@pytest.mark.parametrize(
    "traces, activity, count, prefixes, expected_result, expected_raise",
    [
        (
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
            1,
            1,
            None,
            torch.tensor([False, True, True]),
            does_not_raise(),
        ),
        (
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
            1,
            2,
            None,
            torch.tensor([True, True, True]),
            does_not_raise(),
        ),
        (
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
            1,
            1,
            torch.tensor([[1], [1], [0]]),
            torch.tensor([False, False, True]),
            does_not_raise(),
        ),
        (
            torch.tensor([[1, 2, 3, 4]]),
            1,
            1,
            torch.tensor([[1]]),
            torch.tensor([False]),
            does_not_raise(),
        ),
        (
            torch.tensor([[1, 2, 3, 4]]),
            1,
            1,
            None,
            torch.tensor([False]),
            does_not_raise(),
        ),
        (
            torch.tensor([0]),
            1,
            1,
            None,
            None,
            pytest.raises(AssertionError),
        ),
        (
            torch.tensor([[]]),
            1,
            1,
            None,
            torch.tensor([True]),
            pytest.raises(AssertionError),
        ),
    ],
)
def test_existence_functions(
    traces, activity, count, prefixes, expected_result, expected_raise
):
    with expected_raise:
        assert torch.equal(
            absence(traces, activity=activity, count=count, prefixes=prefixes),
            expected_result,
        )
