from contextlib import nullcontext as does_not_raise

import pytest
import torch

from declare4pylon.choice.exclusive_choice import exclusive_choice


@pytest.mark.parametrize(
    "traces, a, b, prefixes, expected_result, expected_raise",
    [
        (
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
            1,
            2,
            None,
            torch.tensor([False, False, False]),
            does_not_raise(),
        ),
        (
            torch.tensor([[1, 2, 3, 4], [1, 1, 1, 1], [5, 6, 7, 8]]),
            1,
            2,
            None,
            torch.tensor([False, True, False]),
            does_not_raise(),
        ),
        (
            torch.tensor([[1, 2, 3, 4], [1, 1, 1, 1], [5, 6, 7, 8]]),
            1,
            2,
            torch.tensor([[0, 0], [1, 2], [0, 2]]),
            torch.tensor([False, False, True]),
            does_not_raise(),
        ),
        (
            torch.tensor([[1, 2, 3, 4]]),
            0,
            4,
            None,
            torch.tensor([True]),
            does_not_raise(),
        ),
        (
            torch.tensor([[1, 2, 3, 4]]),
            1,
            4,
            None,
            torch.tensor([False]),
            does_not_raise(),
        ),
        (
            torch.tensor([0]),
            None,
            None,
            None,
            None,
            pytest.raises(AssertionError),
        ),
        (
            torch.tensor([[]]),
            None,
            None,
            None,
            None,
            pytest.raises(AssertionError),
        ),
    ],
)
def test_choice_functions(
    traces,
    a,
    b,
    prefixes,
    expected_result,
    expected_raise,
):
    with expected_raise:
        assert torch.equal(
            exclusive_choice(traces, a=a, b=b, prefixes=prefixes),
            expected_result,
        )
