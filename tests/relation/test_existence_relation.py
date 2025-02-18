from contextlib import nullcontext as does_not_raise

import pytest
import torch

from declare4pylon.relation.existence import co_existence, responded_existence

A = 1
B = 2
C = 3
D = 4
_ = 0


@pytest.mark.parametrize(
    "traces, a, b, prefixes, expected_result, expected_raise",
    [
        (
            torch.tensor(
                [
                    [C, A, C, _, _],
                    [C, A, A, C, B],
                    [B, C, A, C, _],
                    [B, C, C, _, _],
                ]
            ),
            A,
            B,
            None,
            torch.tensor([False, True, True, True]),
            does_not_raise(),
        ),
        (
            torch.tensor(
                [
                    [C, _, _],
                    [A, C, B],
                    [A, C, _],
                    [C, _, _],
                ]
            ),
            A,
            B,
            torch.tensor([[C, A], [C, A], [B, C], [B, C]]),
            torch.tensor([False, True, True, True]),
            does_not_raise(),
        ),
    ],
)
def test_responded_existence(traces, a, b, prefixes, expected_result, expected_raise):
    with expected_raise:
        assert torch.equal(
            responded_existence(traces, a=a, b=b, prefixes=prefixes), expected_result
        )


@pytest.mark.parametrize(
    "traces, a, b, prefixes, expected_result",
    [
        (
            torch.tensor(
                [
                    [B, C, A, A, C],
                    [A, C, D, _, _],
                    [B, C, C, _, _],
                ]
            ),
            A,
            B,
            None,
            torch.tensor([True, False, False]),
        ),
        (
            torch.tensor(
                [
                    [A, A, C],
                    [D, _, _],
                    [C, _, _],
                ]
            ),
            A,
            B,
            torch.tensor([[B, C], [A, C], [B, C]]),
            torch.tensor([True, False, False]),
        ),
    ],
)
def test_co_existence(traces, a, b, prefixes, expected_result):
    assert torch.equal(
        co_existence(traces, a=a, b=b, prefixes=prefixes), expected_result
    )
