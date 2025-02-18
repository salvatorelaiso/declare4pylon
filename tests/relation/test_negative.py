from contextlib import nullcontext as does_not_raise

import pytest
import torch

from declare4pylon.relation.negative import (
    not_chain_succession,
    not_co_existence,
    not_succession,
)

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
                    [C, D, C, _, _],
                ]
            ),
            A,
            B,
            None,
            torch.tensor([False, False, True]),
            does_not_raise(),
        )
    ],
)
def test_not_co_existence(traces, a, b, prefixes, expected_result, expected_raise):
    with expected_raise:
        assert not torch.equal(
            not_co_existence(traces, a=a, b=b, prefixes=prefixes), expected_result
        )


@pytest.mark.parametrize(
    "traces, a, b, prefixes, expected_result, expected_raise",
    [
        (
            torch.tensor(
                [
                    [B, C, A, A, C],
                    [C, A, A, C, B],
                    [C, A, C, _, _],
                    [B, C, C, _, _],
                ]
            ),
            A,
            B,
            None,
            torch.tensor([True, False, True, True]),
            does_not_raise(),
        )
    ],
)
def test_not_succession(traces, a, b, prefixes, expected_result, expected_raise):
    with expected_raise:
        assert torch.equal(
            not_succession(traces, a=a, b=b, prefixes=prefixes), expected_result
        )


@pytest.mark.parametrize(
    "traces, a, b, prefixes, expected_result, expected_raise",
    [
        (
            torch.tensor(
                [
                    [B, C, A, A, C, _],
                    [B, C, A, A, B, C],
                    [C, B, A, C, B, C],
                ]
            ),
            A,
            B,
            None,
            torch.tensor([True, False, True]),
            does_not_raise(),
        )
    ],
)
def test_not_chain_succession(traces, a, b, prefixes, expected_result, expected_raise):
    with expected_raise:
        assert torch.equal(
            not_chain_succession(traces, a=a, b=b, prefixes=prefixes), expected_result
        )
