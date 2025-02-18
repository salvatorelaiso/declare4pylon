from contextlib import nullcontext as does_not_raise

import pytest
import torch

from declare4pylon.relation.precedence import (
    alternate_precedence,
    chain_precedence,
    precedence,
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
                    [B, C, A, A, C, _, _],
                    [C, A, A, C, B, _, _],
                    [C, A, C, _, _, _, _],
                ]
            ),
            A,
            B,
            None,
            torch.tensor([False, True, True]),
            does_not_raise(),
        )
    ],
)
def test_precedence(traces, a, b, prefixes, expected_result, expected_raise):
    with expected_raise:
        assert torch.equal(
            precedence(traces, a=a, b=b, prefixes=prefixes), expected_result
        )


@pytest.mark.parametrize(
    "traces, a, b,prefixes, expected_result, expected_raise",
    [
        (
            torch.tensor(
                [
                    [B, C, C, _, _, _],
                    [B, C, A, A, C, _],
                    [C, A, A, C, B, _],
                    [C, A, C, B, _, _],
                    [C, A, B, C, A, _],
                    [C, A, C, B, A, B],
                ]
            ),
            A,
            B,
            None,
            torch.tensor([False, False, True, True, True, True]),
            does_not_raise(),
        )
    ],
)
def test_alternate_precedence(traces, a, b, prefixes, expected_result, expected_raise):
    with expected_raise:
        assert torch.equal(
            alternate_precedence(traces, a=a, b=b, prefixes=prefixes), expected_result
        )


@pytest.mark.parametrize(
    "traces, a, b, prefixes, expected_result, expected_raise",
    [
        (
            torch.tensor(
                [
                    [B, C, A, A, C, _, _],
                    [B, C, A, A, B, C, _],
                    [C, A, B, A, B, C, A],
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
def test_chain_precedence(traces, a, b, prefixes, expected_result, expected_raise):
    with expected_raise:
        assert torch.equal(
            chain_precedence(traces, a=a, b=b, prefixes=prefixes), expected_result
        )
