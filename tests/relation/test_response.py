from contextlib import nullcontext as does_not_raise

import pytest
import torch

from declare4pylon.relation.response import alternate_response, chain_response, response

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
                    [B, C, A, A, C],
                    [C, A, A, C, B],
                    [C, A, C, _, _],
                    [B, C, C, _, _],
                ]
            ),
            A,
            B,
            None,
            torch.tensor([False, True, False, True]),
            does_not_raise(),
        ),
        (
            torch.tensor(
                [
                    [A, A, C],
                    [A, C, B],
                    [C, _, _],
                    [C, _, _],
                ]
            ),
            A,
            B,
            torch.tensor([[B, C], [C, A], [C, A], [B, C]]),
            torch.tensor([False, True, False, True]),
            does_not_raise(),
        ),
    ],
)
def test_response_function(traces, a, b, prefixes, expected_result, expected_raise):
    with expected_raise:
        assert torch.equal(
            response(traces, a=a, b=b, prefixes=prefixes), expected_result
        )


@pytest.mark.parametrize(
    "traces, a, b, prefixes, expected_result, expected_raise",
    [
        (
            torch.tensor(
                [
                    [B, C, A, A, C, _, _],
                    [C, A, A, C, B, _, _],
                    [C, A, C, B, _, _, _],
                    [C, A, B, C, A, _, _],
                    [B, C, C, _, _, _, _],
                    [C, A, C, B, B, A, B],
                ]
            ),
            A,
            B,
            None,
            torch.tensor([False, False, True, False, True, True]),
            does_not_raise(),
        )
    ],
)
def test_alternate_response(traces, a, b, prefixes, expected_result, expected_raise):
    with expected_raise:
        assert torch.equal(
            alternate_response(traces, a=a, b=b, prefixes=prefixes), expected_result
        )


@pytest.mark.parametrize(
    "traces, a, b, prefixes, expected_result, expected_raise",
    [
        (
            torch.tensor(
                [
                    [B, C, A, A, C, _, _],
                    [B, C, A, A, B, C, _],
                    [B, C, A, B, A, B, C],
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
def test_chain_response(traces, a, b, prefixes, expected_result, expected_raise):
    with expected_raise:
        assert torch.equal(
            chain_response(traces, a=a, b=b, prefixes=prefixes), expected_result
        )
