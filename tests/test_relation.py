from contextlib import nullcontext as does_not_raise

import pytest
import torch

from declare4pylon.relation import (
    alternate_precedence,
    alternate_response,
    alternate_succession,
    chain_precedence,
    chain_response,
    chain_succession,
    co_existence,
    not_chain_succession,
    not_co_existence,
    not_succession,
    precedence,
    responded_existence,
    response,
    succession,
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
    "traces, a, b, prefixes, expected_result, expected_raise",
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
            does_not_raise(),
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
            does_not_raise(),
        ),
    ],
)
def test_co_existence(traces, a, b, prefixes, expected_result, expected_raise):
    with expected_raise:
        assert torch.equal(
            co_existence(traces, a=a, b=b, prefixes=prefixes), expected_result
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
                    [C, D, C, _, _],
                ]
            ),
            A,
            B,
            None,
            torch.tensor([False, True, False, False, True]),
            does_not_raise(),
        )
    ],
)
def test_succession(traces, a, b, prefixes, expected_result, expected_raise):
    with expected_raise:
        assert torch.equal(
            succession(traces, a=a, b=b, prefixes=prefixes), expected_result
        )


@pytest.mark.parametrize(
    "traces, a, b, prefixes, expected_result, expected_raise",
    [
        (
            torch.tensor(
                [
                    [B, C, A, A, C, _],
                    [C, A, A, C, B, _],
                    [C, A, C, B, _, _],
                    [C, A, B, C, A, _],
                    [B, C, C, _, _, _],
                    [C, A, C, B, A, B],
                ]
            ),
            A,
            B,
            None,
            torch.tensor([False, False, True, False, False, True]),
            does_not_raise(),
        )
    ],
)
def test_alternate_succession(traces, a, b, prefixes, expected_result, expected_raise):
    with expected_raise:
        assert torch.equal(
            alternate_succession(traces, a=a, b=b, prefixes=prefixes), expected_result
        )


@pytest.mark.parametrize(
    "traces, a, b, prefixes, expected_result, expected_raise",
    [
        (
            torch.tensor(
                [
                    [B, C, A, A, C, _],
                    [B, C, A, A, B, C],
                    [C, A, B, A, B, C],
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
def test_chain_succession(traces, a, b, prefixes, expected_result, expected_raise):
    with expected_raise:
        assert torch.equal(
            chain_succession(traces, a=a, b=b, prefixes=prefixes), expected_result
        )


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
