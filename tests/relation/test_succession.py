from contextlib import nullcontext as does_not_raise

import pytest
import torch
from pylon.sampling_solver import WeightedSamplingSolver

from declare4pylon.relation.settings import RelationConstraintSettings
from declare4pylon.relation.succession import (
    AlternateSuccessionConstraint,
    alternate_succession,
    chain_succession,
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


def test_alternate_succession_constraint():
    alternate_succession_constraint = AlternateSuccessionConstraint(
        settings=RelationConstraintSettings(a=A, b=B),
        solver=WeightedSamplingSolver(num_samples=1),
    )
    alternate_succession_constraint(torch.tensor([[[-torch.inf, -torch.inf, 1]]]))
