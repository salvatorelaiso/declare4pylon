import pytest
import torch

from declare4pylon.constraints.relation import AlternatePrecedenceConstraint
from tests.constants import PAD as _  # noqa: N811
from tests.constants import A, B


@pytest.mark.parametrize(
    ("activity_a", "activity_b", "traces", "expected"),
    [
        (A, B, [[_, _, _]], [True]),
        (A, B, [[A, _, _]], [True]),
        (A, B, [[_, _, B]], [False]),
        (A, B, [[A, _, B]], [True]),
        (A, B, [[B, _, A]], [False]),
        (A, B, [[A, A, B]], [True]),
        (A, B, [[A, B, A]], [True]),
        (A, B, [[B, A, B]], [False]),
        (A, B, [[A, B, B]], [False]),
    ],
    ids=[
        "AlternatePrecedence (Neither A nor B)",
        "AlternatePrecedence (Only A)",
        "AlternatePrecedence (Only B)",
        "AlternatePrecedence (A and B)",
        "AlternatePrecedence (B and A)",
        "AlternatePrecedence (A and another A before B)",
        "AlternatePrecedence (A and B before another A)",
        "AlternatePrecedence (first B, then A before A)",
        "AlternatePrecedence (first A, then two Bs)",
    ],
)
def test_alternate_precedence_constraint_evaluate(
    activity_a: int,
    activity_b: int,
    traces: list[list[int]],
    expected: list[bool],
):
    traces_tensor = torch.tensor(traces, dtype=torch.int32)
    expected_tensor = torch.tensor(expected, dtype=torch.bool)
    constraint = AlternatePrecedenceConstraint(activity_a=activity_a, activity_b=activity_b)
    result = constraint.evaluate(traces_tensor)
    assert torch.equal(result, expected_tensor)
