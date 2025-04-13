import pytest
import torch

from declare4pylon.constraints.relation import SuccessionConstraint
from tests.constants import PAD as _  # noqa: N811
from tests.constants import A, B


@pytest.mark.parametrize(
    ("activity_a", "activity_b", "traces", "expected"),
    [
        (A, B, [[_, _, _]], [True]),
        (A, B, [[A, _, _]], [False]),
        (A, B, [[_, _, B]], [False]),
        (A, B, [[A, _, B]], [True]),
        (A, B, [[B, _, A]], [False]),
        (A, B, [[A, A, B]], [True]),
        (A, B, [[A, B, A]], [False]),
        (A, B, [[B, A, B]], [False]),
        (A, B, [[A, B, B]], [True]),
    ],
    ids=[
        "Succession (Neither A nor B)",
        "Succession (Only A)",
        "Succession (Only B)",
        "Succession (A and B)",
        "Succession (B and A)",
        "Succession (A and another A before B)",
        "Succession (A and B before another A)",
        "Succession (first B, then A before B)",
        "Succession (first A, then two Bs)",
    ],
)
def test_succession_constraint_evaluate(
    activity_a: int,
    activity_b: int,
    traces: list[list[int]],
    expected: list[bool],
):
    traces_tensor = torch.tensor(traces, dtype=torch.int32)
    expected_tensor = torch.tensor(expected, dtype=torch.bool)
    constraint = SuccessionConstraint(activity_a=activity_a, activity_b=activity_b)
    result = constraint.evaluate(traces_tensor)
    assert torch.equal(result, expected_tensor)
