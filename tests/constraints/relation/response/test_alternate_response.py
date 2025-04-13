import pytest
import torch

from declare4pylon.constraints.relation import AlternateResponseConstraint
from tests.constants import PAD as _  # noqa: N811
from tests.constants import A, B


@pytest.mark.parametrize(
    ("activity_a", "activity_b", "traces", "expected"),
    [
        (A, B, [[_, _, _]], [True]),
        (A, B, [[A, _, _]], [False]),
        (A, B, [[_, _, B]], [True]),
        (A, B, [[A, _, B]], [True]),
        (A, B, [[B, _, A]], [False]),
        (A, B, [[A, A, B]], [False]),
        (A, B, [[A, B, A]], [False]),
        (A, B, [[B, A, B]], [True]),
    ],
    ids=[
        "AlternateResponse (Neither A nor B)",
        "AlternateResponse (Only A)",
        "AlternateResponse (Only B)",
        "AlternateResponse (A and B)",
        "AlternateResponse (B and A)",
        "AlternateResponse (A and another A before B)",
        "AlternateResponse (A and B before another A)",
        "AlternateResponse (first B, then A before A)",
    ],
)
def test_alternate_response_constraint_evaluate(
    activity_a: int,
    activity_b: int,
    traces: list[list[int]],
    expected: list[bool],
):
    traces_tensor = torch.tensor(traces, dtype=torch.int32)
    expected_tensor = torch.tensor(expected, dtype=torch.bool)
    constraint = AlternateResponseConstraint(activity_a=activity_a, activity_b=activity_b)
    result = constraint.evaluate(traces_tensor)
    assert torch.equal(result, expected_tensor)
