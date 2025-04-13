import pytest
import torch

from declare4pylon.constraints.relation import RespondedExistenceConstraint
from tests.constants import PAD as _  # noqa: N811
from tests.constants import A, B


@pytest.mark.parametrize(
    ("activity_a", "activity_b", "traces", "expected"),
    [
        (A, B, [[_, _, _]], [True]),
        (A, B, [[A, _, _]], [False]),
        (A, B, [[_, _, B]], [True]),
        (A, B, [[A, _, B]], [True]),
        (A, B, [[B, _, A]], [True]),
    ],
    ids=[
        "RespondedExistenceConstraint (Neither A nor B)",
        "RespondedExistenceConstraint (Only A)",
        "RespondedExistenceConstraint (Only B)",
        "RespondedExistenceConstraint (A and B)",
        "RespondedExistenceConstraint (B and A)",
    ],
)
def test_responded_existence_constraint_evaluate(
    activity_a: int,
    activity_b: int,
    traces: list[list[int]],
    expected: list[bool],
):
    traces_tensor = torch.tensor(traces, dtype=torch.int32)
    expected_tensor = torch.tensor(expected, dtype=torch.bool)
    constraint = RespondedExistenceConstraint(activity_a=activity_a, activity_b=activity_b)
    result = constraint.evaluate(traces_tensor)
    assert torch.equal(result, expected_tensor)
