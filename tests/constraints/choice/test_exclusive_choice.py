import pytest
import torch

from declare4pylon.constraints.choice import ExclusiveChoiceConstraint
from tests.constants import A, B, C, D


@pytest.mark.parametrize(
    ("activity_a", "activity_b", "traces", "expected"),
    [
        (A, B, [[B, C, A, A, C]], [False]),
        (A, B, [[C, D, C]], [False]),
        (A, B, [[B, C, C]], [True]),
    ],
    ids=[
        "ExclusiveChoice (Both A and B)",
        "ExclusiveChoice (Neither A nor B)",
        "ExclusiveChoice (Only B)",
    ],
)
def test_exclusive_choice_constraint_evaluate(
    activity_a: int,
    activity_b: int,
    traces: list[list[int]],
    expected: list[bool],
):
    traces_tensor = torch.tensor(traces, dtype=torch.int32)
    expected_tensor = torch.tensor(expected, dtype=torch.bool)
    constraint = ExclusiveChoiceConstraint(activity_a=activity_a, activity_b=activity_b)
    result = constraint.evaluate(traces_tensor)
    assert torch.equal(result, expected_tensor)
