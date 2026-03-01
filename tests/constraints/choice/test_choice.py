import pytest
import torch

from declare4pylon.constraints.choice import ChoiceConstraint
from tests.constants import A, B, C, D


@pytest.mark.parametrize(
    ("activity_a", "activity_b", "traces", "expected"),
    [
        (A, B, [[B, C, A, A, C]], [True]),
        (A, B, [[C, D, C]], [False]),
        (A, B, [[B, C, C]], [True]),
    ],
    ids=[
        "Choice (Both A and B)",
        "Choice (Neither A nor B)",
        "Choice (Only B)",
    ],
)
def test_choice_constraint_evaluate(
    activity_a: int,
    activity_b: int,
    traces: list[list[int]],
    expected: list[bool],
):
    traces_tensor = torch.tensor(traces, dtype=torch.int32)
    expected_tensor = torch.tensor(expected, dtype=torch.bool)
    constraint = ChoiceConstraint(activity_a=activity_a, activity_b=activity_b)
    result = constraint.evaluate(traces_tensor)
    assert torch.equal(result, expected_tensor)
