import pytest
import torch

from declare4pylon.constraints.existence import ExactlyConstraint
from tests.constants import PAD as _  # noqa: N811
from tests.constants import A

traces = [[_, _, _], [A, _, _], [A, A, _], [A, A, A]]
a_counters = [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("activity", "count", "traces", "expected"),
    [
        (A, 1, traces, [count == 1 for count in a_counters]),
        (A, 2, traces, [count == 2 for count in a_counters]),
        (A, 3, traces, [count == 3 for count in a_counters]),
        (A, 4, traces, [count == 4 for count in a_counters]),
    ],
    ids=["Exactly 1", "Exactly 2", "Exactly 3", "Exactly 4"],
)
def test_exactly_constraint_evaluate(
    activity: int,
    count: int,
    traces: list[list[int]],
    expected: list[bool],
):
    traces_tensor = torch.tensor(traces, dtype=torch.int32)
    expected_tensor = torch.tensor(expected, dtype=torch.bool)
    constraint = ExactlyConstraint(activity=activity, count=count)
    result = constraint.evaluate(traces_tensor)
    assert torch.equal(result, expected_tensor)
