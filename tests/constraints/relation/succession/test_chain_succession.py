import pytest
import torch

from declare4pylon.constraints.relation import ChainSuccessionConstraint
from tests.constants import PAD as _  # noqa: N811
from tests.constants import A, B


@pytest.mark.parametrize(
    ("activity_a", "activity_b", "traces", "expected"),
    [
        (A, B, [[_, _, _]], [True]),
        (A, B, [[A, _, _]], [False]),
        (A, B, [[_, _, B]], [False]),
        (A, B, [[A, _, B]], [False]),
        (A, B, [[B, _, A]], [False]),
        (A, B, [[A, A, B]], [False]),
        (A, B, [[A, B, A]], [False]),
        (A, B, [[B, A, B]], [False]),
    ],
    ids=[
        "ChainSuccession (Neither A nor B)",
        "ChainSuccession (Only A)",
        "ChainSuccession (Only B)",
        "ChainSuccession (A and B)",
        "ChainSuccession (B and A)",
        "ChainSuccession (A and another A before B)",
        "ChainSuccession (A and B before another A)",
        "ChainSuccession (first B, then A before A)",
    ],
)
def test_alternate_succession_constraint_evaluate(
    activity_a: int,
    activity_b: int,
    traces: list[list[int]],
    expected: list[bool],
):
    traces_tensor = torch.tensor(traces, dtype=torch.int32)
    expected_tensor = torch.tensor(expected, dtype=torch.bool)
    constraint = ChainSuccessionConstraint(activity_a=activity_a, activity_b=activity_b)
    result = constraint.evaluate(traces_tensor)
    assert torch.equal(result, expected_tensor)
