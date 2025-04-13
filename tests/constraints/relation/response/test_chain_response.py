import pytest
import torch

from declare4pylon.constraints.relation import ChainResponseConstraint
from tests.constants import PAD as _  # noqa: N811
from tests.constants import A, B


@pytest.mark.parametrize(
    ("activity_a", "activity_b", "traces", "expected"),
    [
        (A, B, [[_, _, _]], [True]),
        (A, B, [[A, _, _]], [False]),
        (A, B, [[_, _, B]], [True]),
        (A, B, [[A, B, _]], [True]),
        (A, B, [[B, A, _]], [False]),
        (A, B, [[A, _, B]], [False]),
        (A, B, [[B, _, A]], [False]),
        (A, B, [[A, A, B]], [False]),
        (A, B, [[A, B, A]], [False]),
        (A, B, [[B, A, B]], [True]),
    ],
    ids=[
        "ChainResponse (Neither A nor B)",
        "ChainResponse (Only A)",
        "ChainResponse (Only B)",
        "ChainResponse (A and immediately B)",
        "ChainResponse (B and immediately A)",
        "ChainResponse (A and B but not immediately)",
        "ChainResponse (B and A but not immediately)",
        "ChainResponse (A and another A before B)",
        "ChainResponse (A and B before another A)",
        "ChainResponse (first B, then A before A)",
    ],
)
def test_chain_response_constraint_evaluate(
    activity_a: int,
    activity_b: int,
    traces: list[list[int]],
    expected: list[bool],
):
    traces_tensor = torch.tensor(traces, dtype=torch.int32)
    expected_tensor = torch.tensor(expected, dtype=torch.bool)
    constraint = ChainResponseConstraint(activity_a=activity_a, activity_b=activity_b)
    result = constraint.evaluate(traces_tensor)
    assert torch.equal(result, expected_tensor)
