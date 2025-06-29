import pytest
import torch

from declare4pylon.constraints.existence import (
    InitConstraint,
    InitConstraintAfterSpecialToken,
)
from tests.constants import PAD as _  # noqa: N811
from tests.constants import A

a_first_col_traces = [
    [A, _, _],
    [A, _, A],
]
a_second_col_traces = [
    [_, A, _],
    [_, A, A],
]

traces = a_first_col_traces + a_second_col_traces


@pytest.mark.parametrize(
    ("init_constraint_type", "activity", "traces", "expected"),
    [
        (
            InitConstraint,
            A,
            traces,
            [True] * len(a_first_col_traces) + [False] * len(a_second_col_traces),
        ),
        (
            InitConstraintAfterSpecialToken,
            A,
            traces,
            [False] * len(a_first_col_traces) + [True] * len(a_second_col_traces),
        ),
    ],
    ids=["Init A", "Init A after special token"],
)
def test_init_constraint_evaluate(
    init_constraint_type: type[InitConstraint],
    activity: int,
    traces: list[list[int]],
    expected: list[bool],
):
    traces_tensor = torch.tensor(traces, dtype=torch.int32)
    expected_tensor = torch.tensor(expected, dtype=torch.bool)
    constraint = init_constraint_type(activity=activity)
    result = constraint.evaluate(traces_tensor)
    assert torch.equal(result, expected_tensor)
