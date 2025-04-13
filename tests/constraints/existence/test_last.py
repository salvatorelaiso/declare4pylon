import pytest
import torch

from declare4pylon.constraints.existence import LastConstraint
from tests.constants import EOS, A, B
from tests.constants import PAD as _  # noqa: N811


@pytest.mark.parametrize(
    ("activity", "pad", "eos", "traces", "expected"),
    [
        (
            A,
            _,
            None,
            [[_, _], [A, _], [_, A], [A, A]],
            [False, True, True, True],
        ),
        (
            A,
            B,
            None,
            [[_, _], [A, B], [_, A], [A, A]],
            [False, True, True, True],
        ),
        (
            A,
            _,
            EOS,
            [[_, _, EOS], [A, _, EOS], [_, A, EOS], [A, A, EOS]],
            [False, False, True, True],
        ),
        (
            A,
            _,
            EOS,
            [[_, _, _, EOS], [_, A, _, EOS], [_, A, EOS, _], [A, A, EOS, _]],
            [False, False, True, True],
        ),
    ],
    ids=[
        "Last A",
        "Last A with modified padding",
        "Last A with EOS",
        "Last A with EOS and padding",
    ],
)
def test_last_constraint_evaluate(
    activity: int,
    pad: int,
    eos: int | None,
    traces: list[list[int]],
    expected: list[bool],
):
    traces_tensor = torch.tensor(traces, dtype=torch.int32)
    expected_tensor = torch.tensor(expected, dtype=torch.bool)
    constraint = LastConstraint(activity=activity, pad=pad, eos=eos)
    result = constraint.evaluate(traces_tensor)
    assert torch.equal(result, expected_tensor)
