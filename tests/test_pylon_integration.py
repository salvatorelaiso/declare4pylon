import pytest
import torch
from pylon.constraint import constraint
from pylon.sampling_solver import WeightedSamplingSolver

from declare4pylon.existence import existence

A = 1
B = 2
C = 3
D = 4
_ = 0


@pytest.mark.parametrize(
    "logits, a, prefixes, expected_loss",
    [
        (
            torch.tensor(
                [
                    [
                        [-torch.inf, 1, -torch.inf, -torch.inf, -torch.inf],
                    ],
                    [
                        [-torch.inf, -torch.inf, 1, -torch.inf, -torch.inf],
                    ],
                ],
                dtype=torch.float32,
            ),
            A,
            None,
            # While the first row has the activity A, the second row does not have it.
            # The loss obtained is `inf`
            torch.tensor(torch.inf),
        ),
        (
            torch.tensor(
                [
                    [
                        [-torch.inf, 1, -torch.inf, -torch.inf, -torch.inf],
                    ],
                    [
                        [-torch.inf, 1, -torch.inf, -torch.inf, -torch.inf],
                    ],
                ],
                dtype=torch.float32,
            ),
            A,
            None,
            # Both rows have the activity A. The loss obtained is `0`
            torch.tensor(0),
        ),
    ],
)
def test_pylon_existence_single_prediction(logits, a, prefixes, expected_loss):
    def existence_constraint(
        sampled: torch.IntTensor, kwargs: dict
    ) -> torch.BoolTensor:
        return existence(sampled, **kwargs)

    existence_constraint = constraint(
        cond=existence_constraint, solver=WeightedSamplingSolver(num_samples=1)
    )

    assert torch.equal(existence_constraint(logits, activity=a), expected_loss)
