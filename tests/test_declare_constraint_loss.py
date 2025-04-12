import torch
from pylon.sampling_solver import WeightedSamplingSolver

from declare4pylon.declare_constraint_loss import DeclareConstraintLoss
from declare4pylon.logic_expression import LogicExpression


class MockSatisfiedLogicExpression(LogicExpression):
    def _condition(self, traces: torch.IntTensor) -> torch.BoolTensor:
        # Simple implementation for testing: return True for each element of the batch
        return torch.ones(traces.shape[0], dtype=torch.bool)


class MockViolatedLogicExpression(LogicExpression):
    def _condition(self, traces: torch.IntTensor) -> torch.BoolTensor:
        # Simple implementation for testing: return False for each element of the batch
        return torch.zeros(traces.shape[0], dtype=torch.bool)


class MockMixedLogicExpression(LogicExpression):
    def _condition(self, traces: torch.IntTensor) -> torch.BoolTensor:
        # Simple implementation for testing: return alternating True and False for each element of the batch
        return torch.tensor([True, False] * (traces.shape[0] // 2), dtype=torch.bool)


class MockAlternatingLogicExpression(LogicExpression):
    sat = True

    def _condition(self, traces: torch.IntTensor) -> torch.BoolTensor:
        # Simple implementation for testing: return alternating True and False for each sampling
        # performed by the solver
        res = torch.tensor([self.sat] * traces.shape[0], dtype=torch.bool)
        self.sat = not self.sat
        return res


def test_evaluate_all_satisfied():
    constraint = DeclareConstraintLoss(
        MockSatisfiedLogicExpression(),
        WeightedSamplingSolver(num_samples=1),
    )
    loss = constraint(torch.rand(32, 3))
    assert loss == 0.0


def test_evaluate_all_violated():
    constraint = DeclareConstraintLoss(
        MockViolatedLogicExpression(),
        WeightedSamplingSolver(num_samples=1),
    )
    loss = constraint(torch.rand(32, 3))
    assert loss == float("inf")


def test_evaluate_mixed():
    constraint = DeclareConstraintLoss(
        MockMixedLogicExpression(),
        WeightedSamplingSolver(num_samples=1),
    )
    loss = constraint(torch.rand(32, 3))
    # loss is inf since the (single) sample does not satisfy the constraint
    # on some of the traces
    assert loss == float("inf")


def test_evaluate_mixed_with_sampling():
    constraint = DeclareConstraintLoss(
        MockAlternatingLogicExpression(),
        WeightedSamplingSolver(num_samples=2),
    )
    loss = constraint(torch.rand(32, 3))
    # 0 < loss < inf since one of the samples satisfies the constraint
    # while the other does not
    assert 0 < loss < float("inf")
