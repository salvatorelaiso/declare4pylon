import torch

from declare4pylon import LogicExpression
from declare4pylon.operators import UnaryNot


def test_unary_not_evaluate_with_all_true():
    class MockLogicExpression(LogicExpression):
        def _condition(self, traces: torch.IntTensor) -> torch.BoolTensor:  # noqa: ARG002
            return torch.tensor([True, True, True], dtype=torch.bool)

    operand = MockLogicExpression()
    unary_not = UnaryNot(operand=operand)
    traces = torch.tensor([0, 0, 0], dtype=torch.int32)  # Dummy traces
    result = unary_not.evaluate(traces)

    assert torch.equal(result, torch.tensor([False, False, False], dtype=torch.bool))


def test_unary_not_evaluate_with_all_false():
    class MockLogicExpression(LogicExpression):
        def _condition(self, traces: torch.IntTensor) -> torch.BoolTensor:  # noqa: ARG002
            return torch.tensor([False, False, False], dtype=torch.bool)

    operand = MockLogicExpression()
    unary_not = UnaryNot(operand=operand)
    traces = torch.tensor([0, 0, 0], dtype=torch.int32)  # Dummy traces
    result = unary_not.evaluate(traces)

    assert torch.equal(result, torch.tensor([True, True, True], dtype=torch.bool))


def test_unary_not_evaluate_with_mixed_values():
    class MockLogicExpression(LogicExpression):
        def _condition(self, traces: torch.IntTensor) -> torch.BoolTensor:  # noqa: ARG002
            return torch.tensor([True, False, True], dtype=torch.bool)

    operand = MockLogicExpression()
    unary_not = UnaryNot(operand=operand)
    traces = torch.tensor([0, 0, 0], dtype=torch.int32)  # Dummy traces
    result = unary_not.evaluate(traces)

    assert torch.equal(result, torch.tensor([False, True, False], dtype=torch.bool))
