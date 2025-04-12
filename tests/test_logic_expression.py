import torch

from declare4pylon.logic_expression import LogicExpression


class MockLogicalExpression(LogicExpression):
    def _condition(self, traces: torch.IntTensor) -> torch.BoolTensor:
        # Simple implementation for testing: return True if all elements are non-zero
        return torch.all(traces != 0, dim=1)


def test_evaluate_all_non_zero():
    expr = MockLogicalExpression()
    traces = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    assert torch.equal(expr.evaluate(traces), torch.BoolTensor([True, True, True]))


def test_evaluate_contains_zero():
    expr = MockLogicalExpression()
    traces = torch.tensor([[1, 0], [3, 4], [0, 0]], dtype=torch.int32)
    assert torch.equal(expr.evaluate(traces), torch.BoolTensor([False, True, False]))


def test_evaluate_empty_tensor():
    expr = MockLogicalExpression()
    traces = torch.tensor([[]], dtype=torch.int32).reshape(0, 0)
    assert torch.equal(expr.evaluate(traces), torch.BoolTensor([]))
