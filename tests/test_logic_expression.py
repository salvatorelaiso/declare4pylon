import pytest
import torch
from jaxtyping import Bool, Int

from tests.conftest import MockLogicExpression


@pytest.fixture
def all_true_logic_expression():
    class AllTrueLogicExpression(MockLogicExpression):
        def _condition(self, traces: Int[torch.Tensor, "batch activities"]) -> Bool[torch.Tensor, " batch"]:
            # return True if all elements of the trace are non-zero, False otherwise
            return torch.tensor([torch.all(trace != 0) for trace in traces], dtype=torch.bool)

    return AllTrueLogicExpression()


def test_evaluate_all_non_zero(all_true_logic_expression: MockLogicExpression):
    expr = all_true_logic_expression
    traces = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    assert torch.equal(expr.evaluate(traces), torch.BoolTensor([True, True, True]))


def test_evaluate_contains_zero(all_true_logic_expression: MockLogicExpression):
    expr = all_true_logic_expression
    traces = torch.tensor([[1, 0], [3, 4], [0, 0]], dtype=torch.int32)
    assert torch.equal(expr.evaluate(traces), torch.BoolTensor([False, True, False]))


def test_evaluate_empty_tensor(all_true_logic_expression: MockLogicExpression):
    expr = all_true_logic_expression
    traces = torch.tensor([[]], dtype=torch.int32).reshape(0, 0)
    assert torch.equal(expr.evaluate(traces), torch.BoolTensor([]))
