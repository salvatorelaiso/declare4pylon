import torch
from jaxtyping import Bool, Int

from declare4pylon.operators import UnaryNot
from tests.conftest import MockLogicExpression


def test_unary_not_evaluate_with_all_true():
    class AlwaysTrue(MockLogicExpression):
        def _condition(self, traces: Int[torch.Tensor, "batch activities"]) -> Bool[torch.Tensor, " batch"]:
            return torch.ones(traces.shape[0], dtype=torch.bool)

    operand = AlwaysTrue()
    unary_not = UnaryNot(operand=operand)
    traces = torch.tensor([0, 0, 0], dtype=torch.int32)
    result = unary_not.evaluate(traces)

    assert torch.equal(result, torch.tensor([False, False, False], dtype=torch.bool))


def test_unary_not_evaluate_with_all_false():
    class AlwaysFalse(MockLogicExpression):
        def _condition(self, traces: Int[torch.Tensor, "batch activities"]) -> Bool[torch.Tensor, " batch"]:
            return torch.zeros(traces.shape[0], dtype=torch.bool)

    operand = AlwaysFalse()
    unary_not = UnaryNot(operand=operand)
    traces = torch.tensor([0, 0, 0], dtype=torch.int32)
    result = unary_not.evaluate(traces)

    assert torch.equal(result, torch.tensor([True, True, True], dtype=torch.bool))


def test_unary_not_evaluate_with_mixed_values():
    class Even(MockLogicExpression):
        def _condition(self, traces: Int[torch.Tensor, "batch activities"]) -> Bool[torch.Tensor, " batch"]:
            # Simple implementation for testing: return alternating True and False for each element of the batch.
            # Uses the length of the traces to determine how many True/False values to return to match the batch size.
            return torch.tensor([i % 2 == 0 for i in range(traces.shape[0])], dtype=torch.bool)

    operand = Even()
    unary_not = UnaryNot(operand=operand)
    traces = torch.tensor([0, 0, 0], dtype=torch.int32)
    result = unary_not.evaluate(traces)
    # Applying NOT to the alternating True/False values should yield alternating False/True values,
    # starting with False for the first element (since the first element is True in the operand).
    assert torch.equal(result, torch.tensor([False, True, False], dtype=torch.bool))
