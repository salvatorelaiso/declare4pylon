import torch
from jaxtyping import Bool, Int

from declare4pylon.operators import BinaryAnd
from tests.conftest import MockLogicExpression


class MockOperandIdentity(MockLogicExpression):
    def _condition(self, traces: Int[torch.Tensor, "batch activities"]) -> Bool[torch.Tensor, " batch"]:
        return traces.bool()


class MockOperandNegation(MockLogicExpression):
    def _condition(self, traces: Int[torch.Tensor, "batch activities"]) -> Bool[torch.Tensor, " batch"]:
        return ~traces.bool()


identity = MockOperandIdentity()
negation = MockOperandNegation()

all_true = torch.ones(3, dtype=torch.bool)
all_false = torch.zeros(3, dtype=torch.bool)


def test_evaluate_all_true():
    binary_and = BinaryAnd(identity, identity)
    result = binary_and.evaluate(all_true)
    expected = all_true
    assert torch.equal(result, expected)


def test_evaluate_mixed_values():
    mixed_values = torch.BoolTensor([True, False, True])

    binary_and = BinaryAnd(identity, identity)
    result = binary_and.evaluate(mixed_values)
    expected = mixed_values
    assert torch.equal(result, expected)

    binary_and = BinaryAnd(identity, negation)
    result = binary_and.evaluate(mixed_values)
    expected = all_false
    assert torch.equal(result, expected)


def test_evaluate_all_false():
    binary_and = BinaryAnd(identity, identity)
    result = binary_and.evaluate(all_false)
    expected = all_false
    assert torch.equal(result, expected)
