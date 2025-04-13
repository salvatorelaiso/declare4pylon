import torch

from declare4pylon import LogicExpression
from declare4pylon.operators import BinaryAnd


class MockOperandIdentity(LogicExpression):
    def _condition(self, traces: torch.BoolTensor) -> torch.BoolTensor:
        return traces


class MockOperandNegation(LogicExpression):
    def _condition(self, traces: torch.BoolTensor) -> torch.BoolTensor:
        return ~traces


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
