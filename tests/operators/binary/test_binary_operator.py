import pytest

from declare4pylon.operators.binary import BinaryOperator


def test_cannot_instantiate_unary_operator():
    # we expect an error when trying to instantiate the base class BinaryOperator
    # since it is abstract
    with pytest.raises(TypeError):
        BinaryOperator()
