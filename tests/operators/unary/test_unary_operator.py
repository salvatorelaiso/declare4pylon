import pytest

from declare4pylon.operators import UnaryOperator


def test_cannot_instantiate_unary_operator():
    # we expect an error when trying to instantiate the base class UnaryOperator
    # since it is abstract
    with pytest.raises(TypeError):
        UnaryOperator()
