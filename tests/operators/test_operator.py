import pytest

from declare4pylon.operators import Operator


def test_cannot_instantiate_operator():
    # we expect an error when trying to instantiate the base class Operator
    # since it is abstract
    with pytest.raises(TypeError):
        Operator()
