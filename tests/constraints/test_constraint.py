import pytest

from declare4pylon.constraints.constraint import DeclareConstraint


def test_cannot_instantiate_constraint():
    # we expect an error when trying to instantiate the base class DeclareConstraint
    # since it is abstract
    with pytest.raises(TypeError):
        DeclareConstraint()
