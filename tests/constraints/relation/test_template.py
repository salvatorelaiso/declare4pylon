import pytest

from declare4pylon.constraints.relation import RelationTemplate


def test_cannot_instantiate_relation_template():
    # we expect an error when trying to instantiate the base class RelationTemplate
    # since it is abstract
    with pytest.raises(TypeError):
        RelationTemplate()
