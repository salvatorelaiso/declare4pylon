import pytest

from declare4pylon.constraints.existence import (
    ExistenceTemplate,
    ExistenceTemplateWithCount,
)


def test_cannot_instantiate_existence_template():
    # we expect an error when trying to instantiate the base class ExistenceTemplate
    # since it is abstract
    with pytest.raises(TypeError):
        ExistenceTemplate()
    # we expect an error when trying to instantiate the base class ExistenceTemplateWithCount
    # since it is abstract
    with pytest.raises(TypeError):
        ExistenceTemplateWithCount()
