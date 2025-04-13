"""A module which defines templates for existence constraints.

This module provides:
- ExistenceTemplate: A base class for existence constraints.
- ExistenceTemplateWithCount: A base class for existence constraints with a count.

These templates are not intended to be used directly but serve as base classes for
specific constraint implementations.
The ExistenceTemplate class is used for defining existence constraints, while the
ExistenceTemplateWithCount class is used for defining existence constraints that
require a count of occurrences.
"""

from declare4pylon.constraints import DeclareConstraint


class ExistenceTemplate(DeclareConstraint):
    """Base class for existence constraints.

    This class is used as a template for defining existence constraints in the
    Declare4Pylon library.
    It is not intended to be used directly, but rather as a base class for
    specific existence constraints.
    """

    def __init__(self, activity: int) -> None:
        """Initialize the ExistenceTemplate with an activity.

        Args:
        ----
            activity (int): The activity to be used in the existence constraint.

        """
        self.activity = activity


class ExistenceTemplateWithCount(ExistenceTemplate):
    """Base class for existence constraints with a count.

    This class is used as a template for defining existence constraints in the
    Declare4Pylon library that require a count of occurrences.
    It is not intended to be used directly, but rather as a base class for
    specific existence constraints with a count.
    """

    def __init__(self, activity: int, count: int = 1) -> None:
        """Initialize the ExistenceTemplateWithCount with an activity and count.

        Args:
        ----
            activity (int): The activity to be used in the existence constraint.
            count (int): The number of occurrences to be considered in the evaluation.
                Default is 1.

        Raises:
        ------
            ValueError: If count is not a positive integer.

        """
        if count <= 0:
            msg = "Count must be a positive integer."
            raise ValueError(msg)
        super().__init__(activity=activity)
        self.count = count
