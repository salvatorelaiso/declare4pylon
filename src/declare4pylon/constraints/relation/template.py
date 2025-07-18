from declare4pylon.constraints import DeclareConstraint


class RelationTemplate(DeclareConstraint):
    """Base class for relation constraints.

    This class is used as a template for defining relation constraints in the
    Declare4Pylon library.
    It is not intended to be used directly, but rather as a base class for
    specific relation constraints.
    """

    def __init__(self, activity_a: int, activity_b: int) -> None:
        """Initialize the RelationTemplate with two activities.

        Args:
        ----
            activity_a (int): The first activity to be used in the relation constraint.
            activity_b (int): The second activity to be used in the relation constraint.


        """
        self.activity_a = activity_a
        self.activity_b = activity_b
