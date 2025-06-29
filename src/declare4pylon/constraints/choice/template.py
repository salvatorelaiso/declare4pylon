from declare4pylon.constraints import DeclareConstraint


class ChoiceTemplate(DeclareConstraint):
    """Base class for choice constraints.

    This class is used as a template for defining choice constraints in the
    Declare4Pylon library.
    It is not intended to be used directly, but rather as a base class for
    specific choice constraints.
    """

    def __init__(self, activity_a: int, activity_b: int) -> None:
        """Initialize the ChoiceTemplate with two activities.

        Args:
        ----
            activity_a (int): The first activity to be used in the choice constraint.
            activity_b (int): The second activity to be used in the choice constraint.


        """
        self.activity_a = activity_a
        self.activity_b = activity_b
