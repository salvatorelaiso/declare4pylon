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

    def __repr__(self) -> str:
        """Return a string representation for debugging."""
        return f"{self.__class__.__name__}(activity_a={self.activity_a}, activity_b={self.activity_b})"

    def __str__(self) -> str:
        """Return a string representation of the choice template."""
        return f"{self.__class__.__name__}({self.activity_a}, {self.activity_b})"

    def __eq__(self, other: object) -> bool:
        """Check equality with another choice template based on the activities."""
        if not isinstance(other, self.__class__):
            return False
        return self.activity_a == other.activity_a and self.activity_b == other.activity_b

    def __hash__(self) -> int:
        """Return the hash based on the operator class and its activities."""
        return hash((self.__class__, self.activity_a, self.activity_b))
