import torch
from jaxtyping import Bool, Int

from .template import ExistenceTemplate


def _is_activity_at_column(
    traces: Int[torch.Tensor, " batch activities"],
    activity: int,
    column: int,
) -> Bool[torch.Tensor, " batch"]:
    return traces[:, column] == activity


class InitConstraint(ExistenceTemplate):
    """Constraint to check if an activity is the first occurrence in traces.

    This constraint states that a certain activity must be the first occurrence in the
    traces.

    If you want to check if the activity is the first occurrence skipping the first
    column (e.g. when the first column is the start-of-sequence (sos) activity), use the
    :class:`InitConstraintAfterSpecialToken` class.

    Args:
    ----
        activity (int): The activity to be used in the init constraint.

    """

    def _condition(
        self,
        traces: Int[torch.Tensor, "batch activities"],
    ) -> Bool[torch.Tensor, " batch"]:
        return _is_activity_at_column(
            traces=traces,
            activity=self.activity,
            column=0,
        )


class InitConstraintAfterSpecialToken(ExistenceTemplate):
    """Constraint to check if an activity is the first occurrence in traces.

    This constraint states that a certain activity must be the first occurrence in the
    traces.
    This constraint assumes that the first column of the traces is the start-of-sequence
    (sos) activity or another special token.

    If you want to check if the activity is the first occurrence without the special token, use
    the :class:`InitConstraint` class.

    Args:
    ----
        activity (int): The activity to be used in the init constraint.

    """

    def _condition(
        self,
        traces: Int[torch.Tensor, "batch activities"],
    ) -> Bool[torch.Tensor, " batch"]:
        return _is_activity_at_column(
            traces=traces,
            activity=self.activity,
            column=1,
        )
