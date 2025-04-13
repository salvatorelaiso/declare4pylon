import torch
from jaxtyping import Bool, Int

from .template import ExistenceTemplate


class LastConstraint(ExistenceTemplate):
    """Constraint to check if an activity is the last occurrence in traces.

    This constraint states that a certain activity must be the last occurrence in the
    traces.
    It can also check if the last occurrence is followed by an end-of-sequence (eos)
    activity.
    The constraint can be used to enforce that the specified activity is the last
    occurrence in the traces, optionally followed by an eos activity.
    The constraint is defined by the activity to be checked, a padding value, and an
    optional eos value.

    Args:
    ----
        activity (int): The activity to be checked.
        pad (int, optional): The padding value to ignore in the traces.
            Defaults to 0.
        eos (int | None, optional): If provided, the constraint checks if the last
            occurrence is followed by the eos activity. Defaults to None.

    """

    def __init__(
        self,
        activity: int,
        pad: int = 0,
        eos: int | None = None,
    ) -> None:
        """Initialize the LastConstraint with an activity, padding value, and eos value.

        This constructor initializes the LastConstraint with the specified activity,
        padding value, and eos value.
        The padding values at the end of the traces are ignored when checking for the
        last occurrence of the activity.

        Args:
        ----
            activity (int): The activity to be checked.
            pad (int, optional): The padding value to ignore in the traces.
                Defaults to 0.
            eos (int | None, optional): _description_. Defaults to None.
                If provided, the constraint checks if the last occurrence is followed by
                the eos activity.

        """
        super().__init__(activity)
        self.pad_value = pad
        self.eos_value = eos

    def _condition(
        self,
        traces: Int[torch.Tensor, "batch activities"],
    ) -> Bool[torch.Tensor, " batch"]:
        # Check which indices are not padding
        non_pad_indices = torch.nonzero(traces != self.pad_value)
        row_indices, col_indices = non_pad_indices[:, 0], non_pad_indices[:, 1]

        # Get the last non-padding index for each row
        last_non_pad_index = torch.zeros(
            traces.shape[0],
            dtype=torch.long,
        ).scatter_reduce(0, row_indices, col_indices, reduce="amax", include_self=False)

        # Set to -1 if the row is all padding
        ignore_row = -1
        last_non_pad_index[(traces != self.pad_value).sum(dim=1) == 0] = ignore_row

        # Create the result tensor
        result = torch.zeros(traces.shape[0], dtype=torch.bool)

        for i, (last_index, row) in enumerate(
            zip(last_non_pad_index, traces, strict=False),
        ):
            # Ignore rows that are all padding
            if last_index == ignore_row:
                continue

            # Check if the last index is the activity
            if self.eos_value is None:
                result[i] = row[last_index] == self.activity

            # Check if the second last index is the activity and the last index is the
            # eos
            else:
                result[i] = (
                    last_index > 0 and row[last_index - 1] == self.activity and row[last_index] == self.eos_value
                )

        return result
