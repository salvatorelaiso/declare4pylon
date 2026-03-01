from abc import ABC, abstractmethod

import torch
from jaxtyping import Bool, Int


class LogicExpression(ABC):
    """Abstract base class for logic expressions.

    This class defines the interface for logic expressions used in the Declare4Pylon library.
    Subclasses must implement the `_condition` method to define the specific logic expression.

    The class provides an `evaluate` method that applies the logic expression to a batch of traces.
    """

    def evaluate(self, traces: Int[torch.Tensor, "batch activities"]) -> Bool[torch.Tensor, " batch"]:
        """Evaluate the logic expression on the given traces.

        Args:
        ----
            traces (Int[torch.Tensor, "batch activities"]): The traces to evaluate.

        Returns:
        -------
            Bool[torch.Tensor, " batch"]: The result of the evaluation.

        """
        return self._condition(traces)

    def __call__(self, traces: Int[torch.Tensor, "batch activities"]) -> Bool[torch.Tensor, " batch"]:
        """Evaluate the logic expression on the given traces.

        Args:
        ----
            traces (Int[torch.Tensor, "batch activities"]): The traces to evaluate.

        Returns:
        -------
            Bool[torch.Tensor, " batch"]: The result of the evaluation.

        """
        return self.evaluate(traces)

    @abstractmethod
    def _condition(self, traces: Int[torch.Tensor, "batch activities"]) -> Bool[torch.Tensor, " batch"]:
        """Compute the truth value of the expression for the given traces."""

    @abstractmethod
    def __repr__(self) -> str:
        """Return the official string representation of the expression."""

    @abstractmethod
    def __str__(self) -> str:
        """Return a human-readable string representation of the expression."""

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Determine equality with another logic expression."""

    @abstractmethod
    def __hash__(self) -> int:
        """Return a hash value for the logic expression."""
