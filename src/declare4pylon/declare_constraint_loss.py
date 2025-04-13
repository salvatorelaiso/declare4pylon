import torch
from jaxtyping import Float, Int
from pylon.constraint import constraint
from pylon.solver import Solver

from .logic_expression import LogicExpression


class DeclareConstraintLoss:
    """A class that wraps a constraint and provides a callable interface for evaluating it.

    Args:
    ----
        logic_expression (LogicExpression): The logic expression to evaluate.
        solver (Solver): The solver to use for the constraint.

    Example:
    -------
        >>> from declare4pylon import DeclareConstraintLoss
        >>> from pylon.solver import WeightedSamplingSolver
        >>> from declare4pylon import ExistenceConstraint
        >>>
        >>> constraint_loss = DeclareConstraintLoss(
        ...     ExistenceConstraint(activity=1, count=1),
        ...     WeightedSamplingSolver(num_samples=1),
        ... )
        >>>
        >>> logits = torch.rand(2, 3, 4)
        >>> loss = constraint_loss(logits)

    """

    def __init__(self, logic_expression: LogicExpression, solver: Solver) -> None:
        """Initialize the DeclareConstraintLoss.

        Args:
        ----
            logic_expression (LogicExpression): The logic expression to evaluate.
            solver (Solver): The solver to use for the constraint.

        Raises:
        ------
            TypeError: If logic_expression is not of type :class:`~declare4pylon.LogicExpression`
                or if solver is not of type `pylon.solver.Solver`.

        """
        if not isinstance(logic_expression, LogicExpression):
            msg = f"Expected logic_expression to be of type LogicExpression, but got {type(logic_expression)}"
            raise TypeError(msg)
        if not isinstance(solver, Solver):
            msg = f"Expected solver to be of type Solver, but got {type(solver)}"
            raise TypeError(msg)
        self.logic_expression = logic_expression

        self._constraint = constraint(
            cond=self._wrapper,
            solver=solver,
        )

    def __call__(
        self,
        logits: Float[torch.Tensor, "batch classes logits"],
        *,
        prefixes: Int[torch.Tensor, "batch prefix"] | None = None,
    ) -> Float[torch.Tensor, ""]:
        """Evaluate the constraint.

        Args:
        ----
            logits (Float[torch.Tensor, "batch classes logits"]): The logits to evaluate.
            prefixes (Int[torch.Tensor, "batch prefix"], optional): Optional prefixes to prepend to the traces.
                Defaults to None.

        Returns:
        -------
            Float[torch.Tensor, ""]: The loss value.

        """
        return self._constraint(logits, prefixes=prefixes)  # type: ignore[no-any-return]

    def _wrapper(self, traces: torch.Tensor, kwargs: dict) -> Float[torch.Tensor, ""]:
        if kwargs.get("prefixes") is not None:
            prefixes = kwargs["prefixes"]
            traces = torch.cat([prefixes, traces], dim=1)
        return self.logic_expression.evaluate(traces)
