"""Shared test utilities and fixtures."""

# ruff: noqa: D105
from declare4pylon import LogicExpression


class MockLogicExpression(LogicExpression):
    """Reusable mock LogicExpression for testing which provides basic implementations for the abstract methods."""

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return self.__repr__()
