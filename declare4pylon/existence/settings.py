from pydantic.dataclasses import dataclass

from declare4pylon.constraint import DeclareConstraintSettings


@dataclass(frozen=True, kw_only=True)
class ExistenceConstraintSettings(DeclareConstraintSettings):
    activity: int


@dataclass(frozen=True, kw_only=True)
class ExistenceCountConstraintSettings(ExistenceConstraintSettings):
    count: int = 1
