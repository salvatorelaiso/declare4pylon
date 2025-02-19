from pydantic.dataclasses import dataclass

from declare4pylon.constraint import DeclareConstraintSettings


@dataclass(frozen=True, kw_only=True)
class ChoiceConstraintSettings(DeclareConstraintSettings):
    a: int
    b: int
