from pydantic.dataclasses import dataclass

from declare4pylon.constraint import DeclareConstraintSettings


@dataclass(frozen=True, kw_only=True)
class RelationConstraintSettings(DeclareConstraintSettings):
    activity_a: int
    activity_b: int
