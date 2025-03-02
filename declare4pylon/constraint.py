import json
from abc import ABC
from dataclasses import asdict
from typing import Self

import torch
from pydantic.dataclasses import dataclass
from pylon.constraint import constraint
from pylon.solver import Solver


def safe_serialize(obj):
    try:
        return json.dumps(obj)
    except TypeError:
        return str(obj)


@dataclass(frozen=True, kw_only=True)
class DeclareConstraintSettings(ABC):
    dict = asdict


class DeclareConstraint(ABC):
    _condition: callable = None

    def __init__(self, settings: DeclareConstraintSettings, solver: Solver):
        self._settings = settings
        self._solver = solver
        self._constraint = constraint(solver=solver, cond=self._compute)

    def __call__(self, logits, prefixes: torch.IntTensor | None = None) -> torch.Tensor:
        if self._condition is None:
            raise NotImplementedError("'_condition' must be defined in a subclass")
        return self._constraint(logits, **self._settings.dict(), prefixes=prefixes)

    @classmethod
    def _compute(
        cls: type[Self],
        sampled: torch.Tensor,
        kwargs: dict,
    ) -> callable:
        return cls._condition(sampled, **kwargs)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(solver={self._solver.__class__.__name__}({', '.join(f'{k}={repr(v)}' for k, v in self._solver.__dict__.items() if k != "cond")}), settings={self._settings})"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def settings(self):
        return self._settings

    @property
    def condition(self):
        return self._condition
