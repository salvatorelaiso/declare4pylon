import json
from abc import ABC, abstractmethod
from dataclasses import asdict

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
    def __init__(self, settings: DeclareConstraintSettings, solver: Solver):
        self._settings = settings
        self._solver = solver
        self._constraint = constraint(solver=solver, cond=self._condition)

    def __call__(self, logits, prefixes: torch.IntTensor | None = None) -> torch.Tensor:
        return self._constraint(logits, **self._settings.dict(), prefixes=prefixes)

    @staticmethod
    @abstractmethod
    def _condition(
        sampled: torch.Tensor,
        kwargs: dict,
    ) -> callable:
        raise NotImplementedError

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
