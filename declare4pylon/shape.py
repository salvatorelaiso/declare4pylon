import torch


def check(sampled: torch.Tensor) -> None:
    assert sampled.dim() == 2
    assert sampled.size(1) > 0


def match(sampled: torch.Tensor, prefixes: torch.Tensor) -> None:
    assert prefixes is None or prefixes.dim() == 2
    assert sampled.size(0) == prefixes.size(0) if prefixes is not None else True
