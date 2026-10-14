from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class ScaleAwareLearningRate:
    """Choose a per-parameter step that is small relative to its prior scale.

    For each component, the desired maximum update is ``relative_step`` times
    ``max(abs(initial), upper-lower, minimum_scale)``. Dividing that update by
    the current gradient magnitude yields the learning rate. This directly
    implements the paper's "roughly two orders of magnitude smaller" rule when
    ``relative_step=1e-2``.
    """

    initial: Tensor
    lower: Tensor
    upper: Tensor
    relative_step: float = 1.0e-2
    minimum_scale: float = 1.0e-8
    minimum_lr: float = 1.0e-10
    maximum_lr: float = 1.0e2
    gradient_epsilon: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.relative_step <= 0:
            raise ValueError("relative_step must be positive")
        if torch.any(self.lower > self.upper):
            raise ValueError("lower bounds must not exceed upper bounds")

    @property
    def scale(self) -> Tensor:
        minimum = torch.full_like(self.initial, self.minimum_scale)
        return torch.maximum(torch.maximum(self.initial.abs(), self.upper - self.lower), minimum)

    def learning_rate(self, gradient: Tensor) -> Tensor:
        desired_update = self.relative_step * self.scale
        rate = desired_update / gradient.abs().clamp_min(self.gradient_epsilon)
        return rate.clamp(min=self.minimum_lr, max=self.maximum_lr)

    def step(self, value: Tensor, gradient: Tensor) -> Tensor:
        return value - self.learning_rate(gradient) * gradient
