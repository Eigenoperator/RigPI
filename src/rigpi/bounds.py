from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .losses import quadratic_bound_penalty
from .parameters import PhysicalParameters


@dataclass(frozen=True)
class ParameterBounds:
    mass_lower: Tensor
    mass_upper: Tensor
    com_lower: Tensor
    com_upper: Tensor
    inertia_lower: Tensor
    inertia_upper: Tensor
    friction_lower: Tensor
    friction_upper: Tensor

    def penalty(self, parameters: PhysicalParameters) -> Tensor:
        lower, upper = self.vector_bounds()
        return quadratic_bound_penalty(parameters.as_vector(), lower, upper)

    def vector_bounds(self) -> tuple[Tensor, Tensor]:
        lower_inertia = torch.stack(
            (
                self.inertia_lower[..., 0, 0],
                self.inertia_lower[..., 1, 1],
                self.inertia_lower[..., 2, 2],
                self.inertia_lower[..., 0, 1],
                self.inertia_lower[..., 0, 2],
                self.inertia_lower[..., 1, 2],
            ),
            dim=-1,
        )
        upper_inertia = torch.stack(
            (
                self.inertia_upper[..., 0, 0],
                self.inertia_upper[..., 1, 1],
                self.inertia_upper[..., 2, 2],
                self.inertia_upper[..., 0, 1],
                self.inertia_upper[..., 0, 2],
                self.inertia_upper[..., 1, 2],
            ),
            dim=-1,
        )
        lower = torch.cat(
            (
                self.mass_lower.reshape(-1),
                self.com_lower.reshape(-1),
                lower_inertia.reshape(-1),
                self.friction_lower.reshape(-1),
            )
        )
        upper = torch.cat(
            (
                self.mass_upper.reshape(-1),
                self.com_upper.reshape(-1),
                upper_inertia.reshape(-1),
                self.friction_upper.reshape(-1),
            )
        )
        return lower, upper
