from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .parameters import normalize_quaternion


@dataclass(frozen=True)
class InteractionTrajectory:
    """Processed interaction data in one explicitly declared world frame.

    ``force`` and ``torque`` are wrenches at ``application_point``. Position is
    the body-frame origin, not implicitly the center of mass. All SI units are
    required: seconds, meters, newtons, and newton-meters.
    """

    time: Tensor
    position: Tensor
    quaternion_xyzw: Tensor
    force: Tensor
    torque: Tensor
    application_point: Tensor
    linear_velocity: Tensor | None = None
    angular_velocity: Tensor | None = None

    def validate(self) -> None:
        if self.time.ndim != 1:
            raise ValueError("time must have shape [N]")
        steps = self.time.shape[0]
        if steps < 2:
            raise ValueError("trajectory must contain at least two states")
        if self.position.shape != (steps, 3):
            raise ValueError("position must have shape [N, 3]")
        if self.quaternion_xyzw.shape != (steps, 4):
            raise ValueError("quaternion_xyzw must have shape [N, 4]")
        for name, value in (
            ("force", self.force),
            ("torque", self.torque),
            ("application_point", self.application_point),
        ):
            if value.shape != (steps - 1, 3):
                raise ValueError(f"{name} must have shape [N-1, 3]")
        for name, value in (
            ("linear_velocity", self.linear_velocity),
            ("angular_velocity", self.angular_velocity),
        ):
            if value is not None and value.shape != (steps, 3):
                raise ValueError(f"{name} must have shape [N, 3]")
        if not torch.all(self.time[1:] > self.time[:-1]):
            raise ValueError("timestamps must be strictly increasing")
        quaternion_norm = torch.linalg.vector_norm(self.quaternion_xyzw, dim=-1)
        if not torch.allclose(quaternion_norm, torch.ones_like(quaternion_norm), atol=1.0e-4):
            raise ValueError("input quaternions must be normalized")
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor) and not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")

    @property
    def dt(self) -> Tensor:
        return self.time[1:] - self.time[:-1]

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> InteractionTrajectory:
        with np.load(path) as data:
            kwargs = {
                key: torch.as_tensor(data[key], dtype=dtype, device=device)
                for key in (
                    "time",
                    "position",
                    "quaternion_xyzw",
                    "force",
                    "torque",
                    "application_point",
                )
            }
            for optional in ("linear_velocity", "angular_velocity"):
                kwargs[optional] = (
                    torch.as_tensor(data[optional], dtype=dtype, device=device)
                    if optional in data
                    else None
                )
        quaternion_norm = torch.linalg.vector_norm(kwargs["quaternion_xyzw"], dim=-1)
        if not torch.isfinite(quaternion_norm).all() or torch.any(
            (quaternion_norm - 1.0).abs() > 1.0e-2
        ):
            raise ValueError("input quaternions must be near unit length")
        kwargs["quaternion_xyzw"] = normalize_quaternion(kwargs["quaternion_xyzw"])
        trajectory = cls(**kwargs)
        trajectory.validate()
        return trajectory


def torque_at_center_of_mass(
    torque_at_application_point: Tensor,
    force: Tensor,
    application_point_world: Tensor,
    center_of_mass_world: Tensor,
) -> Tensor:
    """Move a measured torque to the center of mass.

    With ``r = p_application - p_com``, the equivalent COM torque is
    ``tau_com = tau_application + r × force``.
    """
    lever = application_point_world - center_of_mass_world
    return torque_at_application_point + torch.linalg.cross(lever, force, dim=-1)
