from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


def inverse_softplus(value: Tensor) -> Tensor:
    """Numerically stable inverse of softplus for strictly positive values."""
    eps = torch.finfo(value.dtype).eps
    value = value.clamp_min(eps)
    return value + torch.log(-torch.expm1(-value))


@dataclass(frozen=True)
class PhysicalParameters:
    """Physical parameters consumed by the simulator.

    Inertia is expressed about the center of mass in the body frame.
    """

    mass: Tensor
    com: Tensor
    inertia: Tensor
    friction: Tensor

    def as_vector(self) -> Tensor:
        inertia_unique = torch.stack(
            (
                self.inertia[..., 0, 0],
                self.inertia[..., 1, 1],
                self.inertia[..., 2, 2],
                self.inertia[..., 0, 1],
                self.inertia[..., 0, 2],
                self.inertia[..., 1, 2],
            ),
            dim=-1,
        )
        return torch.cat(
            (
                self.mass.reshape(-1),
                self.com.reshape(-1),
                inertia_unique.reshape(-1),
                self.friction.reshape(-1),
            )
        )


@dataclass
class RawPhysicalParameters:
    """Unconstrained optimization variables with a physical decoding.

    The principal moments are represented as

        Ixx = b + c, Iyy = a + c, Izz = a + b

    for positive ``a, b, c``. This guarantees positive moments and the rigid
    body triangle inequalities. A unit quaternion rotates the principal-axis
    inertia into the object frame.
    """

    mass_raw: Tensor
    com: Tensor
    principal_raw: Tensor
    inertia_rotation_raw: Tensor
    friction_raw: Tensor
    epsilon: float = 1.0e-8

    @classmethod
    def from_physical(
        cls,
        mass: Tensor,
        com: Tensor,
        principal_moments: Tensor,
        friction: Tensor,
        inertia_rotation: Tensor | None = None,
    ) -> RawPhysicalParameters:
        if mass.numel() != 1 or not torch.isfinite(mass).all() or torch.any(mass <= 0):
            raise ValueError("mass must be one positive finite scalar")
        if com.shape != (3,) or not torch.isfinite(com).all():
            raise ValueError("com must be a finite 3-vector")
        if principal_moments.shape != (3,) or not torch.isfinite(principal_moments).all():
            raise ValueError("principal_moments must be a finite 3-vector")
        if friction.numel() != 1 or not torch.isfinite(friction).all() or torch.any(friction < 0):
            raise ValueError("friction must be one non-negative finite scalar")

        ixx, iyy, izz = principal_moments.unbind(dim=-1)
        abc = torch.stack(
            (
                0.5 * (iyy + izz - ixx),
                0.5 * (ixx + izz - iyy),
                0.5 * (ixx + iyy - izz),
            ),
            dim=-1,
        )
        if torch.any(abc <= 0):
            raise ValueError("principal moments violate rigid-body triangle inequalities")

        if inertia_rotation is None:
            inertia_rotation = torch.tensor(
                [0.0, 0.0, 0.0, 1.0], dtype=mass.dtype, device=mass.device
            )
        elif (
            inertia_rotation.shape != (4,)
            or not torch.isfinite(inertia_rotation).all()
            or torch.linalg.vector_norm(inertia_rotation) < 1e-12
        ):
            raise ValueError("inertia_rotation must be a finite nonzero xyzw quaternion")

        return cls(
            mass_raw=inverse_softplus(mass),
            com=com.clone(),
            principal_raw=inverse_softplus(abc),
            inertia_rotation_raw=inertia_rotation.clone(),
            friction_raw=inverse_softplus(friction),
        )

    def tensors(self) -> list[Tensor]:
        return [
            self.mass_raw,
            self.com,
            self.principal_raw,
            self.inertia_rotation_raw,
            self.friction_raw,
        ]

    def decode(self) -> PhysicalParameters:
        mass = F.softplus(self.mass_raw) + self.epsilon
        friction = F.softplus(self.friction_raw)

        abc = F.softplus(self.principal_raw) + self.epsilon
        a, b, c = abc.unbind(dim=-1)
        principal = torch.stack((b + c, a + c, a + b), dim=-1)
        diagonal = torch.diag_embed(principal)

        rotation = quaternion_to_matrix(self.inertia_rotation_raw)
        inertia = rotation @ diagonal @ rotation.transpose(-1, -2)
        return PhysicalParameters(mass=mass, com=self.com, inertia=inertia, friction=friction)


def normalize_quaternion(quaternion: Tensor) -> Tensor:
    """Normalize an xyzw quaternion, falling back safely near zero."""
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion must have four components in xyzw order")
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    identity = torch.zeros_like(quaternion)
    identity[..., 3] = 1.0
    return torch.where(norm > 1.0e-12, quaternion / norm.clamp_min(1.0e-12), identity)


def quaternion_to_matrix(quaternion: Tensor) -> Tensor:
    """Convert normalized xyzw quaternions to rotation matrices."""
    x, y, z, w = normalize_quaternion(quaternion).unbind(dim=-1)
    two = 2.0
    return torch.stack(
        (
            1 - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1 - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1 - two * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))
