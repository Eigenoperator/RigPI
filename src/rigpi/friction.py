from __future__ import annotations

from math import isfinite

import torch
from torch import Tensor


def coulomb_joint_load(
    normal_force: float, *, joint_type: str, effective_radius: float | None = None
) -> float:
    """Return the load multiplying a dimensionless Coulomb coefficient.

    A prismatic joint needs a force in N; a revolute joint needs the moment
    arm as well, yielding N m. The load must be supplied by the experimenter.
    """
    if not isfinite(normal_force) or normal_force <= 0:
        raise ValueError("normal_force must be finite and positive")
    if joint_type == "prismatic":
        if effective_radius is not None:
            raise ValueError("effective_radius only applies to revolute joints")
        return normal_force
    if joint_type == "revolute":
        if effective_radius is None or not isfinite(effective_radius) or effective_radius <= 0:
            raise ValueError("revolute joints require a finite positive effective_radius")
        return normal_force * effective_radius
    raise ValueError("joint_type must be 'revolute' or 'prismatic'")


def smooth_coulomb_force(
    tangential_velocity: Tensor,
    normal_force: Tensor,
    friction: Tensor,
    *,
    velocity_scale: float = 1.0e-3,
) -> Tensor:
    """Differentiable isotropic Coulomb friction in the tangent plane.

    The tanh transition approximates static/kinetic switching without a
    discontinuity, so gradients with respect to the friction coefficient and
    velocity are available near zero speed.
    """
    if not isfinite(velocity_scale) or velocity_scale <= 0:
        raise ValueError("velocity_scale must be finite and positive")
    speed = torch.linalg.vector_norm(tangential_velocity, dim=-1, keepdim=True)
    direction = tangential_velocity / speed.clamp_min(torch.finfo(speed.dtype).eps)
    magnitude = friction * normal_force * torch.tanh(speed / velocity_scale)
    return -magnitude * direction


def smooth_coulomb_joint_torque(
    joint_velocity: Tensor,
    friction: Tensor,
    *,
    load: Tensor | float = 1.0,
    velocity_scale: float = 1.0e-3,
) -> Tensor:
    """Differentiable Coulomb generalized force for revolute/prismatic joints."""
    if not isfinite(velocity_scale) or velocity_scale <= 0:
        raise ValueError("velocity_scale must be finite and positive")
    load_tensor = torch.as_tensor(load, dtype=joint_velocity.dtype, device=joint_velocity.device)
    return -friction * load_tensor * torch.tanh(joint_velocity / velocity_scale)
