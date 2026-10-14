from __future__ import annotations

import torch
from torch import Tensor

from .parameters import normalize_quaternion


def quaternion_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Sign-invariant squared chordal loss for unit xyzw quaternions."""
    prediction = normalize_quaternion(prediction)
    target = normalize_quaternion(target)
    direct = torch.sum((prediction - target) ** 2, dim=-1)
    antipodal = torch.sum((prediction + target) ** 2, dim=-1)
    return torch.minimum(direct, antipodal)


def pose_loss(
    position_prediction: Tensor,
    quaternion_prediction: Tensor,
    position_target: Tensor,
    quaternion_target: Tensor,
    *,
    position_weight: float = 1.0,
    rotation_weight: float = 1.0,
) -> Tensor:
    """Mean trajectory pose loss corresponding to the paper's Eq. (4)."""
    if position_prediction.shape != position_target.shape:
        raise ValueError("predicted and target positions must have equal shapes")
    if quaternion_prediction.shape != quaternion_target.shape:
        raise ValueError("predicted and target quaternions must have equal shapes")

    position_error = torch.sum((position_prediction - position_target) ** 2, dim=-1)
    rotation_error = quaternion_loss(quaternion_prediction, quaternion_target)
    return (position_weight * position_error + rotation_weight * rotation_error).mean()


def quadratic_bound_penalty(value: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    """Squared hinge penalty from the paper's Eq. (5)."""
    if torch.any(lower > upper):
        raise ValueError("lower bounds must not exceed upper bounds")
    lower_violation = torch.relu(lower - value)
    upper_violation = torch.relu(value - upper)
    return torch.sum(lower_violation.square() + upper_violation.square())


def total_loss(pose: Tensor, bound_penalty: Tensor, penalty_weight: float = 0.1) -> Tensor:
    """Augmented identification objective from the paper's Eq. (6)."""
    if penalty_weight < 0:
        raise ValueError("penalty_weight must be non-negative")
    return pose + penalty_weight * bound_penalty
