from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..data import InteractionTrajectory, torque_at_center_of_mass
from ..friction import smooth_coulomb_force
from ..losses import pose_loss
from ..parameters import PhysicalParameters, normalize_quaternion, quaternion_to_matrix


def _quaternion_multiply(left: Tensor, right: Tensor) -> Tensor:
    """Multiply xyzw quaternions."""
    lx, ly, lz, lw = left.unbind(dim=-1)
    rx, ry, rz, rw = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dim=-1,
    )


@dataclass(frozen=True)
class RigidBodyInitialState:
    """Body-origin pose and velocity, with angular velocity in world coordinates."""

    position: Tensor
    quaternion_xyzw: Tensor
    linear_velocity: Tensor
    angular_velocity: Tensor


@dataclass
class TorchRigidBodyBackend:
    """Differentiable semi-implicit rollout for an unconstrained rigid body.

    This is the cleaned release implementation of the free-body model in
    ``DRBDI``/``DRBDIF``. Wrenches and states use the conventions declared by
    :class:`InteractionTrajectory`. Optional planar Coulomb friction is
    smoothed near zero velocity to retain useful gradients.
    """

    trajectory: InteractionTrajectory
    initial_state: RigidBodyInitialState
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    contact_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    normal_force: Tensor | float | None = None
    contact_point_body: tuple[float, float, float] | None = None
    friction_velocity_scale: float = 1.0e-3
    angular_damping: float = 0.0

    def __post_init__(self) -> None:
        self.trajectory.validate()
        if self.angular_damping < 0:
            raise ValueError("angular_damping must be non-negative")
        if self.friction_velocity_scale <= 0:
            raise ValueError("friction_velocity_scale must be positive")
        if self.normal_force is not None and self.contact_point_body is None:
            raise ValueError("contact_point_body is required when normal_force is supplied")

    def __call__(self, parameters: PhysicalParameters) -> tuple[Tensor, Tensor]:
        reference = parameters.mass
        position = self.initial_state.position.to(reference)
        quaternion = normalize_quaternion(self.initial_state.quaternion_xyzw.to(reference))
        linear_velocity = self.initial_state.linear_velocity.to(reference)
        angular_velocity = self.initial_state.angular_velocity.to(reference)
        com_offset = quaternion_to_matrix(quaternion) @ parameters.com
        com_world = position + com_offset
        com_velocity = linear_velocity + torch.linalg.cross(angular_velocity, com_offset)
        gravity = torch.as_tensor(self.gravity, dtype=reference.dtype, device=reference.device)
        normal = torch.as_tensor(
            self.contact_normal, dtype=reference.dtype, device=reference.device
        )
        normal = normal / torch.linalg.vector_norm(normal).clamp_min(1.0e-12)
        positions = [position]
        quaternions = [quaternion]
        mass = parameters.mass.reshape(())
        contact_body = (
            torch.as_tensor(self.contact_point_body, dtype=reference.dtype, device=reference.device)
            if self.contact_point_body is not None
            else None
        )

        for step, dt_value in enumerate(self.trajectory.dt):
            dt = dt_value.to(reference)
            measured_force = self.trajectory.force[step].to(reference)
            torque = self.trajectory.torque[step].to(reference)
            application_point = self.trajectory.application_point[step].to(reference)
            rotation = quaternion_to_matrix(quaternion)
            force = measured_force
            torque_com = torque_at_center_of_mass(
                torque, measured_force, application_point, com_world
            )
            if self.normal_force is not None:
                contact_world = position + rotation @ contact_body
                contact_lever = contact_world - com_world
                contact_velocity = com_velocity + torch.linalg.cross(
                    angular_velocity, contact_lever
                )
                tangent_velocity = contact_velocity - torch.dot(contact_velocity, normal) * normal
                supplied_load = self.normal_force
                if isinstance(supplied_load, Tensor) and supplied_load.ndim > 0:
                    supplied_load = supplied_load[step]
                normal_load = torch.as_tensor(
                    supplied_load, dtype=reference.dtype, device=reference.device
                )
                if torch.any(normal_load < 0):
                    raise ValueError("normal_force must be non-negative")
                friction_force = smooth_coulomb_force(
                    tangent_velocity,
                    normal_load,
                    parameters.friction.reshape(()),
                    velocity_scale=self.friction_velocity_scale,
                )
                contact_force = normal_load * normal + friction_force
                force = force + contact_force
                torque_com = torque_com + torch.linalg.cross(contact_lever, contact_force)

            com_velocity = com_velocity + (force / mass + gravity) * dt
            com_world = com_world + com_velocity * dt

            inertia_world = rotation @ parameters.inertia @ rotation.transpose(-1, -2)
            gyroscopic = torch.linalg.cross(
                angular_velocity, inertia_world @ angular_velocity, dim=-1
            )
            angular_acceleration = torch.linalg.solve(inertia_world, torque_com - gyroscopic)
            angular_velocity = angular_velocity + angular_acceleration * dt
            angular_velocity = angular_velocity * torch.clamp(
                1.0 - self.angular_damping * dt, min=0.0
            )
            omega_quaternion = torch.cat(
                (angular_velocity, torch.zeros(1, dtype=reference.dtype, device=reference.device))
            )
            quaternion = normalize_quaternion(
                quaternion + 0.5 * _quaternion_multiply(omega_quaternion, quaternion) * dt
            )
            position = com_world - quaternion_to_matrix(quaternion) @ parameters.com
            positions.append(position)
            quaternions.append(quaternion)

        return torch.stack(positions), torch.stack(quaternions)

    def loss(
        self,
        parameters: PhysicalParameters,
        target: InteractionTrajectory,
        *,
        position_weight: float = 1.0,
        rotation_weight: float = 1.0,
    ) -> Tensor:
        position, quaternion = self(parameters)
        return pose_loss(
            position,
            quaternion,
            target.position.to(position),
            target.quaternion_xyzw.to(quaternion),
            position_weight=position_weight,
            rotation_weight=rotation_weight,
        )
