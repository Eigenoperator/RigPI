"""Newton semi-implicit, differentiable single free-body rollout."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
import warp as wp
from torch import Tensor

from ..data import InteractionTrajectory
from ..losses import pose_loss
from ..parameters import PhysicalParameters, quaternion_to_matrix
from ._version import require_supported_newton
from .torch_rigid import RigidBodyInitialState


@wp.kernel
def _apply_free_body_wrench(
    body_f: wp.array(dtype=wp.spatial_vector),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    measured_force: wp.array(dtype=wp.vec3),
    measured_torque: wp.array(dtype=wp.vec3),
    application_point: wp.array(dtype=wp.vec3),
    friction: wp.array(dtype=float),
    normal_load: wp.array(dtype=float),
    contact_point_body: wp.vec3,
    contact_normal: wp.vec3,
    velocity_scale: float,
):
    pose = body_q[0]
    com_world = wp.transform_point(pose, body_com[0])
    force = measured_force[0]
    torque = measured_torque[0] + wp.cross(application_point[0] - com_world, force)
    if normal_load[0] > 0.0:
        contact_world = wp.transform_point(pose, contact_point_body)
        lever = contact_world - com_world
        twist = body_qd[0]
        velocity = wp.spatial_top(twist) + wp.cross(wp.spatial_bottom(twist), lever)
        tangent = velocity - wp.dot(velocity, contact_normal) * contact_normal
        speed = wp.length(tangent)
        friction_magnitude = -friction[0] * normal_load[0] * wp.tanh(speed / velocity_scale)
        friction_force = friction_magnitude * tangent / wp.max(speed, 1.0e-12)
        contact_force = normal_load[0] * contact_normal + friction_force
        force = force + contact_force
        torque = torque + wp.cross(lever, contact_force)
    body_f[0] = wp.spatial_vector(force[0], force[1], force[2], torque[0], torque[1], torque[2])


class _SemiImplicitRollout(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, mass, com, inertia, inverse_mass, inverse_inertia, friction, initial_twist, backend
    ):
        model = backend.model
        arrays = [
            wp.from_torch(mass.detach().contiguous(), requires_grad=True),
            wp.from_torch(com.detach().contiguous(), dtype=wp.vec3, requires_grad=True),
            wp.from_torch(inertia.detach().contiguous(), dtype=wp.mat33, requires_grad=True),
            wp.from_torch(inverse_mass.detach().contiguous(), requires_grad=True),
            wp.from_torch(
                inverse_inertia.detach().contiguous(), dtype=wp.mat33, requires_grad=True
            ),
            wp.from_torch(friction.detach().contiguous(), requires_grad=True),
            wp.from_torch(
                initial_twist.detach().contiguous(), dtype=wp.spatial_vector, requires_grad=True
            ),
        ]
        (
            model.body_mass,
            model.body_com,
            model.body_inertia,
            model.body_inv_mass,
            model.body_inv_inertia,
            friction_wp,
            twist_wp,
        ) = arrays
        states = [model.state() for _ in range(backend.steps + 1)]
        states[0].body_qd = twist_wp
        tape = wp.Tape()
        with tape:
            for step in range(backend.steps):
                states[step].clear_forces()
                wp.launch(
                    _apply_free_body_wrench,
                    dim=1,
                    inputs=[
                        states[step].body_f,
                        states[step].body_q,
                        states[step].body_qd,
                        model.body_com,
                        backend.forces[step],
                        backend.torques[step],
                        backend.application_points[step],
                        friction_wp,
                        backend.normal_loads[step],
                        backend.contact_point_body,
                        backend.contact_normal,
                        backend.friction_velocity_scale,
                    ],
                    device=model.device,
                )
                backend.solver.step(states[step], states[step + 1], None, None, backend.dt[step])
        trajectory = torch.stack([wp.to_torch(state.body_q)[0] for state in states])
        ctx.tape = tape
        ctx.states = states
        ctx.arrays = arrays
        ctx.model = model
        return trajectory

    @staticmethod
    def backward(ctx, trajectory_adjoint):
        for index, state in enumerate(ctx.states):
            state.body_q.grad = wp.from_torch(
                trajectory_adjoint[index : index + 1].contiguous(), dtype=wp.transform
            )
        ctx.tape.backward()

        def gradient(array):
            value = ctx.tape.gradients.get(array)
            return wp.to_torch(value) if value is not None else torch.zeros_like(wp.to_torch(array))

        return (*[gradient(array) for array in ctx.arrays], None)


@dataclass
class NewtonSemiImplicitBackend:
    """Paper-aligned free-body Newton backend; poses are body-origin poses."""

    model: object
    solver: object
    trajectory: InteractionTrajectory
    initial_state: RigidBodyInitialState
    forces: list[wp.array]
    torques: list[wp.array]
    application_points: list[wp.array]
    normal_loads: list[wp.array]
    dt: list[float]
    contact_point_body: wp.vec3
    contact_normal: wp.vec3
    friction_velocity_scale: float

    @property
    def steps(self) -> int:
        return len(self.forces)

    @classmethod
    def from_trajectory(
        cls,
        trajectory: InteractionTrajectory,
        initial_state: RigidBodyInitialState,
        *,
        gravity: float = -9.81,
        normal_force: Tensor | float | None = None,
        contact_point_body: tuple[float, float, float] | None = None,
        contact_normal: tuple[float, float, float] = (0, 0, 1),
        friction_velocity_scale: float = 1.0e-3,
        device: str | None = None,
    ) -> NewtonSemiImplicitBackend:
        import newton

        require_supported_newton(newton)

        trajectory.validate()
        for name, value, shape in (
            ("position", initial_state.position, (3,)),
            ("quaternion_xyzw", initial_state.quaternion_xyzw, (4,)),
            ("linear_velocity", initial_state.linear_velocity, (3,)),
            ("angular_velocity", initial_state.angular_velocity, (3,)),
        ):
            if value.shape != shape or not torch.isfinite(value).all():
                raise ValueError(f"initial_state.{name} must be a finite {shape}-vector")
        quaternion_norm = torch.linalg.vector_norm(initial_state.quaternion_xyzw)
        if abs(float(quaternion_norm) - 1.0) > 1e-3:
            raise ValueError("initial_state quaternion must be unit length")
        if normal_force is not None and contact_point_body is None:
            raise ValueError("contact_point_body is required with normal_force")
        if contact_point_body is not None:
            contact_point = torch.as_tensor(contact_point_body, dtype=torch.float32)
            if contact_point.shape != (3,) or not torch.isfinite(contact_point).all():
                raise ValueError("contact_point_body must be a finite 3-vector")
        if not isfinite(gravity):
            raise ValueError("gravity must be finite")
        if not isfinite(friction_velocity_scale) or friction_velocity_scale <= 0:
            raise ValueError("friction_velocity_scale must be finite and positive")
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=wp.vec3(0.0, 0.0, gravity))
        position = initial_state.position.tolist()
        quaternion = initial_state.quaternion_xyzw.tolist()
        inertia = wp.mat33(1, 0, 0, 0, 1, 0, 0, 0, 1)
        builder.add_body(
            xform=wp.transform(wp.vec3(*position), wp.quat(*quaternion)),
            mass=1.0,
            com=wp.vec3(0, 0, 0),
            inertia=inertia,
        )
        model = builder.finalize(device=device, requires_grad=True)
        solver = newton.solvers.SolverSemiImplicit(model, angular_damping=0.0)
        torch_device = wp.device_to_torch(model.device)

        def vec3(value):
            return wp.from_torch(
                value.reshape(1, 3).to(device=torch_device, dtype=torch.float32), dtype=wp.vec3
            )

        if normal_force is None:
            loads = torch.zeros(trajectory.time.shape[0] - 1)
        else:
            loads = torch.as_tensor(normal_force, dtype=torch.float32)
            if loads.ndim == 0:
                loads = loads.expand(trajectory.time.shape[0] - 1)
            if (
                loads.shape != (trajectory.time.shape[0] - 1,)
                or not torch.isfinite(loads).all()
                or torch.any(loads < 0)
            ):
                raise ValueError("normal_force must be finite, non-negative, and scalar or [N-1]")
        normal = torch.as_tensor(contact_normal, dtype=torch.float32)
        if normal.shape != (3,) or not torch.isfinite(normal).all():
            raise ValueError("contact_normal must be a finite 3-vector")
        if float(torch.linalg.vector_norm(normal)) <= 1e-12:
            raise ValueError("contact_normal must be nonzero")
        normal = normal / torch.linalg.vector_norm(normal)
        return cls(
            model=model,
            solver=solver,
            trajectory=trajectory,
            initial_state=initial_state,
            forces=[vec3(value) for value in trajectory.force],
            torques=[vec3(value) for value in trajectory.torque],
            application_points=[vec3(value) for value in trajectory.application_point],
            normal_loads=[
                wp.from_torch(value.reshape(1).to(device=torch_device)) for value in loads
            ],
            dt=[float(value) for value in trajectory.dt],
            contact_point_body=wp.vec3(*(contact_point_body or (0, 0, 0))),
            contact_normal=wp.vec3(*normal.tolist()),
            friction_velocity_scale=friction_velocity_scale,
        )

    def __call__(self, parameters: PhysicalParameters) -> tuple[Tensor, Tensor]:
        mass = parameters.mass.reshape(1)
        com = parameters.com.reshape(1, 3)
        inertia = parameters.inertia.reshape(1, 3, 3)
        inverse_mass = mass.reciprocal()
        inverse_inertia = torch.linalg.inv(inertia)
        rotation = quaternion_to_matrix(self.initial_state.quaternion_xyzw.to(mass))
        com_offset = rotation @ parameters.com
        angular = self.initial_state.angular_velocity.to(mass)
        origin_velocity = self.initial_state.linear_velocity.to(mass)
        com_velocity = origin_velocity + torch.linalg.cross(angular, com_offset)
        twist = torch.cat((com_velocity, angular)).reshape(1, 6)
        device = wp.device_to_torch(self.model.device)
        trajectory = _SemiImplicitRollout.apply(
            mass.to(device),
            com.to(device),
            inertia.to(device),
            inverse_mass.to(device),
            inverse_inertia.to(device),
            parameters.friction.reshape(1).to(device),
            twist.to(device),
            self,
        )
        return trajectory[:, :3], trajectory[:, 3:7]

    def loss(self, parameters: PhysicalParameters, target: InteractionTrajectory) -> Tensor:
        position, quaternion = self(parameters)
        return pose_loss(
            position,
            quaternion,
            target.position.to(position),
            target.quaternion_xyzw.to(quaternion),
        )
