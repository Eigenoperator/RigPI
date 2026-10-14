from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import torch
import warp as wp
from torch import Tensor

from ..data import InteractionTrajectory
from ..friction import coulomb_joint_load
from ..losses import pose_loss
from ..parameters import PhysicalParameters
from ._version import require_supported_newton


@wp.kernel
def _apply_body_wrench(
    body_f: wp.array(dtype=wp.spatial_vector),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    force: wp.array(dtype=wp.vec3),
    torque: wp.array(dtype=wp.vec3),
    application_point: wp.array(dtype=wp.vec3),
    body_index: int,
):
    """Transfer a measured world-frame wrench to COM and apply it."""
    f = force[0]
    com_world = wp.transform_point(body_q[body_index], body_com[body_index])
    lever = application_point[0] - com_world
    t = torque[0] + wp.cross(lever, f)
    body_f[body_index] = wp.spatial_vector(f[0], f[1], f[2], t[0], t[1], t[2])


@wp.kernel
def _apply_smooth_joint_friction(
    joint_qd: wp.array(dtype=float),
    joint_f: wp.array(dtype=float),
    friction: wp.array(dtype=float),
    joint_dof_index: int,
    load: float,
    velocity_scale: float,
):
    """Apply differentiable Coulomb friction as a generalized joint force."""
    velocity = joint_qd[joint_dof_index]
    joint_f[joint_dof_index] = joint_f[joint_dof_index] - friction[0] * load * wp.tanh(
        velocity / velocity_scale
    )


class _FeatherstoneRollout(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        body_mass: Tensor,
        body_com: Tensor,
        body_inertia: Tensor,
        friction: Tensor,
        backend: NewtonFeatherstoneBackend,
    ) -> Tensor:
        import newton

        require_supported_newton(newton)

        model = backend.model
        # This custom autograd function returns Warp gradients explicitly, so
        # detached leaf views avoid PyTorch's non-leaf .grad warning.
        model.body_mass = wp.from_torch(body_mass.detach().contiguous(), requires_grad=True)
        model.body_com = wp.from_torch(
            body_com.detach().contiguous(), dtype=wp.vec3, requires_grad=True
        )
        model.body_inertia = wp.from_torch(
            body_inertia.detach().contiguous(), dtype=wp.mat33, requires_grad=True
        )
        friction_wp = wp.from_torch(friction.detach().contiguous(), requires_grad=True)

        states = [model.state() for _ in range(backend.steps + 1)]
        states[0].joint_q.assign(backend.initial_joint_q)
        states[0].joint_qd.assign(backend.initial_joint_qd)
        newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])

        controls = [model.control() for _ in range(backend.steps)]
        tape = wp.Tape()
        with tape:
            # Featherstone caches spatial-inertia work buffers at construction.
            # Build it after installing the current differentiable parameters
            # and inside the tape so mass/inertia remain on the gradient path.
            backend.solver = backend.make_solver()
            for step in range(backend.steps):
                states[step].clear_forces()
                controls[step].joint_f.zero_()
                wp.launch(
                    _apply_body_wrench,
                    dim=1,
                    inputs=[
                        states[step].body_f,
                        states[step].body_q,
                        model.body_com,
                        backend.forces[step],
                        backend.torques[step],
                        backend.application_points[step],
                        backend.body_index,
                    ],
                    device=model.device,
                )
                wp.launch(
                    _apply_smooth_joint_friction,
                    dim=1,
                    inputs=[
                        states[step].joint_qd,
                        controls[step].joint_f,
                        friction_wp,
                        backend.joint_dof_index,
                        backend.friction_load,
                        backend.friction_velocity_scale,
                    ],
                    device=model.device,
                )
                backend.step_solver(
                    states[step], states[step + 1], controls[step], backend.dt[step]
                )

        # Newton transforms are exposed to Torch as xyzw transform rows:
        # translation [0:3], quaternion [3:7].
        trajectory = torch.stack(
            [wp.to_torch(state.body_q)[backend.body_index] for state in states],
            dim=0,
        )
        ctx.backend = backend
        ctx.states = states
        ctx.tape = tape
        ctx.model = model
        ctx.friction_wp = friction_wp
        return trajectory

    @staticmethod
    def backward(ctx, trajectory_adjoint: Tensor):
        for index, state in enumerate(ctx.states):
            full_adjoint = torch.zeros(
                (ctx.model.body_count, 7),
                dtype=trajectory_adjoint.dtype,
                device=trajectory_adjoint.device,
            )
            full_adjoint[ctx.backend.body_index] = trajectory_adjoint[index]
            state.body_q.grad = wp.from_torch(full_adjoint, dtype=wp.transform)

        ctx.tape.backward()

        def gradient(array, like: Tensor) -> Tensor:
            value = ctx.tape.gradients.get(array)
            if value is None:
                return torch.zeros_like(like)
            return wp.to_torch(value)

        body_mass = ctx.model.body_mass
        body_com = ctx.model.body_com
        body_inertia = ctx.model.body_inertia
        return (
            gradient(body_mass, wp.to_torch(body_mass)),
            gradient(body_com, wp.to_torch(body_com)),
            gradient(body_inertia, wp.to_torch(body_inertia)),
            gradient(ctx.friction_wp, wp.to_torch(ctx.friction_wp)),
            None,
        )


@dataclass
class NewtonFeatherstoneBackend:
    """Newton migration of the original ``djdi`` Featherstone rollout.

    Only the selected moving link is identified. Other links retain their URDF
    inertial properties. Input wrenches are expressed in the world frame at
    their measured application points and are transferred to the current COM
    inside the differentiable rollout.
    """

    model: object
    body_index: int
    joint_dof_index: int
    forces: list[wp.array]
    torques: list[wp.array]
    application_points: list[wp.array]
    dt: list[float]
    initial_joint_q: wp.array
    initial_joint_qd: wp.array
    base_mass: Tensor
    base_com: Tensor
    base_inertia: Tensor
    friction_load: float
    friction_velocity_scale: float = 1.0e-3
    solver_class: object | None = None
    solver: object | None = None

    @property
    def steps(self) -> int:
        return len(self.forces)

    def step_solver(self, state_in, state_out, control, dt: float) -> None:
        """Advance the pinned Newton 1.6.1 Featherstone solver."""
        self.solver.step(state_in, state_out, control, None, dt)

    def make_solver(self):
        if self.solver_class is None:
            raise RuntimeError("Featherstone solver class was not configured")
        return self.solver_class(
            self.model,
            update_mass_matrix_interval=1,
            friction_smoothing=self.friction_velocity_scale,
        )

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str | Path,
        trajectory: InteractionTrajectory,
        *,
        body_index: int,
        joint_dof_index: int,
        initial_joint_q: Tensor,
        initial_joint_qd: Tensor,
        joint_type: str,
        friction_normal_force: float,
        friction_effective_radius: float | None = None,
        friction_velocity_scale: float = 1.0e-3,
        device: str | None = None,
    ) -> NewtonFeatherstoneBackend:
        import newton

        require_supported_newton(newton)
        trajectory.validate()
        friction_load = coulomb_joint_load(
            friction_normal_force,
            joint_type=joint_type,
            effective_radius=friction_effective_radius,
        )
        if not isfinite(friction_velocity_scale) or friction_velocity_scale <= 0:
            raise ValueError("friction_velocity_scale must be finite and positive")
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        resolved_urdf = str(Path(urdf_path).resolve())
        builder.add_urdf(
            resolved_urdf,
            floating=False,
            enable_self_collisions=False,
            ignore_inertial_definitions=False,
        )
        model = builder.finalize(device=device, requires_grad=True)
        if not 0 <= body_index < model.body_count:
            raise IndexError("body_index is outside the imported model")
        if not 0 <= joint_dof_index < model.joint_dof_count:
            raise IndexError("joint_dof_index is outside the imported model")
        if initial_joint_q.numel() != model.joint_coord_count:
            raise ValueError("initial_joint_q must include every model joint coordinate")
        if initial_joint_qd.numel() != model.joint_dof_count:
            raise ValueError("initial_joint_qd must include every model joint velocity")
        if not torch.isfinite(initial_joint_q).all() or not torch.isfinite(initial_joint_qd).all():
            raise ValueError("initial joint positions and velocities must be finite")
        starts = model.joint_qd_start.numpy()
        owner = next(
            index
            for index in range(model.joint_count)
            if starts[index] <= joint_dof_index < starts[index + 1]
        )
        actual_type = int(model.joint_type.numpy()[owner])
        expected_type = {
            "revolute": int(newton.JointType.REVOLUTE),
            "prismatic": int(newton.JointType.PRISMATIC),
        }[joint_type]
        if actual_type != expected_type:
            raise ValueError(f"joint_type={joint_type!r} does not match the selected URDF joint")
        if int(model.joint_child.numpy()[owner]) != body_index:
            raise ValueError("body_index must be the moving link of the selected joint")

        solver_class = newton.solvers.SolverFeatherstone
        torch_device = wp.device_to_torch(model.device)
        dtype = torch.float32
        forces = [
            wp.from_torch(value.reshape(1, 3).to(device=torch_device, dtype=dtype), dtype=wp.vec3)
            for value in trajectory.force
        ]
        torques = [
            wp.from_torch(value.reshape(1, 3).to(device=torch_device, dtype=dtype), dtype=wp.vec3)
            for value in trajectory.torque
        ]
        application_points = [
            wp.from_torch(value.reshape(1, 3).to(device=torch_device, dtype=dtype), dtype=wp.vec3)
            for value in trajectory.application_point
        ]
        return cls(
            model=model,
            solver=None,
            body_index=body_index,
            joint_dof_index=joint_dof_index,
            forces=forces,
            torques=torques,
            application_points=application_points,
            dt=[float(value) for value in trajectory.dt],
            initial_joint_q=wp.from_torch(
                initial_joint_q.to(device=torch_device, dtype=dtype).contiguous()
            ),
            initial_joint_qd=wp.from_torch(
                initial_joint_qd.to(device=torch_device, dtype=dtype).contiguous()
            ),
            base_mass=wp.to_torch(model.body_mass).clone(),
            base_com=wp.to_torch(model.body_com).clone(),
            base_inertia=wp.to_torch(model.body_inertia).clone(),
            friction_load=friction_load,
            friction_velocity_scale=friction_velocity_scale,
            solver_class=solver_class,
        )

    def __call__(self, parameters: PhysicalParameters) -> tuple[Tensor, Tensor]:
        mass = self.base_mass.clone()
        com = self.base_com.clone()
        inertia = self.base_inertia.clone()
        mass[self.body_index] = parameters.mass.to(mass).reshape(())
        com[self.body_index] = parameters.com.to(com)
        inertia[self.body_index] = parameters.inertia.to(inertia)
        friction = parameters.friction.to(mass).reshape(1)
        trajectory = _FeatherstoneRollout.apply(mass, com, inertia, friction, self)
        return trajectory[:, :3], trajectory[:, 3:7]

    def loss(
        self,
        parameters: PhysicalParameters,
        target: InteractionTrajectory,
        *,
        position_weight: float = 1.0,
        rotation_weight: float = 1.0,
    ) -> Tensor:
        """Roll out Newton and evaluate the paper-aligned trajectory loss."""
        position, quaternion = self(parameters)
        return pose_loss(
            position,
            quaternion,
            target.position.to(position),
            target.quaternion_xyzw.to(quaternion),
            position_weight=position_weight,
            rotation_weight=rotation_weight,
        )
