"""Integration check for the Newton free-body gradient path."""

import torch

from rigpi.backends import NewtonSemiImplicitBackend, RigidBodyInitialState
from rigpi.data import InteractionTrajectory
from rigpi.gradcheck import compare_autograd_to_finite_difference
from rigpi.parameters import PhysicalParameters, RawPhysicalParameters


def test_newton_force_translates_and_torque_rotates():
    dtype = torch.float32
    state = RigidBodyInitialState(
        position=torch.zeros(3, dtype=dtype),
        quaternion_xyzw=torch.tensor([0, 0, 0, 1], dtype=dtype),
        linear_velocity=torch.zeros(3, dtype=dtype),
        angular_velocity=torch.zeros(3, dtype=dtype),
    )

    def rollout(force, torque):
        trajectory = InteractionTrajectory(
            time=torch.tensor([0, 0.01], dtype=dtype),
            position=torch.zeros((2, 3), dtype=dtype),
            quaternion_xyzw=torch.tensor([[0, 0, 0, 1]] * 2, dtype=dtype),
            force=torch.tensor([force], dtype=dtype),
            torque=torch.tensor([torque], dtype=dtype),
            application_point=torch.zeros((1, 3), dtype=dtype),
        )
        backend = NewtonSemiImplicitBackend.from_trajectory(
            trajectory, state, gravity=0, device="cpu"
        )
        parameters = PhysicalParameters(
            mass=torch.tensor([1.0], dtype=dtype),
            com=torch.zeros(3, dtype=dtype),
            inertia=torch.eye(3, dtype=dtype) * 0.2,
            friction=torch.tensor([0.1], dtype=dtype),
        )
        return backend(parameters)

    force_position, force_quaternion = rollout([0, 1, 0], [0, 0, 0])
    assert force_position[-1, 1] > 0
    assert torch.allclose(force_quaternion[-1], force_quaternion[0], atol=1e-7)
    torque_position, torque_quaternion = rollout([0, 0, 0], [0, 0, 1])
    assert torch.allclose(torque_position[-1], torque_position[0], atol=1e-7)
    assert torque_quaternion[-1, 2].abs() > 1e-6


def test_newton_semi_implicit_mass_inertia_and_com_gradients():
    dtype = torch.float32
    steps = 4
    trajectory = InteractionTrajectory(
        time=torch.arange(steps, dtype=dtype) * 0.01,
        position=torch.zeros((steps, 3), dtype=dtype),
        quaternion_xyzw=torch.tensor([[0, 0, 0, 1]] * steps, dtype=dtype),
        force=torch.tensor([[0, 1, 0]] * (steps - 1), dtype=dtype),
        torque=torch.tensor([[0, 0, 0.1]] * (steps - 1), dtype=dtype),
        application_point=torch.tensor([[0, 0.1, 0]] * (steps - 1), dtype=dtype),
    )
    backend = NewtonSemiImplicitBackend.from_trajectory(
        trajectory,
        RigidBodyInitialState(
            position=torch.zeros(3, dtype=dtype),
            quaternion_xyzw=torch.tensor([0, 0, 0, 1], dtype=dtype),
            linear_velocity=torch.zeros(3, dtype=dtype),
            angular_velocity=torch.zeros(3, dtype=dtype),
        ),
        gravity=0,
        device="cpu",
    )
    raw = RawPhysicalParameters.from_physical(
        mass=torch.tensor([1.0], dtype=dtype),
        com=torch.tensor([0.05, 0, 0], dtype=dtype),
        principal_moments=torch.tensor([0.2, 0.2, 0.2], dtype=dtype),
        friction=torch.tensor([0.2], dtype=dtype),
    )
    for tensor in raw.tensors():
        tensor.requires_grad_(True)
    position, quaternion = backend(raw.decode())
    (position.square().sum() + quaternion[:, 2].square().sum()).backward()
    assert raw.mass_raw.grad is not None and raw.mass_raw.grad.abs().sum() > 0
    assert raw.com.grad is not None and raw.com.grad.abs().sum() > 0
    assert raw.principal_raw.grad is not None and raw.principal_raw.grad.abs().sum() > 0

    physical = raw.decode()

    def endpoint_y(mass):
        parameters = PhysicalParameters(
            mass=mass,
            com=physical.com.detach(),
            inertia=physical.inertia.detach(),
            friction=physical.friction.detach(),
        )
        return backend(parameters)[0][-1, 1]

    analytic, numeric = compare_autograd_to_finite_difference(
        endpoint_y,
        torch.tensor([1.0], dtype=dtype),
        epsilon=1e-3,
        absolute_tolerance=1e-5,
        relative_tolerance=2e-2,
    )
    assert analytic.abs().sum() > 1e-5 and numeric.abs().sum() > 1e-5


def test_newton_surface_friction_has_coefficient_gradient():
    dtype = torch.float32
    trajectory = InteractionTrajectory(
        time=torch.tensor([0, 0.01, 0.02], dtype=dtype),
        position=torch.zeros((3, 3), dtype=dtype),
        quaternion_xyzw=torch.tensor([[0, 0, 0, 1]] * 3, dtype=dtype),
        force=torch.zeros((2, 3), dtype=dtype),
        torque=torch.zeros((2, 3), dtype=dtype),
        application_point=torch.zeros((2, 3), dtype=dtype),
    )
    backend = NewtonSemiImplicitBackend.from_trajectory(
        trajectory,
        RigidBodyInitialState(
            position=torch.zeros(3, dtype=dtype),
            quaternion_xyzw=torch.tensor([0, 0, 0, 1], dtype=dtype),
            linear_velocity=torch.tensor([1, 0, 0], dtype=dtype),
            angular_velocity=torch.zeros(3, dtype=dtype),
        ),
        gravity=0,
        normal_force=5.0,
        contact_point_body=(0, 0, -0.1),
        device="cpu",
    )
    raw = RawPhysicalParameters.from_physical(
        mass=torch.tensor([1.0], dtype=dtype),
        com=torch.zeros(3, dtype=dtype),
        principal_moments=torch.tensor([0.2, 0.2, 0.2], dtype=dtype),
        friction=torch.tensor([0.2], dtype=dtype),
    )
    raw.friction_raw.requires_grad_(True)
    position, _ = backend(raw.decode())
    position[-1, 0].backward()
    assert raw.friction_raw.grad is not None and raw.friction_raw.grad.abs().sum() > 0

    physical = raw.decode()

    def endpoint_x(friction):
        parameters = PhysicalParameters(
            mass=physical.mass.detach(),
            com=physical.com.detach(),
            inertia=physical.inertia.detach(),
            friction=friction,
        )
        return backend(parameters)[0][-1, 0]

    analytic, numeric = compare_autograd_to_finite_difference(
        endpoint_x,
        torch.tensor([0.2], dtype=dtype),
        epsilon=1e-3,
        absolute_tolerance=1e-5,
        relative_tolerance=2e-2,
    )
    assert analytic.abs().sum() > 1e-5 and numeric.abs().sum() > 1e-5
