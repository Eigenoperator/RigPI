"""Physical and gradient checks for the Featherstone backend."""

from pathlib import Path

import torch

from rigpi.backends import NewtonFeatherstoneBackend
from rigpi.data import InteractionTrajectory
from rigpi.gradcheck import compare_autograd_to_finite_difference
from rigpi.parameters import PhysicalParameters


def test_applied_force_rotates_hinge_and_mass_gradient_matches_finite_difference():
    dtype = torch.float32
    steps = 12
    trajectory = InteractionTrajectory(
        time=torch.arange(steps, dtype=dtype) * 0.01,
        position=torch.zeros((steps, 3), dtype=dtype),
        quaternion_xyzw=torch.tensor([[0, 0, 0, 1]] * steps, dtype=dtype),
        force=torch.tensor([[1, 0, 0]] * (steps - 1), dtype=dtype),
        torque=torch.zeros((steps - 1, 3), dtype=dtype),
        application_point=torch.tensor([[0, 0, 0.5]] * (steps - 1), dtype=dtype),
    )
    backend = NewtonFeatherstoneBackend.from_urdf(
        Path(__file__).parents[1] / "assets" / "simple_revolute.urdf",
        trajectory,
        body_index=1,
        joint_dof_index=0,
        initial_joint_q=torch.zeros(1, dtype=dtype),
        initial_joint_qd=torch.zeros(1, dtype=dtype),
        joint_type="revolute",
        friction_normal_force=5.0,
        friction_effective_radius=0.01,
        device="cpu",
    )
    com = torch.tensor([0, 0, 0.25], dtype=dtype)
    inertia = torch.diag(torch.tensor([0.02104, 0.02104, 0.0004167], dtype=dtype))
    friction = torch.tensor([0.1], dtype=dtype)

    def endpoint_rotation(mass):
        parameters = PhysicalParameters(mass, com, inertia, friction)
        return backend(parameters)[1][-1, 1]

    base = endpoint_rotation(torch.tensor([1.0], dtype=dtype))
    assert base.abs() > 1e-5
    analytic, numeric = compare_autograd_to_finite_difference(
        endpoint_rotation,
        torch.tensor([1.0], dtype=dtype),
        epsilon=1e-3,
        absolute_tolerance=2e-4,
        relative_tolerance=5e-2,
    )
    assert analytic.abs().sum() > 1e-5 and numeric.abs().sum() > 1e-5

    def endpoint_rotation_with_friction(mu):
        parameters = PhysicalParameters(torch.tensor([1.0], dtype=dtype), com, inertia, mu)
        return backend(parameters)[1][-1, 1]

    analytic_mu, numeric_mu = compare_autograd_to_finite_difference(
        endpoint_rotation_with_friction,
        torch.tensor([0.1], dtype=dtype),
        epsilon=1e-2,
        absolute_tolerance=2e-4,
        relative_tolerance=5e-2,
    )
    assert analytic_mu.abs().sum() > 1e-5 and numeric_mu.abs().sum() > 1e-5
