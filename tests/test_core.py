import numpy as np
import pytest
import torch

from rigpi.backends import RigidBodyInitialState, TorchRigidBodyBackend
from rigpi.bounds import ParameterBounds
from rigpi.data import InteractionTrajectory
from rigpi.friction import coulomb_joint_load, smooth_coulomb_force, smooth_coulomb_joint_torque
from rigpi.gradcheck import compare_autograd_to_finite_difference
from rigpi.legacy import load_processed_json
from rigpi.losses import pose_loss, quadratic_bound_penalty
from rigpi.optimizer import RigPIOptimizer
from rigpi.parameters import RawPhysicalParameters
from rigpi.prior import VLMPrior
from rigpi.scheduler import ScaleAwareLearningRate

DTYPE = torch.float64


def test_quaternion_loss_is_antipodal_invariant():
    position = torch.zeros((1, 3), dtype=DTYPE)
    quaternion = torch.tensor([[0.1, 0.2, 0.3, 0.9]], dtype=DTYPE)
    loss = pose_loss(position, quaternion, position, -quaternion)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1.0e-12)


def test_quadratic_penalty_and_gradient():
    value = torch.tensor([-1.0, 0.5, 3.0], dtype=DTYPE, requires_grad=True)
    lower = torch.tensor([0.0, 0.0, 0.0], dtype=DTYPE)
    upper = torch.tensor([2.0, 2.0, 2.0], dtype=DTYPE)
    penalty = quadratic_bound_penalty(value, lower, upper)
    penalty.backward()
    assert penalty.item() == 2.0
    assert torch.allclose(value.grad, torch.tensor([-2.0, 0.0, 2.0], dtype=DTYPE))


def test_physical_inertia_is_positive_and_satisfies_triangle_inequality():
    raw = RawPhysicalParameters.from_physical(
        mass=torch.tensor([1.0], dtype=DTYPE),
        com=torch.zeros(3, dtype=DTYPE),
        principal_moments=torch.tensor([2.0, 3.0, 4.0], dtype=DTYPE),
        friction=torch.tensor([0.3], dtype=DTYPE),
    )
    physical = raw.decode()
    eigenvalues = torch.linalg.eigvalsh(physical.inertia)
    assert torch.all(eigenvalues > 0)
    assert eigenvalues[0] + eigenvalues[1] >= eigenvalues[2]
    assert physical.mass.item() > 0
    assert physical.friction.item() >= 0


def test_scale_aware_step_is_one_percent_of_prior_scale():
    value = torch.tensor([2.0, 0.1], dtype=DTYPE)
    scheduler = ScaleAwareLearningRate(
        initial=value,
        lower=torch.tensor([0.0, -1.0], dtype=DTYPE),
        upper=torch.tensor([4.0, 1.0], dtype=DTYPE),
    )
    gradient = torch.tensor([100.0, -10.0], dtype=DTYPE)
    updated = scheduler.step(value, gradient)
    assert torch.allclose((updated - value).abs(), 0.01 * scheduler.scale)


def test_surface_friction_has_mu_gradient():
    velocity = torch.tensor([[0.2, -0.1]], dtype=DTYPE)
    mu = torch.tensor([[0.4]], dtype=DTYPE, requires_grad=True)
    force = smooth_coulomb_force(velocity, torch.tensor([[9.81]], dtype=DTYPE), mu)
    force.square().sum().backward()
    assert mu.grad is not None
    assert mu.grad.abs().item() > 0


def test_joint_friction_has_mu_gradient():
    velocity = torch.tensor([0.2], dtype=DTYPE)
    mu = torch.tensor([0.4], dtype=DTYPE, requires_grad=True)
    torque = smooth_coulomb_joint_torque(velocity, mu, load=3.0)
    torque.square().sum().backward()
    assert mu.grad is not None
    assert mu.grad.abs().item() > 0


def test_joint_friction_gradient_matches_finite_difference():
    velocity = torch.tensor([0.2], dtype=DTYPE)

    def objective(mu):
        return smooth_coulomb_joint_torque(velocity, mu, load=3.0).square().sum()

    compare_autograd_to_finite_difference(
        objective, torch.tensor([0.4], dtype=DTYPE), epsilon=1.0e-6
    )


def test_joint_friction_load_has_correct_force_or_torque_units():
    assert coulomb_joint_load(5.0, joint_type="prismatic") == 5.0
    assert coulomb_joint_load(5.0, joint_type="revolute", effective_radius=0.01) == 0.05


def test_vlm_prior_file_to_physical_parameters():
    from pathlib import Path

    path = Path(__file__).parents[1] / "configs" / "vlm_prior_demo.json"
    raw, bounds = VLMPrior.from_json(path).to_optimization_inputs(dtype=DTYPE)
    assert torch.allclose(raw.decode().mass, torch.tensor([0.8], dtype=DTYPE), atol=1e-7)
    assert bounds.penalty(raw.decode()) == 0


def test_prior_rejects_nonphysical_mass_and_asymmetric_inertia_bounds():
    import json
    from pathlib import Path

    payload = json.loads(
        (Path(__file__).parents[1] / "configs" / "vlm_prior_demo.json").read_text()
    )
    payload["initial"]["mass"] = -1.0
    with pytest.raises(ValueError, match="mass"):
        VLMPrior(payload["initial"], payload["bounds"]).to_optimization_inputs()
    payload["initial"]["mass"] = 0.8
    payload["bounds"]["inertia_lower"][0][1] = -0.004
    with pytest.raises(ValueError, match="symmetric"):
        VLMPrior(payload["initial"], payload["bounds"]).to_optimization_inputs()


def test_npz_rejects_invalid_quaternion_and_time_shape(tmp_path):
    path = tmp_path / "invalid.npz"
    data = {
        "time": np.array([0.0, 0.01]),
        "position": np.zeros((2, 3)),
        "quaternion_xyzw": np.zeros((2, 4)),
        "force": np.zeros((1, 3)),
        "torque": np.zeros((1, 3)),
        "application_point": np.zeros((1, 3)),
    }
    np.savez(path, **data)
    with pytest.raises(ValueError, match="quaternions"):
        InteractionTrajectory.from_npz(path)
    data["quaternion_xyzw"][:, 3] = 1.0
    data["time"] = data["time"].reshape(2, 1)
    np.savez(path, **data)
    with pytest.raises(ValueError, match="time"):
        InteractionTrajectory.from_npz(path)


def test_optimizer_updates_mass_and_friction_without_hard_clamping():
    raw = RawPhysicalParameters.from_physical(
        mass=torch.tensor([0.5], dtype=DTYPE),
        com=torch.zeros(3, dtype=DTYPE),
        principal_moments=torch.tensor([0.2, 0.2, 0.2], dtype=DTYPE),
        friction=torch.tensor([0.1], dtype=DTYPE),
    )
    physical = raw.decode()
    zero_inertia = torch.full_like(physical.inertia, -10.0)
    high_inertia = torch.full_like(physical.inertia, 10.0)
    bounds = ParameterBounds(
        mass_lower=torch.tensor([0.1], dtype=DTYPE),
        mass_upper=torch.tensor([2.0], dtype=DTYPE),
        com_lower=torch.full((3,), -1.0, dtype=DTYPE),
        com_upper=torch.full((3,), 1.0, dtype=DTYPE),
        inertia_lower=zero_inertia,
        inertia_upper=high_inertia,
        friction_lower=torch.tensor([0.0], dtype=DTYPE),
        friction_upper=torch.tensor([1.0], dtype=DTYPE),
    )
    optimizer = RigPIOptimizer(bounds=bounds, iterations=100)

    def objective(parameters):
        return (parameters.mass - 1.0).square().sum() + (parameters.friction - 0.4).square().sum()

    result = optimizer.identify(raw, objective)
    assert abs(result.parameters.mass.item() - 1.0) < 0.4
    assert abs(result.parameters.friction.item() - 0.4) < 0.25
    assert all(
        after.total_loss <= before.total_loss
        for before, after in zip(result.history, result.history[1:])
    )
    lower, upper = bounds.vector_bounds()
    initial = result.history[0].parameters.as_vector()
    scale = torch.maximum(
        torch.maximum(initial.abs(), upper - lower), torch.full_like(initial, 1e-8)
    )
    for before, after in zip(result.history, result.history[1:]):
        change = (after.parameters.as_vector() - before.parameters.as_vector()).abs()
        assert torch.all(change <= 0.01 * scale * 1.0001)

    with pytest.raises(ValueError, match="iterations must be positive"):
        RigPIOptimizer(bounds=bounds, iterations=0)
    with pytest.raises(ValueError, match="non-finite or non-scalar objective"):
        optimizer.identify(raw, lambda parameters: parameters.mass.sum() * float("nan"))


def test_unconstrained_rollout_has_mass_and_inertia_gradients():
    steps = 5
    trajectory = InteractionTrajectory(
        time=torch.arange(steps, dtype=DTYPE) * 0.01,
        position=torch.zeros((steps, 3), dtype=DTYPE),
        quaternion_xyzw=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * steps, dtype=DTYPE),
        force=torch.tensor([[1.0, 0.0, 0.0]] * (steps - 1), dtype=DTYPE),
        torque=torch.tensor([[0.0, 0.0, 0.1]] * (steps - 1), dtype=DTYPE),
        application_point=torch.zeros((steps - 1, 3), dtype=DTYPE),
    )
    backend = TorchRigidBodyBackend(
        trajectory,
        RigidBodyInitialState(
            position=torch.zeros(3, dtype=DTYPE),
            quaternion_xyzw=torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=DTYPE),
            linear_velocity=torch.zeros(3, dtype=DTYPE),
            angular_velocity=torch.zeros(3, dtype=DTYPE),
        ),
        gravity=(0.0, 0.0, 0.0),
    )
    raw = RawPhysicalParameters.from_physical(
        mass=torch.tensor([1.0], dtype=DTYPE),
        com=torch.zeros(3, dtype=DTYPE),
        principal_moments=torch.tensor([0.1, 0.1, 0.1], dtype=DTYPE),
        friction=torch.tensor([0.2], dtype=DTYPE),
    )
    for value in raw.tensors():
        value.requires_grad_(True)
    position, quaternion = backend(raw.decode())
    (position.square().sum() + quaternion[..., :3].square().sum()).backward()
    assert raw.mass_raw.grad is not None and raw.mass_raw.grad.abs().item() > 0
    assert raw.principal_raw.grad is not None and raw.principal_raw.grad.abs().sum().item() > 0


def test_free_body_origin_moves_around_offset_com_during_pure_rotation():
    steps = 3
    trajectory = InteractionTrajectory(
        time=torch.arange(steps, dtype=DTYPE) * 0.01,
        position=torch.zeros((steps, 3), dtype=DTYPE),
        quaternion_xyzw=torch.tensor([[0, 0, 0, 1]] * steps, dtype=DTYPE),
        force=torch.zeros((steps - 1, 3), dtype=DTYPE),
        torque=torch.zeros((steps - 1, 3), dtype=DTYPE),
        application_point=torch.zeros((steps - 1, 3), dtype=DTYPE),
    )
    backend = TorchRigidBodyBackend(
        trajectory,
        RigidBodyInitialState(
            position=torch.zeros(3, dtype=DTYPE),
            quaternion_xyzw=torch.tensor([0, 0, 0, 1], dtype=DTYPE),
            linear_velocity=torch.tensor([0, -1, 0], dtype=DTYPE),
            angular_velocity=torch.tensor([0, 0, 1], dtype=DTYPE),
        ),
        gravity=(0, 0, 0),
    )
    raw = RawPhysicalParameters.from_physical(
        mass=torch.tensor([1.0], dtype=DTYPE),
        com=torch.tensor([1.0, 0, 0], dtype=DTYPE),
        principal_moments=torch.tensor([0.2, 0.2, 0.2], dtype=DTYPE),
        friction=torch.tensor([0.2], dtype=DTYPE),
    )
    position, quaternion = backend(raw.decode())
    from rigpi.parameters import quaternion_to_matrix

    com_world = position + torch.matmul(quaternion_to_matrix(quaternion), raw.decode().com)
    assert torch.allclose(com_world, com_world[0].expand_as(com_world), atol=1e-10)
    assert position[1, 1] < 0  # origin circles the stationary COM


def test_load_processed_legacy_json(tmp_path):
    path = tmp_path / "processed.json"
    path.write_text(
        '{"x":[[0,0,0],[1,0,0]],"r":[[0,0,0,1],[0,0,0,1]],'
        '"f":[[1,0,0],[1,0,0]],"t":[[0,0,0],[0,0,0]],"dt":0.1}',
        encoding="utf-8",
    )
    trajectory = load_processed_json(path)
    assert trajectory.position.shape == (2, 3)
    assert trajectory.force.shape == (1, 3)
    assert torch.allclose(trajectory.dt, torch.tensor([0.1], dtype=DTYPE))
