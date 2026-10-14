"""Minimal end-to-end Newton Featherstone gradient smoke test."""

from pathlib import Path

import torch
import warp as wp

from rigpi.backends import NewtonFeatherstoneBackend
from rigpi.data import InteractionTrajectory
from rigpi.parameters import RawPhysicalParameters


def main() -> None:
    steps = 20
    dtype = torch.float32
    trajectory = InteractionTrajectory(
        time=torch.arange(steps, dtype=dtype) * 0.01,
        position=torch.zeros((steps, 3), dtype=dtype),
        quaternion_xyzw=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * steps, dtype=dtype),
        force=torch.tensor([[1.0, 0.0, 0.0]] * (steps - 1), dtype=dtype),
        torque=torch.tensor([[0.0, 1.0, 0.0]] * (steps - 1), dtype=dtype),
        application_point=torch.tensor([[0.0, 0.0, 0.5]] * (steps - 1), dtype=dtype),
    )
    asset = Path(__file__).parents[1] / "assets" / "simple_revolute.urdf"
    backend = NewtonFeatherstoneBackend.from_urdf(
        asset,
        trajectory,
        body_index=1,
        joint_dof_index=0,
        initial_joint_q=torch.zeros(1, dtype=dtype),
        initial_joint_qd=torch.tensor([0.1], dtype=dtype),
        joint_type="revolute",
        friction_normal_force=5.0,
        friction_effective_radius=0.01,
        device="cuda:0" if torch.cuda.is_available() and wp.is_cuda_available() else "cpu",
    )
    raw = RawPhysicalParameters.from_physical(
        mass=torch.tensor([1.0], dtype=dtype),
        com=torch.tensor([0.0, 0.0, 0.25], dtype=dtype),
        principal_moments=torch.tensor([0.02104, 0.02104, 0.0004167], dtype=dtype),
        friction=torch.tensor([0.2], dtype=dtype),
    )
    for value in raw.tensors():
        value.requires_grad_(True)

    _, quaternion = backend(raw.decode())
    loss = (quaternion[-1, 1] - 0.2).square()
    loss.backward()
    gradients = {
        "mass": raw.mass_raw.grad,
        "com": raw.com.grad,
        "inertia": raw.principal_raw.grad,
        "friction": raw.friction_raw.grad,
    }
    if any(value is None or not torch.isfinite(value).all() for value in gradients.values()):
        raise RuntimeError(f"invalid Featherstone gradients: {gradients}")
    print(f"loss={float(loss.detach()):.6g}")
    for name, value in gradients.items():
        print(f"{name}_gradient={value}")


if __name__ == "__main__":
    main()
