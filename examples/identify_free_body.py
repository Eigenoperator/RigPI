"""Identify a free rigid body with Newton's semi-implicit integrator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import warp as wp

from rigpi.backends import NewtonSemiImplicitBackend, RigidBodyInitialState
from rigpi.data import InteractionTrajectory
from rigpi.losses import pose_loss
from rigpi.optimizer import RigPIOptimizer
from rigpi.parameters import RawPhysicalParameters
from rigpi.prior import VLMPrior

ROOT = Path(__file__).parents[1]


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float32)


def synthetic_signal(value, time):
    """Expand a constant 3-vector or a per-axis sinusoidal wrench signal."""
    if not isinstance(value, dict):
        return tensor(value).expand(time.numel(), 3).clone()
    offset = tensor(value["offset"])
    amplitude = tensor(value["amplitude"])
    frequency = tensor(value["angular_frequency"])
    phase = tensor(value.get("phase", [0, 0, 0]))
    return offset + amplitude * torch.sin(time[:, None] * frequency + phase)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "free_body_demo.json")
    parser.add_argument(
        "--prior", type=Path, default=ROOT / "configs" / "vlm_prior_free_body_demo.json"
    )
    parser.add_argument(
        "--trajectory", type=Path, help="Aligned measured .npz; omit for synthetic demo"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "free_body_demo.json")
    parser.add_argument(
        "--trajectory-output", type=Path, help="Save full input and fitted poses to .npz"
    )
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    sim = cfg["simulation"]
    steps = int(sim["steps"])
    synthetic_time = torch.arange(steps - 1, dtype=torch.float32) * float(sim["dt"])
    trajectory = (
        InteractionTrajectory.from_npz(args.trajectory, dtype=torch.float32)
        if args.trajectory
        else InteractionTrajectory(
            time=torch.arange(steps, dtype=torch.float32) * float(sim["dt"]),
            position=torch.zeros((steps, 3), dtype=torch.float32),
            quaternion_xyzw=tensor([[0, 0, 0, 1]] * steps),
            force=synthetic_signal(sim["force"], synthetic_time),
            torque=synthetic_signal(sim["torque"], synthetic_time),
            application_point=synthetic_signal(sim["application_point"], synthetic_time),
        )
    )
    initial_state = RigidBodyInitialState(
        position=trajectory.position[0] if args.trajectory else tensor(sim["initial_position"]),
        quaternion_xyzw=(
            trajectory.quaternion_xyzw[0]
            if args.trajectory
            else tensor(sim["initial_quaternion_xyzw"])
        ),
        linear_velocity=(
            trajectory.linear_velocity[0]
            if args.trajectory and trajectory.linear_velocity is not None
            else tensor(sim["initial_linear_velocity"])
        ),
        angular_velocity=(
            trajectory.angular_velocity[0]
            if args.trajectory and trajectory.angular_velocity is not None
            else tensor(sim["initial_angular_velocity"])
        ),
    )
    device = str(sim["device"])
    if device.startswith("cuda") and not (torch.cuda.is_available() and wp.is_cuda_available()):
        device = "cpu"
    backend = NewtonSemiImplicitBackend.from_trajectory(
        trajectory,
        initial_state,
        gravity=float(sim["gravity"]),
        normal_force=float(sim["normal_force"]),
        contact_point_body=tuple(sim["contact_point_body"]),
        device=device,
    )
    if args.trajectory:
        target_position = trajectory.position.to(wp.device_to_torch(backend.model.device))
        target_quaternion = trajectory.quaternion_xyzw.to(target_position)
    else:
        truth = cfg["ground_truth"]
        raw_truth = RawPhysicalParameters.from_physical(
            mass=tensor([truth["mass"]]),
            com=tensor(truth["com"]),
            principal_moments=tensor(truth["principal_inertia"]),
            friction=tensor([truth["friction"]]),
        )
        with torch.no_grad():
            target_position, target_quaternion = backend(raw_truth.decode())
            target_position = target_position.detach()
            target_quaternion = target_quaternion.detach()
    raw, bounds = VLMPrior.from_json(args.prior).to_optimization_inputs()
    optimizer = RigPIOptimizer(
        bounds=bounds,
        penalty_weight=float(cfg["optimizer"]["penalty_weight"]),
        relative_step=float(cfg["optimizer"]["relative_step"]),
        iterations=int(cfg["optimizer"]["iterations"]),
    )

    def objective(parameters):
        position, quaternion = backend(parameters)
        return pose_loss(position, quaternion, target_position, target_quaternion)

    result = optimizer.identify(raw, objective)
    final = result.parameters
    report = {
        "source": "measured" if args.trajectory else "synthetic",
        "converged": result.converged,
        "iterations": result.history[-1].iteration,
        "initial_loss": result.history[0].total_loss,
        "final_loss": result.history[-1].total_loss,
        "mass": float(final.mass),
        "com": final.com.tolist(),
        "inertia": final.inertia.tolist(),
        "friction": float(final.friction),
    }
    if not args.trajectory:
        truth = cfg["ground_truth"]
        report["synthetic_errors"] = {
            "mass_relative": abs(float(final.mass) / truth["mass"] - 1.0),
            "com_absolute_m": float(torch.linalg.vector_norm(final.com - tensor(truth["com"]))),
            "inertia_diagonal_relative": (
                (final.inertia.diagonal() / tensor(truth["principal_inertia"]) - 1.0).abs().tolist()
            ),
            "friction_relative": abs(float(final.friction) / truth["friction"] - 1.0),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.trajectory_output:
        with torch.no_grad():
            fitted_position, fitted_quaternion = backend(final)
        args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.trajectory_output,
            time=trajectory.time.detach().cpu().numpy(),
            position=target_position.detach().cpu().numpy(),
            quaternion_xyzw=target_quaternion.detach().cpu().numpy(),
            force=trajectory.force.detach().cpu().numpy(),
            torque=trajectory.torque.detach().cpu().numpy(),
            application_point=trajectory.application_point.detach().cpu().numpy(),
            observed_position=target_position.detach().cpu().numpy(),
            observed_quaternion_xyzw=target_quaternion.detach().cpu().numpy(),
            fitted_position=fitted_position.detach().cpu().numpy(),
            fitted_quaternion_xyzw=fitted_quaternion.detach().cpu().numpy(),
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
