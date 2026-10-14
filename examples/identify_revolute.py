"""Identify one moving link from synthetic or aligned measured data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import warp as wp

from rigpi.backends import NewtonFeatherstoneBackend
from rigpi.data import InteractionTrajectory
from rigpi.losses import pose_loss
from rigpi.optimizer import RigPIOptimizer
from rigpi.parameters import RawPhysicalParameters
from rigpi.prior import VLMPrior

ROOT = Path(__file__).parents[1]


def as_tensor(value):
    return torch.as_tensor(value, dtype=torch.float32)


def synthetic_inputs(simulation: dict) -> InteractionTrajectory:
    steps = int(simulation["steps"])
    return InteractionTrajectory(
        time=torch.arange(steps, dtype=torch.float32) * float(simulation["dt"]),
        position=torch.zeros((steps, 3), dtype=torch.float32),
        quaternion_xyzw=as_tensor([[0, 0, 0, 1]] * steps),
        force=as_tensor([simulation["force"]] * (steps - 1)),
        torque=as_tensor([simulation["torque"]] * (steps - 1)),
        application_point=as_tensor([simulation["application_point"]] * (steps - 1)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "revolute_demo.json")
    parser.add_argument("--prior", type=Path, default=ROOT / "configs" / "vlm_prior_demo.json")
    parser.add_argument(
        "--trajectory", type=Path, help="Aligned measured .npz; omit for synthetic demo"
    )
    parser.add_argument("--urdf", type=Path, default=ROOT / "assets" / "simple_revolute.urdf")
    parser.add_argument("--body-index", type=int, default=1)
    parser.add_argument("--joint-dof-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "revolute_demo.json")
    parser.add_argument(
        "--trajectory-output", type=Path, help="Save observed and fitted poses to .npz"
    )
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    simulation = cfg["simulation"]
    inputs = (
        InteractionTrajectory.from_npz(args.trajectory, dtype=torch.float32)
        if args.trajectory
        else synthetic_inputs(simulation)
    )
    device = str(simulation["device"])
    if device.startswith("cuda") and not (torch.cuda.is_available() and wp.is_cuda_available()):
        device = "cpu"
    backend = NewtonFeatherstoneBackend.from_urdf(
        args.urdf,
        inputs,
        body_index=args.body_index,
        joint_dof_index=args.joint_dof_index,
        initial_joint_q=as_tensor(simulation["initial_joint_q"]),
        initial_joint_qd=as_tensor(simulation["initial_joint_qd"]),
        joint_type=simulation["joint_type"],
        friction_normal_force=float(simulation["friction_normal_force"]),
        friction_effective_radius=simulation.get("friction_effective_radius"),
        device=device,
    )

    if args.trajectory:
        target_position = inputs.position.to(backend.base_mass)
        target_quaternion = inputs.quaternion_xyzw.to(target_position)
    else:
        ground_truth = cfg["ground_truth"]
        truth = RawPhysicalParameters.from_physical(
            mass=as_tensor([ground_truth["mass"]]),
            com=as_tensor(ground_truth["com"]),
            principal_moments=as_tensor(ground_truth["principal_inertia"]),
            friction=as_tensor([ground_truth["friction"]]),
        )
        with torch.no_grad():
            target_position, target_quaternion = backend(truth.decode())
            target_position = target_position.detach()
            target_quaternion = target_quaternion.detach()

    initial, bounds = VLMPrior.from_json(args.prior).to_optimization_inputs()
    optimizer = RigPIOptimizer(
        bounds=bounds,
        penalty_weight=float(cfg["optimizer"]["penalty_weight"]),
        relative_step=float(cfg["optimizer"]["relative_step"]),
        iterations=int(cfg["optimizer"]["iterations"]),
    )

    def objective(parameters):
        position, quaternion = backend(parameters)
        return pose_loss(position, quaternion, target_position, target_quaternion)

    result = optimizer.identify(initial, objective)
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
            "com_absolute_m": float(torch.linalg.vector_norm(final.com - as_tensor(truth["com"]))),
            "inertia_diagonal_relative": (
                (final.inertia.diagonal() / as_tensor(truth["principal_inertia"]) - 1.0)
                .abs()
                .tolist()
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
            time=inputs.time.detach().cpu().numpy(),
            position=target_position.detach().cpu().numpy(),
            quaternion_xyzw=target_quaternion.detach().cpu().numpy(),
            force=inputs.force.detach().cpu().numpy(),
            torque=inputs.torque.detach().cpu().numpy(),
            application_point=inputs.application_point.detach().cpu().numpy(),
            observed_position=target_position.detach().cpu().numpy(),
            observed_quaternion_xyzw=target_quaternion.detach().cpu().numpy(),
            fitted_position=fitted_position.detach().cpu().numpy(),
            fitted_quaternion_xyzw=fitted_quaternion.detach().cpu().numpy(),
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
