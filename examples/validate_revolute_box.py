"""Check mass and friction recovery when box geometry is known from the URDF."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from rigpi.backends import NewtonFeatherstoneBackend
from rigpi.data import InteractionTrajectory
from rigpi.losses import pose_loss
from rigpi.parameters import PhysicalParameters
from rigpi.prior import VLMPrior

ROOT = Path(__file__).parents[1]


def box_geometry(urdf_path: Path, link_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Read box center and unit-mass inertia from collision geometry, not inertial tags."""
    root = ET.parse(urdf_path).getroot()
    link = root.find(f"./link[@name='{link_name}']")
    if link is None:
        raise ValueError(f"link {link_name!r} was not found")
    collision = link.find("collision")
    if collision is None:
        raise ValueError("moving link needs box collision geometry")
    box = collision.find("geometry/box")
    if box is None:
        raise ValueError("moving link needs box collision geometry")
    size = torch.tensor([float(value) for value in box.attrib["size"].split()])
    origin = collision.find("origin")
    if origin is not None and origin.get("rpy", "0 0 0") != "0 0 0":
        raise ValueError("rotated box collision geometry is not supported")
    center = torch.tensor(
        [float(value) for value in origin.get("xyz", "0 0 0").split()]
        if origin is not None
        else [0.0, 0.0, 0.0]
    )
    if (
        size.shape != (3,)
        or center.shape != (3,)
        or not torch.isfinite(size).all()
        or not torch.isfinite(center).all()
        or torch.any(size <= 0)
    ):
        raise ValueError("box size and center must be finite 3-vectors")
    x, y, z = size
    unit_inertia = torch.diag(
        torch.stack(((y * y + z * z) / 12, (x * x + z * z) / 12, (x * x + y * y) / 12))
    )
    return center, unit_inertia


def bounded_initial(value: float, lower: float, upper: float) -> torch.Tensor:
    if not lower < value < upper:
        raise ValueError("initial value must lie strictly inside its bounds")
    fraction = torch.tensor((value - lower) / (upper - lower))
    return torch.logit(fraction).requires_grad_()


def torque_trajectory(simulation: dict, segments: list[dict]) -> InteractionTrajectory:
    steps = int(simulation["steps"])
    values = []
    for segment in segments:
        values.extend([float(segment["value"])] * (int(segment["until_step"]) - len(values)))
    if len(values) != steps - 1:
        raise ValueError("torque segments must cover exactly N-1 intervals")
    return InteractionTrajectory(
        time=torch.arange(steps, dtype=torch.float32) * float(simulation["dt"]),
        position=torch.zeros((steps, 3), dtype=torch.float32),
        quaternion_xyzw=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * steps),
        force=torch.zeros((steps - 1, 3), dtype=torch.float32),
        torque=torch.tensor([[0.0, value, 0.0] for value in values]),
        application_point=torch.zeros((steps - 1, 3), dtype=torch.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/revolute_box_validation.json"
    )
    parser.add_argument("--prior", type=Path, default=ROOT / "configs/vlm_prior_demo.json")
    parser.add_argument("--urdf", type=Path, default=ROOT / "assets/simple_revolute.urdf")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/revolute_box_validation.json"
    )
    parser.add_argument("--check", action="store_true", help="Require less than 2% parameter error")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    sim = cfg["simulation"]
    prior = VLMPrior.from_json(args.prior)
    center, unit_inertia = box_geometry(args.urdf, "moving_link")

    def parameters(mass: torch.Tensor, friction: torch.Tensor) -> PhysicalParameters:
        return PhysicalParameters(
            mass=mass.reshape(1),
            com=center,
            inertia=mass * unit_inertia,
            friction=friction.reshape(1),
        )

    def backend(data: dict, segments: list[dict]) -> NewtonFeatherstoneBackend:
        return NewtonFeatherstoneBackend.from_urdf(
            args.urdf,
            torque_trajectory(sim, segments),
            body_index=1,
            joint_dof_index=0,
            initial_joint_q=torch.tensor(data["initial_joint_q"], dtype=torch.float32),
            initial_joint_qd=torch.tensor(data["initial_joint_qd"], dtype=torch.float32),
            joint_type=sim["joint_type"],
            friction_normal_force=float(sim["friction_normal_force"]),
            friction_effective_radius=float(sim["friction_effective_radius"]),
            device="cpu",
        )

    training = backend(sim, sim["torque_y_segments"])
    heldout_cfg = cfg["heldout"]
    heldout = backend(heldout_cfg, heldout_cfg["torque_y_segments"])
    truth = cfg["ground_truth"]
    true_parameters = parameters(torch.tensor(truth["mass"]), torch.tensor(truth["friction"]))
    with torch.no_grad():
        target = tuple(value.detach() for value in training(true_parameters))
        heldout_target = tuple(value.detach() for value in heldout(true_parameters))

    mass_lower, mass_upper = prior.bounds["mass"]
    friction_lower, friction_upper = prior.bounds["friction"]
    mass_raw = bounded_initial(prior.initial["mass"], mass_lower, mass_upper)
    friction_raw = bounded_initial(prior.initial["friction"], friction_lower, friction_upper)

    def decoded() -> tuple[torch.Tensor, torch.Tensor]:
        mass = mass_lower + (mass_upper - mass_lower) * torch.sigmoid(mass_raw)
        friction = friction_lower + (friction_upper - friction_lower) * torch.sigmoid(friction_raw)
        return mass, friction

    optimizer = torch.optim.LBFGS(
        [mass_raw, friction_raw],
        lr=1.0,
        max_iter=int(cfg["optimizer"]["max_iter"]),
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
    )
    evaluations = 0

    def closure() -> torch.Tensor:
        nonlocal evaluations
        optimizer.zero_grad()
        mass, friction = decoded()
        loss = pose_loss(*training(parameters(mass, friction)), *target)
        loss.backward()
        evaluations += 1
        return loss

    initial_loss = float(closure().detach())
    optimizer.step(closure)
    with torch.no_grad():
        mass, friction = decoded()
        fitted = parameters(mass, friction)
        training_loss = float(pose_loss(*training(fitted), *target))
        heldout_loss = float(pose_loss(*heldout(fitted), *heldout_target))

    report = {
        "mode": "synthetic; known box geometry; mass and joint friction only",
        "geometry_source": str(args.urdf),
        "prior_mass": prior.initial["mass"],
        "prior_friction": prior.initial["friction"],
        "mass": float(mass),
        "friction": float(friction),
        "mass_relative_error": abs(float(mass) / truth["mass"] - 1.0),
        "friction_relative_error": abs(float(friction) / truth["friction"] - 1.0),
        "initial_pose_loss": initial_loss,
        "training_pose_loss": training_loss,
        "heldout_pose_loss": heldout_loss,
        "closure_evaluations": evaluations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.check and (
        report["mass_relative_error"] >= 0.02
        or report["friction_relative_error"] >= 0.02
        or report["heldout_pose_loss"] >= 1e-6
    ):
        raise SystemExit("box validation failed its parameter or held-out trajectory checks")


if __name__ == "__main__":
    main()
