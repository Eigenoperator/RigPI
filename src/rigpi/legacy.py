from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import InteractionTrajectory
from .parameters import normalize_quaternion


def _timestamp_seconds(value: str | float) -> float:
    return datetime.fromisoformat(value).timestamp() if isinstance(value, str) else float(value)


def _rpy_degrees_to_quaternion(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(rpy).T / 2.0
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.stack(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        axis=-1,
    )


def load_processed_json(
    path: str | Path,
    *,
    application_point: Any | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> InteractionTrajectory:
    """Load the ``processed_traj.json`` format used by all three prototypes."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    position = torch.as_tensor(data["x"], dtype=dtype, device=device)
    quaternion = normalize_quaternion(torch.as_tensor(data["r"], dtype=dtype, device=device))
    steps = position.shape[0]
    point = data.get("application_point", application_point)
    if point is None:
        point = position[:-1]
    point_tensor = torch.as_tensor(point, dtype=dtype, device=device)
    if point_tensor.shape == (3,):
        point_tensor = point_tensor.expand(steps - 1, 3).clone()
    trajectory = InteractionTrajectory(
        time=torch.arange(steps, dtype=dtype, device=device) * float(data.get("dt", 0.01)),
        position=position,
        quaternion_xyzw=quaternion,
        force=torch.as_tensor(data["f"][: steps - 1], dtype=dtype, device=device),
        torque=torch.as_tensor(data["t"][: steps - 1], dtype=dtype, device=device),
        application_point=point_tensor,
    )
    trajectory.validate()
    return trajectory


def convert_xarm_json(
    path: str | Path,
    *,
    dt: float = 0.01,
    steps: int | None = None,
    skip: int = 0,
    application_point: Any | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> InteractionTrajectory:
    """Resample a raw xArm/ATI recording from the prototype repositories."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("data", payload) if isinstance(payload, dict) else payload
    records = records[skip:]
    if len(records) < 2:
        raise ValueError("recording must contain at least two samples after skip")
    times = np.asarray([_timestamp_seconds(row["timestamp"]) for row in records])
    times -= times[0]
    if steps is None:
        steps = int(np.floor(times[-1] / dt))
    sample_time = np.arange(steps + 1, dtype=np.float64) * dt
    if sample_time[-1] > times[-1]:
        raise ValueError("recording is shorter than the requested output duration")

    def values(keys: tuple[str, ...], fallback: str | None = None) -> np.ndarray:
        if fallback and fallback in records[0]:
            return np.asarray([row[fallback] for row in records], dtype=np.float64)
        return np.asarray([[row[key] for key in keys] for row in records], dtype=np.float64)

    def interpolate(array: np.ndarray) -> np.ndarray:
        return np.stack([np.interp(sample_time, times, array[:, i]) for i in range(3)], axis=-1)

    position = interpolate(values(("x", "y", "z")))
    force = interpolate(values(("fx", "fy", "fz"), "f"))
    torque = interpolate(values(("tx", "ty", "tz"), "t"))
    quaternion_samples = _rpy_degrees_to_quaternion(values(("roll", "pitch", "yaw")))
    nearest = np.searchsorted(times, sample_time, side="left").clip(max=len(times) - 1)
    quaternion = quaternion_samples[nearest]
    for index in range(1, len(quaternion)):
        if np.dot(quaternion[index - 1], quaternion[index]) < 0:
            quaternion[index] *= -1
    if application_point is None:
        application_point = position[:-1]
    point = np.asarray(application_point, dtype=np.float64)
    if point.shape == (3,):
        point = np.broadcast_to(point, (steps, 3)).copy()
    trajectory = InteractionTrajectory(
        time=torch.as_tensor(sample_time, dtype=dtype, device=device),
        position=torch.as_tensor(position, dtype=dtype, device=device),
        quaternion_xyzw=normalize_quaternion(
            torch.as_tensor(quaternion, dtype=dtype, device=device)
        ),
        force=torch.as_tensor(force[:-1], dtype=dtype, device=device),
        torque=torch.as_tensor(torque[:-1], dtype=dtype, device=device),
        application_point=torch.as_tensor(point, dtype=dtype, device=device),
    )
    trajectory.validate()
    return trajectory
