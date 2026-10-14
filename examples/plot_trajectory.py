"""Plot a saved identification trajectory and a point rigidly attached to the link."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def rotate_point(quaternion: np.ndarray, point: np.ndarray) -> np.ndarray:
    vector = quaternion[:, :3]
    scalar = quaternion[:, 3:4]
    cross = 2.0 * np.cross(vector, np.broadcast_to(point, vector.shape))
    return point + scalar * cross + np.cross(vector, cross)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/revolute_trajectory.png"))
    parser.add_argument(
        "--point",
        type=float,
        nargs=3,
        default=[0, 0, 0.5],
        help="Point in the moving link's body frame, in meters",
    )
    args = parser.parse_args()
    with np.load(args.trajectory) as data:
        time = data["time"]
        observed_position = data["observed_position"]
        fitted_position = data["fitted_position"]
        observed_quaternion = data["observed_quaternion_xyzw"]
        fitted_quaternion = data["fitted_quaternion_xyzw"]
    point = np.asarray(args.point)
    observed_point = observed_position + rotate_point(observed_quaternion, point)
    fitted_point = fitted_position + rotate_point(fitted_quaternion, point)
    observed_angle = 2 * np.unwrap(np.arctan2(observed_quaternion[:, 1], observed_quaternion[:, 3]))
    fitted_angle = 2 * np.unwrap(np.arctan2(fitted_quaternion[:, 1], fitted_quaternion[:, 3]))

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    axes[0].plot(observed_point[:, 0], observed_point[:, 2], "o-", label="observed", markersize=3)
    axes[0].plot(fitted_point[:, 0], fitted_point[:, 2], "--", label="fitted")
    axes[0].set(xlabel="attached point x (m)", ylabel="attached point z (m)", title="Link path")
    axes[0].axis("equal")
    axes[0].legend()
    axes[1].plot(time, observed_angle, label="observed")
    axes[1].plot(time, fitted_angle, "--", label="fitted")
    axes[1].set(xlabel="time (s)", ylabel="joint angle (rad)", title="Rotation about y")
    axes[2].plot(time, observed_point[:, 0], label="observed x")
    axes[2].plot(time, fitted_point[:, 0], "--", label="fitted x")
    axes[2].set(xlabel="time (s)", ylabel="attached point x (m)", title="Position over time")
    for axis in axes:
        axis.grid(alpha=0.25)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
