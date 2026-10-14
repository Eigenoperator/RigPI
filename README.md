# RigPI

Official implementation of [RigPI: Dynamic Parameter Identification of Rigid
Body via VLM-Seeded Differentiable Simulation](https://arxiv.org/abs/2606.25212).
See also the [project page](https://sites.google.com/view/rigpi).

RigPI fits mass, center of mass, inertia, and Coulomb friction to a measured pose
trajectory and applied forces and torques. A VLM provides the initial estimate
and parameter bounds; a differentiable physics rollout refines the estimate.
The code supports a free rigid body and a moving link in an articulated body.

This repository includes example priors and simulated trajectories. It does not
run a VLM or include the robot data from the paper.

## Installation

```bash
git clone https://github.com/Eigenoperator/RigPI.git
cd RigPI
python -m pip install -e ".[viz]"
```

Requires Python 3.10+. Newton 1.6.1 and Warp 1.18.0 are pinned because the
rollouts and gradients were checked with these versions. The examples switch to
CPU when Warp cannot use the local CUDA driver. Matplotlib is only needed for
the plot; install with `pip install -e .` if you do not need it.

## Examples

The revolute-joint example uses
[`simple_revolute.urdf`](https://github.com/Eigenoperator/RigPI/blob/main/assets/simple_revolute.urdf):

```bash
python examples/identify_revolute.py --trajectory-output outputs/revolute_trajectory.npz
python examples/plot_trajectory.py outputs/revolute_trajectory.npz
```

The fit is written to `outputs/revolute_demo.json`. The `.npz` file contains the
observed and fitted poses; the plot is saved as `outputs/revolute_trajectory.png`.
The figure below follows a point fixed to the moving link.

![Revolute-joint simulation and fit](https://raw.githubusercontent.com/Eigenoperator/RigPI/main/docs/revolute_trajectory.png)

The unconstrained revolute fit is useful for checking the full gradient path,
but its single-axis motion cannot determine all link parameters separately. To
validate mass and joint friction when the box geometry is known, use a separate
reversing-motion experiment:

```bash
python examples/validate_revolute_box.py --check
```

This check reads box dimensions and center from the URDF collision geometry,
ties inertia to mass for a uniform box, and optimizes only mass and friction.
It also evaluates a held-out torque trajectory. Its small errors validate that
restricted identification problem; they are not full mass, COM, and inertia
recovery from an unknown articulated object.

For the free-body example:

```bash
python examples/identify_free_body.py --trajectory-output outputs/free_body_trajectory.npz
```

Its fit is written to `outputs/free_body_demo.json`. Both examples generate
their observations in simulation and start from the sample priors in `configs/`.
The free-body demo varies force, torque, and wrench application point across
three axes; the corresponding config accepts constant 3-vectors or per-axis
sinusoids with `offset`, `amplitude`, `angular_frequency` (rad/s), and optional
`phase` (radians). Synthetic reports include errors against their known ground
truth, so a small trajectory loss is not mistaken for accurate parameter
identification. A single revolute-joint trajectory cannot fully constrain every
inertia component; its parameter errors should be read alongside its pose loss.

## VLM prior

The scripts read a prior from JSON with `--prior`. Copy
[`configs/vlm_prior_demo.json`](https://github.com/Eigenoperator/RigPI/blob/main/configs/vlm_prior_demo.json)
or
[`configs/vlm_prior_free_body_demo.json`](https://github.com/Eigenoperator/RigPI/blob/main/configs/vlm_prior_free_body_demo.json)
and replace the values with your model's estimate. The file has two objects:

| Key | Value |
| --- | --- |
| `initial.mass` | Mass in kg |
| `initial.com` | Center of mass in the body frame, 3 values in m |
| `initial.principal_inertia` | Principal moments about the center of mass, 3 values in kg m² |
| `initial.friction` | Dimensionless Coulomb coefficient |
| `initial.inertia_rotation_xyzw` | Optional rotation of the principal-inertia frame; identity if omitted |
| `bounds.mass`, `bounds.friction` | `[lower, upper]` |
| `bounds.com_lower`, `bounds.com_upper` | 3 values each, in the body frame |
| `bounds.inertia_lower`, `bounds.inertia_upper` | Symmetric 3 × 3 matrices of element-wise limits |

For a direct Python integration, implement
[`VLMPriorProvider.estimate(rgb_path, depth_path)`](https://github.com/Eigenoperator/RigPI/blob/main/src/rigpi/prior.py). It
returns a `VLMPrior`; `to_optimization_inputs()` converts it to the initial
parameters and bounds used by the optimizer. The example scripts read JSON
instead of calling the provider.

## Measured trajectories

`--trajectory` accepts an aligned `.npz` file with these arrays:

| Array | Shape | Units and frame |
| --- | --- | --- |
| `time` | `[N]` | Seconds; strictly increasing |
| `position` | `[N, 3]` | Moving body-frame origin in world coordinates, m |
| `quaternion_xyzw` | `[N, 4]` | Body orientation in world coordinates; unit quaternion |
| `force` | `[N-1, 3]` | World-frame force, N |
| `torque` | `[N-1, 3]` | World-frame torque about `application_point`, N m |
| `application_point` | `[N-1, 3]` | World-frame point where the wrench acts, m |
| `linear_velocity`, `angular_velocity` | `[N, 3]` | Optional; row 0 sets the free body's initial velocity |

Wrench row `k` acts over `[time[k], time[k+1]]`. The application point is not
necessarily the center of mass. Align the pose and force/torque streams and
convert them to the stated frames before saving the file.
`InteractionTrajectory.from_npz()` checks the shapes, timestamps, quaternion
norms, and finite values.

Copy the relevant demo config and set the parameters of your setup:

- **Articulated body:** full initial joint position and velocity arrays, joint
  type, normal load, and effective friction radius for a revolute joint. Supply
  your URDF and select the moving body and joint degree of freedom on the
  command line.
- **Free body:** gravity, normal load, and contact point in body coordinates.
  The first observed pose sets the initial pose. If the `.npz` file has no
  velocity arrays, set the initial velocities in the config.

```bash
python examples/identify_revolute.py --trajectory data/aligned.npz --prior data/prior.json --config data/revolute.json --urdf data/object.urdf --body-index 1 --joint-dof-index 0
python examples/identify_free_body.py --trajectory data/aligned.npz --prior data/prior.json --config data/free_body.json
```

The friction load is `mu × normal_force × radius` for a revolute joint and
`mu × normal_force` for a prismatic joint or free-body contact. Use loads and
contact geometry measured or estimated for your setup. Older processed JSON
files can be read with `rigpi.legacy.load_processed_json()`.

## Tests

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest -q
python examples/featherstone_smoke.py
```

CI also runs both simulation examples and the known-box validation, reloads the
example trajectories through the measured-data path, and checks the wheel and
source distribution. The paper's robot measurements, baseline code, and
individual trials are not distributed here.

## Citation

```bibtex
@article{he2026rigpi,
  title   = {RigPI: Dynamic Parameter Identification of Rigid Body via VLM-Seeded Differentiable Simulation},
  author  = {He, Xincheng and Zhang, Rongrong and Jiang, Wei and Xu, Wenqiang},
  journal = {arXiv preprint arXiv:2606.25212},
  year    = {2026}
}
```

## License

[Apache License 2.0](https://github.com/Eigenoperator/RigPI/blob/main/LICENSE).
