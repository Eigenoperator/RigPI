import numpy as np
import warp as wp

from rigpi.backends.newton_articulated import _apply_smooth_joint_friction


def test_warp_joint_friction_coefficient_gradient():
    device = wp.get_preferred_device()
    joint_qd = wp.array([0.2], dtype=wp.float32, device=device)
    joint_f = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    friction = wp.array([0.4], dtype=wp.float32, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(
            _apply_smooth_joint_friction,
            dim=1,
            inputs=[joint_qd, joint_f, friction, 0, 3.0, 1.0e-3],
            device=device,
        )
    joint_f.grad.assign([1.0])
    tape.backward()

    assert friction.grad is not None
    assert np.isfinite(friction.grad.numpy()).all()
    assert abs(float(friction.grad.numpy()[0])) > 0.0
