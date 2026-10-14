from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def central_difference(
    function: Callable[[Tensor], Tensor],
    value: Tensor,
    *,
    epsilon: float = 1.0e-4,
) -> Tensor:
    """Compute a central finite-difference gradient for a scalar function."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    flat = value.detach().clone().reshape(-1)
    gradient = torch.zeros_like(flat)
    for index in range(flat.numel()):
        positive = flat.clone()
        negative = flat.clone()
        positive[index] += epsilon
        negative[index] -= epsilon
        gradient[index] = (
            function(positive.reshape_as(value)) - function(negative.reshape_as(value))
        ) / (2.0 * epsilon)
    return gradient.reshape_as(value)


def compare_autograd_to_finite_difference(
    function: Callable[[Tensor], Tensor],
    value: Tensor,
    *,
    epsilon: float = 1.0e-4,
    absolute_tolerance: float = 1.0e-4,
    relative_tolerance: float = 1.0e-3,
) -> tuple[Tensor, Tensor]:
    """Raise if an analytic gradient disagrees with finite differences."""
    variable = value.detach().clone().requires_grad_(True)
    output = function(variable)
    (analytic,) = torch.autograd.grad(output, variable)
    numeric = central_difference(function, value, epsilon=epsilon)
    if not torch.allclose(analytic, numeric, atol=absolute_tolerance, rtol=relative_tolerance):
        raise AssertionError(f"gradient mismatch\nanalytic={analytic}\nnumeric={numeric}")
    return analytic, numeric
