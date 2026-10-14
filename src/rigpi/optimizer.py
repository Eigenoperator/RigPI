from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor

from .bounds import ParameterBounds
from .parameters import PhysicalParameters, RawPhysicalParameters


@dataclass(frozen=True)
class OptimizationRecord:
    iteration: int
    pose_loss: float
    bound_penalty: float
    total_loss: float
    parameters: PhysicalParameters


@dataclass(frozen=True)
class IdentificationResult:
    parameters: PhysicalParameters
    history: tuple[OptimizationRecord, ...]
    converged: bool


RolloutLoss = Callable[[PhysicalParameters], Tensor]


class RigPIOptimizer:
    """Gradient descent with updates bounded in physical parameter units."""

    def __init__(
        self,
        *,
        bounds: ParameterBounds,
        penalty_weight: float = 0.1,
        relative_step: float = 1.0e-2,
        iterations: int = 200,
        gradient_tolerance: float = 1.0e-9,
        loss_tolerance: float = 1.0e-10,
    ) -> None:
        if not isfinite(penalty_weight) or penalty_weight < 0:
            raise ValueError("penalty_weight must be finite and non-negative")
        if not isfinite(relative_step) or relative_step <= 0:
            raise ValueError("relative_step must be finite and positive")
        if iterations < 1:
            raise ValueError("iterations must be positive")
        if not isfinite(gradient_tolerance) or gradient_tolerance < 0:
            raise ValueError("gradient_tolerance must be finite and non-negative")
        if not isfinite(loss_tolerance) or loss_tolerance < 0:
            raise ValueError("loss_tolerance must be finite and non-negative")
        self.bounds = bounds
        self.penalty_weight = penalty_weight
        self.relative_step = relative_step
        self.iterations = iterations
        self.gradient_tolerance = gradient_tolerance
        self.loss_tolerance = loss_tolerance

    @staticmethod
    def _snapshot(parameters: PhysicalParameters) -> PhysicalParameters:
        return PhysicalParameters(
            mass=parameters.mass.detach().clone(),
            com=parameters.com.detach().clone(),
            inertia=parameters.inertia.detach().clone(),
            friction=parameters.friction.detach().clone(),
        )

    def identify(
        self,
        raw: RawPhysicalParameters,
        rollout_loss: RolloutLoss,
    ) -> IdentificationResult:
        tensors = raw.tensors()
        for tensor in tensors:
            tensor.requires_grad_(True)

        lower, upper = self.bounds.vector_bounds()
        initial_vector = raw.decode().as_vector().detach()
        if lower.shape != initial_vector.shape or upper.shape != initial_vector.shape:
            raise ValueError("parameter bounds must match the physical parameter vector")
        if not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
            raise ValueError("parameter bounds must be finite")
        if torch.any(lower > upper):
            raise ValueError("each lower parameter bound must not exceed its upper bound")
        scale = torch.maximum(
            torch.maximum(initial_vector.abs(), upper - lower),
            torch.full_like(initial_vector, 1.0e-8),
        )

        def physical_vector(*values: Tensor) -> Tensor:
            return RawPhysicalParameters(*values, epsilon=raw.epsilon).decode().as_vector()

        history: list[OptimizationRecord] = []
        previous_loss: float | None = None
        converged = False

        for iteration in range(self.iterations):
            for tensor in tensors:
                tensor.grad = None

            parameters = raw.decode()
            pose = rollout_loss(parameters)
            penalty = self.bounds.penalty(parameters)
            loss = pose + self.penalty_weight * penalty
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError(f"non-finite or non-scalar objective at iteration {iteration}")
            loss.backward()

            gradients = [tensor.grad for tensor in tensors if tensor.grad is not None]
            if not gradients or any(not torch.isfinite(grad).all() for grad in gradients):
                raise ValueError(
                    f"missing or non-finite parameter gradient at iteration {iteration}"
                )
            gradient_norm = torch.sqrt(sum(torch.sum(grad.detach().square()) for grad in gradients))
            loss_value = float(loss.detach())
            history.append(
                OptimizationRecord(
                    iteration=iteration,
                    pose_loss=float(pose.detach()),
                    bound_penalty=float(penalty.detach()),
                    total_loss=loss_value,
                    parameters=self._snapshot(parameters),
                )
            )

            if float(gradient_norm) <= self.gradient_tolerance:
                converged = True
                break
            if previous_loss is not None and abs(previous_loss - loss_value) <= self.loss_tolerance:
                converged = True
                break

            jacobians = torch.autograd.functional.jacobian(
                physical_vector, tuple(tensor.detach() for tensor in tensors)
            )
            proposed: list[Tensor] = []
            for tensor, jacobian in zip(tensors, jacobians, strict=True):
                if tensor.grad is None:
                    proposed.append(torch.zeros_like(tensor))
                    continue
                shape = (-1,) + (1,) * tensor.ndim
                sensitivity = (jacobian.abs() / scale.reshape(shape)).amax(dim=0)
                raw_step = -self.relative_step * tensor.grad.sign() / sensitivity.clamp_min(1e-12)
                proposed.append(raw_step.clamp(min=-0.5, max=0.5))

            # Simultaneous changes can accumulate, and softplus is nonlinear.
            # Backtrack until the physical step is small enough and the actual
            # rollout objective decreases. The latter also lets the steps
            # shrink as the fit approaches a minimum.
            with torch.no_grad():
                factor = 1.0
                current = parameters.as_vector().detach()
                accepted = False
                for _ in range(24):
                    candidate_raw = RawPhysicalParameters(
                        *(
                            tensor + factor * step
                            for tensor, step in zip(tensors, proposed, strict=True)
                        ),
                        epsilon=raw.epsilon,
                    )
                    candidate_parameters = candidate_raw.decode()
                    candidate = candidate_parameters.as_vector()
                    if not torch.isfinite(candidate).all():
                        raise ValueError(f"non-finite parameter update at iteration {iteration}")
                    ratio = float(
                        ((candidate - current).abs() / (self.relative_step * scale)).max()
                    )
                    if ratio > 1.000001:
                        factor *= min(0.5, 0.95 / ratio)
                        continue
                    candidate_loss = rollout_loss(candidate_parameters) + self.penalty_weight * (
                        self.bounds.penalty(candidate_parameters)
                    )
                    if (
                        candidate_loss.ndim == 0
                        and torch.isfinite(candidate_loss)
                        and float(candidate_loss) < loss_value
                    ):
                        for tensor, step in zip(tensors, proposed, strict=True):
                            tensor += factor * step
                        accepted = True
                        break
                    factor *= 0.5
            if not accepted:
                break
            previous_loss = loss_value

        if history and history[-1].iteration == self.iterations - 1 and not converged:
            final_parameters = raw.decode()
            final_pose = rollout_loss(final_parameters)
            final_penalty = self.bounds.penalty(final_parameters)
            if (
                final_pose.ndim != 0
                or final_penalty.ndim != 0
                or not torch.isfinite(final_pose)
                or not torch.isfinite(final_penalty)
            ):
                raise ValueError("non-finite objective after the final optimizer update")
            history.append(
                OptimizationRecord(
                    iteration=self.iterations,
                    pose_loss=float(final_pose.detach()),
                    bound_penalty=float(final_penalty.detach()),
                    total_loss=float((final_pose + self.penalty_weight * final_penalty).detach()),
                    parameters=self._snapshot(final_parameters),
                )
            )
        return IdentificationResult(
            parameters=self._snapshot(raw.decode()),
            history=tuple(history),
            converged=converged,
        )
