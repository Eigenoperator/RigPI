"""Boundary between a user's vision model and RigPI identification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from .bounds import ParameterBounds
from .parameters import RawPhysicalParameters


@dataclass(frozen=True)
class VLMPrior:
    """Physical initial guess and element-wise bounds, all in SI units.

    ``com`` is in the moving link's body frame. Inertia is about that COM and
    expressed in the same frame. A VLM adapter may infer these values however
    it wishes, but must return this common representation.
    """

    initial: dict
    bounds: dict

    @classmethod
    def from_json(cls, path: str | Path) -> VLMPrior:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(initial=payload["initial"], bounds=payload["bounds"])

    def to_optimization_inputs(
        self, *, dtype: torch.dtype = torch.float32, device: str | torch.device = "cpu"
    ) -> tuple[RawPhysicalParameters, ParameterBounds]:
        def tensor(value):
            return torch.as_tensor(value, dtype=dtype, device=device)

        initial = self.initial
        raw = RawPhysicalParameters.from_physical(
            mass=tensor([initial["mass"]]),
            com=tensor(initial["com"]),
            principal_moments=tensor(initial["principal_inertia"]),
            friction=tensor([initial["friction"]]),
            inertia_rotation=tensor(initial.get("inertia_rotation_xyzw", [0, 0, 0, 1])),
        )
        bounds = self.bounds
        inertia_lower = tensor(bounds["inertia_lower"])
        inertia_upper = tensor(bounds["inertia_upper"])
        if inertia_lower.shape != (3, 3) or inertia_upper.shape != (3, 3):
            raise ValueError("inertia bounds must each be 3x3 matrices")
        if not torch.allclose(inertia_lower, inertia_lower.T) or not torch.allclose(
            inertia_upper, inertia_upper.T
        ):
            raise ValueError("inertia bounds must be symmetric")
        result = ParameterBounds(
            mass_lower=tensor([bounds["mass"][0]]),
            mass_upper=tensor([bounds["mass"][1]]),
            com_lower=tensor(bounds["com_lower"]),
            com_upper=tensor(bounds["com_upper"]),
            inertia_lower=inertia_lower,
            inertia_upper=inertia_upper,
            friction_lower=tensor([bounds["friction"][0]]),
            friction_upper=tensor([bounds["friction"][1]]),
        )
        physical = raw.decode().as_vector()
        lower, upper = result.vector_bounds()
        if not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
            raise ValueError("prior bounds must be finite")
        if physical.shape != lower.shape or not torch.all(lower <= upper):
            raise ValueError("prior bounds have invalid shapes or ordering")
        if not torch.all((physical >= lower) & (physical <= upper)):
            raise ValueError("VLM initial parameters must lie within the supplied bounds")
        return raw, result


class VLMPriorProvider(Protocol):
    """Implement this method to connect any image/VLM pipeline to RigPI."""

    def estimate(self, rgb_path: Path, depth_path: Path | None = None) -> VLMPrior: ...
