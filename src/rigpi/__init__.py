"""RigPI differentiable rigid-body parameter identification."""

__version__ = "0.1.0"

from .data import InteractionTrajectory
from .losses import pose_loss, quadratic_bound_penalty, total_loss
from .optimizer import IdentificationResult, RigPIOptimizer
from .parameters import PhysicalParameters, RawPhysicalParameters
from .prior import VLMPrior, VLMPriorProvider
from .scheduler import ScaleAwareLearningRate

__all__ = [
    "IdentificationResult",
    "InteractionTrajectory",
    "PhysicalParameters",
    "RawPhysicalParameters",
    "RigPIOptimizer",
    "ScaleAwareLearningRate",
    "VLMPrior",
    "VLMPriorProvider",
    "__version__",
    "pose_loss",
    "quadratic_bound_penalty",
    "total_loss",
]
