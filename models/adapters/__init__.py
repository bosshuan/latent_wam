"""Input/conditioning adapters for the unified DiT (doc §2.4)."""

from models.adapters.action import (
    CategorySpecificLinear,
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
    SinusoidalPositionalEncoding,
)
from models.adapters.condition import ConditionAdapter
from models.adapters.latent import VJEPALatentInputAdapter
from models.adapters.state import StateAdapter

__all__ = [
    "CategorySpecificLinear",
    "CategorySpecificMLP",
    "MultiEmbodimentActionEncoder",
    "SinusoidalPositionalEncoding",
    "VJEPALatentInputAdapter",
    "StateAdapter",
    "ConditionAdapter",
]
