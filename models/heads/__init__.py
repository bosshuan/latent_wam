"""Prediction heads for the unified DiT (doc §2.4 / §2.5)."""

from models.heads.action_flow import ActionFlowHead
from models.heads.latent_flow import VJEPALatentFlowHead
from models.heads.value import ValueHead

__all__ = ["VJEPALatentFlowHead", "ActionFlowHead", "ValueHead"]
