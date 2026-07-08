"""Preferred VJ-RAE import surface.

The old milestone code used "codec" for this module. Stage A now treats it as a
V-JEPA Representation Autoencoder: a frozen representation-space autoencoder that
maps pooled multi-layer V-JEPA features to the 384-d latent grid used by the flow
model. This file keeps the new name small and explicit while preserving old
checkpoint/test imports from ``latent_codec.py``.
"""

from models.latent_codec import (
    ActionDiscriminabilityProbe,
    CodecDecoder,
    CodecEncoder,
    FixedFeatureNormalizer,
    MultiLevelFusion,
    TokenReducer,
    VJEPALatentCodec,
    VJEPRepresentationAutoencoder,
    pooled_latent_delta,
)

__all__ = [
    "ActionDiscriminabilityProbe",
    "CodecDecoder",
    "CodecEncoder",
    "FixedFeatureNormalizer",
    "MultiLevelFusion",
    "TokenReducer",
    "VJEPALatentCodec",
    "VJEPRepresentationAutoencoder",
    "pooled_latent_delta",
]
