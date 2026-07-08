"""Preferred entry point for VJ-RAE training.

``train_codec.py`` remains as a backward-compatible milestone name. New scripts
should use this module.
"""

from train.train_codec import (
    SyntheticVJRAELoader,
    build_codec_from_encoder,
    build_dataloaders,
    codec_train_step,
    fit_normalizer,
    main,
)

__all__ = [
    "SyntheticVJRAELoader",
    "build_codec_from_encoder",
    "build_dataloaders",
    "codec_train_step",
    "fit_normalizer",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    main()
