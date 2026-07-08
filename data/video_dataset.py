"""Actionless video dataset.

Emits ``TrajectorySample`` with ``action_valid=False`` and **no action tensor**
(CLAUDE.md §2.3). Clips without a caption use the null-text sentinel, unifying
"no caption" with "no action" for CFG.

Frame decoding is injected (``frame_loader``) so the unit tests run on CPU with a
synthetic loader; the server path plugs in a decord/torchvision decoder. This
class deliberately does not import any video backend at module load.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
from torch.utils.data import Dataset

from data.collate import TrajectorySample
from data.registry import INVALID_EMBODIMENT_ID
from data.schemas import NULL_TEXT


@dataclass
class VideoClipSpec:
    """Locator for one clip window (resolved by the frame loader)."""

    path: str
    frame_start: int
    num_frames: int  # raw frames = (history+future) * tubelet
    fps: float
    text: str = NULL_TEXT
    view_id: int = 0


# loader: spec -> pixels [T, 3, H, W]
FrameLoader = Callable[[VideoClipSpec], torch.Tensor]


class VideoDataset(Dataset):
    def __init__(
        self,
        clips: Sequence[VideoClipSpec],
        frame_loader: FrameLoader,
        token_grid: tuple[int, int, int],
        embodiment_id: int = INVALID_EMBODIMENT_ID,
        action_schema_id: int = -1,
    ) -> None:
        """
        Args:
            clips: clip windows to sample.
            frame_loader: decodes a spec into pixels [T, 3, H, W].
            token_grid: (T_tok, grid_h, grid_w) the encoder will produce.
            embodiment_id: defaults to INVALID_EMBODIMENT_ID — video never
                selects a per-embodiment action/state adapter (its action branch
                is omitted entirely). The modality/view embedding is keyed by
                ``view_id`` / a modality tag, not by embodiment_id, so -1 is safe
                and prevents video from borrowing a real embodiment's weights.
            action_schema_id: -1 marks "no schema" (video). Never batched with
                the counterfactual permutation set.
        """
        self.clips = list(clips)
        self.frame_loader = frame_loader
        self.token_grid = token_grid
        self.embodiment_id = embodiment_id
        self.action_schema_id = action_schema_id

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> TrajectorySample:
        spec = self.clips[idx]
        pixels = self.frame_loader(spec)
        if pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError(
                f"frame_loader must return [T,3,H,W]; got {tuple(pixels.shape)}"
            )
        text = spec.text if spec.text else NULL_TEXT
        return TrajectorySample(
            pixels=pixels,
            token_grid=self.token_grid,
            action_valid=False,  # video: never a real action
            embodiment_id=self.embodiment_id,
            action_schema_id=self.action_schema_id,
            view_id=spec.view_id,
            fps=spec.fps,
            text=text,
            actions=None,  # explicit: no pseudo-action
            action_step_mask=None,
            proprio=None,
        )
