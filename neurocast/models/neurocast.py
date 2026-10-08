"""The assembled model: montage -> tokens -> causal representation.

Everything from :mod:`neurocast.tokenizer` and
:mod:`neurocast.models.backbone` wired into one module, so the bake-off can hold
it fixed and vary only the objective.

    raw (B, C, T)
      -> TypedPatchEmbed         one projection per sensor family
       + SensorEmbedding         frozen Fourier geometry features
      -> SensorPerceiver         quadrant-confined pooling, G tokens per patch
      -> ar_flatten              [g0(t0) .. g4(t0), g0(t1) ..]
      -> Backbone                causal in the flattened sequence
      -> (B, T'*G, d)

The AR factorisation is exact because the perceiver confines each token to one
partition cell -- see :mod:`neurocast.tokenizer.perceiver`. Nothing in this file
may break that confinement.

Masked arms (``causal=False``) additionally get a learned mask token. Masked
positions are overwritten with it *after* the perceiver, and because each
position is a function of one (patch, quadrant) cell alone, the raw samples
behind a masked position then reach no part of the network. That is what makes
masked reconstruction a prediction task rather than an autoencoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..tokenizer.descriptor import Montage, Quadrant
from ..tokenizer.embedding import SensorEmbedding
from ..tokenizer.patch import PATCH_SAMPLES, TypedPatchEmbed
from ..tokenizer.perceiver import SensorPerceiver, ar_flatten, ar_unflatten
from .backbone import Backbone, BackboneConfig

__all__ = ["NeuroCast"]


class NeuroCast(nn.Module):
    """Front-end plus causal backbone. Objective-agnostic by construction."""

    def __init__(
        self,
        montage: Montage,
        *,
        rung: str = "6m",
        window: int = 512,
        latents_per_group: int = 8,
        n_cross_layers: int = 1,
        stim_every: int | None = None,
        d_stimulus: int = 64,
        stim_mode: str = "decode",
        causal: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        cfg = BackboneConfig(rung, window=window, stim_every=stim_every,
                             d_stimulus=d_stimulus, stim_mode=stim_mode, causal=causal)
        d = cfg.d_model
        self.cfg = cfg
        self.d_model = d
        self.n_groups = Quadrant.n_all()
        self.causal = bool(causal)

        self.sensor = SensorEmbedding(d)
        self.patch = TypedPatchEmbed(d)
        self.perceiver = SensorPerceiver(
            d, latents_per_group=latents_per_group, n_cross_layers=n_cross_layers
        )
        self.backbone = Backbone(cfg)
        # Only the bidirectional (masked) arms own a mask token, so every causal
        # model keeps exactly the parameter set it had before masking existed.
        self.mask_token = None if causal else nn.Parameter(torch.randn(d) * 0.02)

        self.register_buffer("type_id", torch.from_numpy(montage.type_ids))
        self.register_buffer("quadrant", torch.from_numpy(montage.quadrants))
        self._montage_tensors = SensorEmbedding.montage_tensors(montage, device)
        self.montage_name = montage.name
        self.n_channels = len(montage)

    def retarget(self, montage: Montage, device: torch.device | str = "cpu") -> None:
        """Point the model at a different montage. Adds no parameters.

        This is H5's mechanism in one method: channels are identified by physical
        geometry, so a new array needs no new weights and no identification
        procedure.
        """
        self.type_id = torch.from_numpy(montage.type_ids).to(device)
        self.quadrant = torch.from_numpy(montage.quadrants).to(device)
        self._montage_tensors = SensorEmbedding.montage_tensors(montage, device)
        self.montage_name = montage.name
        self.n_channels = len(montage)

    def tokens(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """``(B, C, T)`` -> ``(B, T'*G, d)`` group tokens, pre-backbone.

        ``mask`` is a ``(B, T'*G)`` bool tensor, True where the position is
        hidden and replaced by the mask token. Bidirectional models only.
        """
        s = self.sensor(self._montage_tensors)
        tok = self.patch(x, self.type_id) + s[None, :, None, :]
        seq = ar_flatten(self.perceiver(tok, self.quadrant))
        if mask is None:
            return seq
        if self.mask_token is None:
            raise ValueError(
                "input masking is for the bidirectional arms (causal=False); a "
                "causal model already hides the future and has no mask token"
            )
        if mask.shape != seq.shape[:2]:
            raise ValueError(f"mask must be {tuple(seq.shape[:2])}, got {tuple(mask.shape)}")
        return torch.where(mask[..., None], self.mask_token.to(seq.dtype), seq)

    def forward(
        self,
        x: torch.Tensor,
        prompt: torch.Tensor | None = None,
        stimulus: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``(B, C, T)`` -> ``(B, T'*G, d)`` representations, causal unless built otherwise."""
        return self.backbone(self.tokens(x, mask), prompt=prompt, stimulus=stimulus)

    def sensor_embedding(self) -> torch.Tensor:
        """``(C, d)`` embedding of the current montage. For nuisance-token rows."""
        return self.sensor(self._montage_tensors)

    def pooled(self, x: torch.Tensor, prompt: torch.Tensor | None = None) -> torch.Tensor:
        """Mean-pooled representation, for frozen probes and FMScope.

        Pooling over the whole sequence is what the identity audit consumes: it
        is the closest analogue to the frozen-embedding protocol the Identity
        Trap paper uses.
        """
        return self.forward(x, prompt).mean(dim=1)

    def groups(self, x: torch.Tensor, prompt: torch.Tensor | None = None) -> torch.Tensor:
        """``(B, T', G, d)`` -- the unflattened view, for per-quadrant analysis."""
        return ar_unflatten(self.forward(x, prompt), self.n_groups)

    @property
    def patch_samples(self) -> int:
        return PATCH_SAMPLES

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        mode = "" if self.causal else " bidirectional"
        return (
            f"NeuroCast[{self.cfg.rung}]{mode} montage={self.montage_name} "
            f"({self.n_channels} ch) d={self.d_model} "
            f"groups={self.n_groups}  {self.n_parameters():,} params"
        )
