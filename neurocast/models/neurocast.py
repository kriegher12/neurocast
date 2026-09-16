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
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        cfg = BackboneConfig(rung, window=window)
        d = cfg.d_model
        self.cfg = cfg
        self.d_model = d
        self.n_groups = Quadrant.n_all()

        self.sensor = SensorEmbedding(d)
        self.patch = TypedPatchEmbed(d)
        self.perceiver = SensorPerceiver(
            d, latents_per_group=latents_per_group, n_cross_layers=n_cross_layers
        )
        self.backbone = Backbone(cfg)

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

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, T)`` -> ``(B, T'*G, d)`` group tokens, pre-backbone."""
        s = self.sensor(self._montage_tensors)
        tok = self.patch(x, self.type_id) + s[None, :, None, :]
        return ar_flatten(self.perceiver(tok, self.quadrant))

    def forward(
        self, x: torch.Tensor, prompt: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``(B, C, T)`` -> ``(B, T'*G, d)`` causal representations."""
        return self.backbone(self.tokens(x), prompt=prompt)

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
        return (
            f"NeuroCast[{self.cfg.rung}] montage={self.montage_name} "
            f"({self.n_channels} ch) d={self.d_model} "
            f"groups={self.n_groups}  {self.n_parameters():,} params"
        )
