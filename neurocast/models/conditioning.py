"""Conditioning streams: the subject prompt (H3) and the stimulus (encoding/decoding).

Two ways information enters the model besides the brain signal itself. Both are
cross-attention, both start as a no-op (zero-initialised output projection), and
both are built so a leak is structurally impossible rather than merely avoided.

PromptEncoder -- a new subject enters as context, not as a gradient step
-----------------------------------------------------------------------
Compresses K minutes of a subject's own recording into a fixed memory the
backbone reads as keys and values:

* **Tier 2, summaries.** Each 8 s prompt chunk (500 group tokens) is pooled into
  ``S = 8`` summary tokens by learned queries -- about 62x compression, so 40
  minutes of MEG becomes 2,400 tokens.
* **Tier 3, nuisance.** Hand-computed measurement statistics (1/f exponent,
  gains, band power, bad channels; see :mod:`neurocast.adapt.nuisance`) pooled
  into a few tokens. These are the carriers FMScope names, gathered into one
  ablatable pathway -- which is what makes the prompt-swap intervention possible.
* Each chunk carries a staleness embedding (minutes before the query), because
  session drift is real and the model must know how old its evidence is.

The memory is computed once per subject and cached; nothing about the query
flows back into it.

StimulusCrossAttention -- what the brain is responding to
----------------------------------------------------------
Neural responses lag the stimulus by 50-300 ms, with the lag varying by feature
and region. Hard-coding one lag is wrong; free attention invites cheating. So a
query at patch ``t`` may attend only to stimulus patches ``s`` with
``t - s`` inside a **band**, and each head learns a bias over the lag bins inside
that band. That learned bias is a temporal response function -- directly
comparable to the classical TRF literature, and interpretable on its own.

Two modes, and the difference matters:

* ``decode`` -- band is ``[0, max]``. Only past and present stimulus. The stimulus
  is what is being inferred, so reading its future would be cheating.
* ``encode`` -- band may start below 0 (a small lookahead). The stimulus is
  exogenous: the audiobook's next 100 ms is fixed before the brain hears it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..tokenizer.embedding import GaussianFourierFeatures

__all__ = ["PromptEncoder", "StimulusCrossAttention"]


class _PoolQueries(nn.Module):
    """``n_queries`` learned queries cross-attending over a token set."""

    def __init__(self, d_model: int, n_queries: int, n_heads: int) -> None:
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model), nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b = tokens.shape[0]
        q = self.q.unsqueeze(0).expand(b, -1, -1)
        kv = self.norm(tokens)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        out = q + out
        return out + self.ff(out)


class PromptEncoder(nn.Module):
    """Turn a subject's own recording into a cached cross-attention memory.

    Parameters
    ----------
    d_model
        Backbone width.
    summary_tokens
        Tier-2 tokens per prompt chunk. 8 by default (~62x compression).
    nuisance_features
        Width of the per-channel nuisance matrix from
        :meth:`neurocast.adapt.nuisance.NuisanceDescriptors.to_matrix`.
    nuisance_tokens
        Tier-3 tokens.
    """

    SEG_SUMMARY = 0
    SEG_NUISANCE = 1

    def __init__(
        self,
        d_model: int,
        *,
        summary_tokens: int = 8,
        nuisance_features: int = 10,
        nuisance_tokens: int = 4,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.summary_tokens = summary_tokens
        self.nuisance_tokens = nuisance_tokens
        self.summarise = _PoolQueries(d_model, summary_tokens, n_heads)
        self.nuis_in = nn.Sequential(
            nn.Linear(nuisance_features, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.nuis_pool = _PoolQueries(d_model, nuisance_tokens, n_heads)
        self.segment = nn.Embedding(2, d_model)
        self.staleness = GaussianFourierFeatures(1, (0.05, 0.5, 5.0), 8, seed=0xD7)
        self.staleness_proj = nn.Linear(self.staleness.out_dim, d_model)

    def forward(
        self,
        chunk_tokens: torch.Tensor | None,
        *,
        minutes_before: torch.Tensor | None = None,
        nuisance: torch.Tensor | None = None,
        sensor_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Build the prompt memory.

        Parameters
        ----------
        chunk_tokens
            ``(B, n_chunks, L, d)`` backbone-input tokens for each prompt chunk,
            from :meth:`neurocast.models.neurocast.NeuroCast.tokens`. ``None``
            for a prompt-free episode.
        minutes_before
            ``(B, n_chunks)`` how long before the query each chunk was recorded.
        nuisance
            ``(B, C, F)`` per-channel nuisance matrix, or ``None`` to ablate Tier 3.
        sensor_embedding
            ``(C, d)``, added to nuisance rows so each statistic is tied to the
            physical sensor it describes.

        Returns
        -------
        ``(B, M, d)`` memory, or ``None`` if there is nothing to condition on.
        """
        parts = []
        if chunk_tokens is not None:
            b, n_chunks, length, d = chunk_tokens.shape
            s = self.summarise(chunk_tokens.reshape(b * n_chunks, length, d))
            s = s.reshape(b, n_chunks, self.summary_tokens, d)
            if minutes_before is not None:
                st = self.staleness_proj(self.staleness(minutes_before.unsqueeze(-1)))
                s = s + st.unsqueeze(2)
            s = s.reshape(b, n_chunks * self.summary_tokens, d)
            parts.append(s + self.segment.weight[self.SEG_SUMMARY])

        if nuisance is not None:
            rows = self.nuis_in(nuisance)
            if sensor_embedding is not None:
                rows = rows + sensor_embedding.unsqueeze(0)
            parts.append(self.nuis_pool(rows) + self.segment.weight[self.SEG_NUISANCE])

        if not parts:
            return None
        return torch.cat(parts, dim=1)

    def swap_nuisance(self, memory: torch.Tensor, donor: torch.Tensor) -> torch.Tensor:
        """Replace the Tier-3 tokens of ``memory`` with ``donor``'s.

        The prompt-swap intervention: hold the brain-derived summaries fixed and
        transplant another subject's measurement statistics. If decoding survives
        while an identity probe follows the transplanted tokens, identity is being
        *read from context* rather than stored in weights -- causal evidence for
        H2's mechanism, not a correlation.
        """
        k = self.nuisance_tokens
        return torch.cat([memory[:, :-k], donor[:, -k:]], dim=1)


class StimulusCrossAttention(nn.Module):
    """Banded cross-attention to a stimulus stream, with a learned lag profile.

    Parameters
    ----------
    lag_min, lag_max
        Allowed ``t - s`` in patches (64 ms each at the default patch size).
        ``decode`` mode requires ``lag_min >= 0``.
    n_groups
        Spatial groups per patch in the flattened brain sequence, so a sequence
        position can be mapped back to its patch index.
    """

    def __init__(
        self,
        d_model: int,
        d_stimulus: int,
        n_heads: int,
        *,
        lag_min: int = 0,
        lag_max: int = 12,
        n_groups: int = 5,
        mode: str = "decode",
    ) -> None:
        super().__init__()
        if mode not in ("decode", "encode"):
            raise ValueError(f"mode must be 'decode' or 'encode', got {mode!r}")
        if mode == "decode" and lag_min < 0:
            raise ValueError(
                "decode mode forbids lag_min < 0: the stimulus is the unknown being "
                "inferred, so reading its future would leak the answer"
            )
        if lag_max < lag_min:
            raise ValueError("lag_max must be >= lag_min")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.mode = mode
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.lag_min, self.lag_max = int(lag_min), int(lag_max)
        self.n_groups = int(n_groups)

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_stimulus)
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_kv = nn.Linear(d_stimulus, 2 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.proj.weight)
        # One bias per head per lag bin: the learned temporal response function.
        self.lag_bias = nn.Parameter(torch.zeros(n_heads, self.lag_max - self.lag_min + 1))

    def band(self, n_query: int, n_stim: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Lag index per (query, stimulus) pair, and the mask of pairs outside the band."""
        patch = torch.arange(n_query, device=device) // self.n_groups
        s = torch.arange(n_stim, device=device)
        lag = patch[:, None] - s[None, :]
        outside = (lag < self.lag_min) | (lag > self.lag_max)
        return (lag - self.lag_min).clamp(0, self.lag_bias.shape[1] - 1), outside

    def response_function(self) -> torch.Tensor:
        """``(n_heads, n_lags)`` softmax-normalised lag profile. The learned TRF."""
        return torch.softmax(self.lag_bias, dim=-1)

    def forward(self, x: torch.Tensor, stimulus: torch.Tensor | None) -> torch.Tensor:
        if stimulus is None:
            return x
        b, n, d = x.shape
        m = stimulus.shape[1]
        q = self.to_q(self.norm_q(x)).view(b, n, self.n_heads, self.d_head).transpose(1, 2)
        k, v = self.to_kv(self.norm_kv(stimulus)).chunk(2, dim=-1)
        k = k.view(b, m, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, m, self.n_heads, self.d_head).transpose(1, 2)

        lag_idx, outside = self.band(n, m, x.device)
        bias = self.lag_bias[:, lag_idx]                          # (heads, n, m)
        bias = bias.masked_fill(outside.unsqueeze(0), float("-inf"))

        # Positions whose entire band falls outside the stimulus (e.g. the first
        # patches in decode mode with lag_min > 0) would softmax over nothing.
        empty = outside.all(dim=1)
        if bool(empty.any()):
            bias = bias.masked_fill(empty[None, :, None], 0.0)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.unsqueeze(0))
        out = out.transpose(1, 2).reshape(b, n, d)
        if bool(empty.any()):
            out = out.masked_fill(empty[None, :, None], 0.0)
        return x + self.proj(out)
