"""Causal backbone over the AR sequence.

Consumes the ``(B, T'*G, d)`` sequence produced by
:func:`neurocast.tokenizer.perceiver.ar_flatten` and produces, at each position,
a representation of everything up to and including that position. The head then
predicts position ``i+1`` from the output at ``i``, which realises

    p(x_t^(g) | x_t^(<g), x_<t)

Layer stack
-----------
Repeating 6-layer block, four times for the 90M rung::

    [SWA, SWA, SWA, SWA, GLOBAL, XATTN->prompt]

* **SWA** -- causal sliding-window attention, window 512 tokens. At 62.5
  tokens/s and G=5 groups that is ~1.6 s of context per layer; stacking deepens
  the effective receptive field multiplicatively.
* **GLOBAL** -- full causal attention, one layer per block. The hybrid-attention
  literature finds roughly one global layer per 6-7 local layers is enough
  (MiniMax-01 uses 1 softmax block per 7 linear blocks).
* **XATTN** -- cross-attention to the subject prompt, KV-only, zero-init output
  projection so the model starts prompt-free and learns to use it.

Substitution from the design, stated plainly
--------------------------------------------
The design called for Mamba-2 selective state-space blocks in the SSM slots, for
linear-time scaling to the full 2.5-minute context. This implements those slots
with **full causal attention** instead, and exposes :class:`SSMBlock` as the
drop-in interface.

The reason is honesty about what can be verified here: a hand-rolled chunked
associative scan is easy to get subtly wrong in ways that leak future
information, and a leak in *that* direction fabricates exactly the result this
project is trying to measure. Attention is O(N^2) but its causality is trivially
testable, and the test in ``scripts/validate_backbone.py`` does test it. Swap in
``mamba-ssm`` once the causality harness can be pointed at it.

At the Phase-1 bake-off scale (16 s windows, ~1,600 positions) quadratic
attention is entirely affordable. The substitution only starts to bite at the
Phase-2 long-context stage.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "RotaryEmbedding",
    "CausalSelfAttention",
    "PromptCrossAttention",
    "SSMBlock",
    "Backbone",
    "BackboneConfig",
]


class RotaryEmbedding(nn.Module):
    """Rotary position embedding, applied to queries and keys."""

    def __init__(self, dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError(f"rotary dim must be even, got {dim}")
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(n, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[None, None], emb.sin()[None, None]

    @staticmethod
    def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin


class CausalSelfAttention(nn.Module):
    """Causal attention, optionally restricted to a sliding window.

    ``window=None`` gives full causal attention. Otherwise position ``i``
    attends to ``[i - window + 1, i]``.
    """

    def __init__(self, d_model: int, n_heads: int, window: int | None = None) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window = window
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.d_head)

    def _mask(self, n: int, device: torch.device) -> torch.Tensor:
        """True where attention is blocked."""
        i = torch.arange(n, device=device)[:, None]
        j = torch.arange(n, device=device)[None, :]
        blocked = j > i  # strictly future
        if self.window is not None:
            blocked = blocked | (j < i - self.window + 1)
        return blocked

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q, k, v = (
            t.view(b, n, self.n_heads, self.d_head).transpose(1, 2) for t in (q, k, v)
        )
        cos, sin = self.rope(n, x.device)
        q = RotaryEmbedding.rotate(q, cos, sin)
        k = RotaryEmbedding.rotate(k, cos, sin)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=~self._mask(n, x.device))
        out = out.transpose(1, 2).reshape(b, n, d)
        return x + self.proj(out)


class PromptCrossAttention(nn.Module):
    """Cross-attention from the query sequence to the subject prompt.

    The prompt is **keys and values only** -- it never attends back. Two
    consequences, both load-bearing:

    1. The prompt encoding is computed once per subject/session and cached, so
       scoring 50 word candidates re-uses one prompt. Subject conditioning costs
       essentially nothing at inference.
    2. Query information cannot leak into the prompt representation. That
       leakage class is eliminated by construction rather than by discipline.

    The output projection is zero-initialised, so the model begins prompt-free
    and learns to use context. This makes "how much does the prompt help?" a
    clean trajectory rather than a confound.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor, prompt: torch.Tensor | None) -> torch.Tensor:
        if prompt is None:
            return x
        b, n, d = x.shape
        m = prompt.shape[1]
        q = self.to_q(self.norm_q(x)).view(b, n, self.n_heads, self.d_head).transpose(1, 2)
        k, v = self.to_kv(self.norm_kv(prompt)).chunk(2, dim=-1)
        k, v = (
            t.view(b, m, self.n_heads, self.d_head).transpose(1, 2) for t in (k, v)
        )
        out = F.scaled_dot_product_attention(q, k, v)
        return x + self.proj(out.transpose(1, 2).reshape(b, n, d))


class SSMBlock(nn.Module):
    """Interface placeholder for a selective state-space block.

    Deliberately not implemented. Point ``mamba-ssm`` here once the causality
    test in ``scripts/validate_backbone.py`` can be run against it -- a scan
    that leaks one step of future information would manufacture precisely the
    forecasting result this project exists to measure honestly.
    """

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        raise NotImplementedError(
            "SSMBlock is an interface stub. Use CausalSelfAttention for now, or "
            "wire in mamba-ssm and run validate_backbone.py against it first."
        )


class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, expansion * d_model),
            nn.GELU(),
            nn.Linear(expansion * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class BackboneConfig:
    """Model-size rungs from the token-budget analysis.

    The corpus is ~1.35e8 sequence positions, so Chinchilla-optimal is ~27M
    parameters at four epochs. MEG-XL is 20M and is SOTA; that is not a
    coincidence. Rungs exist to *measure* the scaling curve, not because the
    large end is expected to win.
    """

    RUNGS: dict[str, dict[str, int]] = {
        "6m": {"d_model": 384, "n_layers": 12, "n_heads": 6},
        "25m": {"d_model": 512, "n_layers": 18, "n_heads": 8},
        "90m": {"d_model": 768, "n_layers": 24, "n_heads": 12},
        "320m": {"d_model": 1024, "n_layers": 36, "n_heads": 16},
    }

    def __init__(
        self,
        rung: str = "25m",
        *,
        window: int = 512,
        global_every: int = 5,
        prompt_every: int = 6,
    ) -> None:
        if rung not in self.RUNGS:
            raise ValueError(f"unknown rung {rung!r}; choose from {list(self.RUNGS)}")
        self.rung = rung
        self.__dict__.update(self.RUNGS[rung])
        self.window = window
        self.global_every = global_every
        self.prompt_every = prompt_every


class Backbone(nn.Module):
    """Causal stack over the flattened AR sequence.

    Every sublayer is causal or prompt-directed, so the whole stack is causal.
    That invariant is tested by perturbation, not assumed.
    """

    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, h = cfg.d_model, cfg.n_heads

        self.layers = nn.ModuleList()
        self.kinds: list[str] = []
        for i in range(cfg.n_layers):
            if (i + 1) % cfg.prompt_every == 0:
                self.layers.append(PromptCrossAttention(d, h))
                self.kinds.append("xattn")
            elif (i + 1) % cfg.global_every == 0:
                self.layers.append(CausalSelfAttention(d, h, window=None))
                self.kinds.append("global")
            else:
                self.layers.append(CausalSelfAttention(d, h, window=cfg.window))
                self.kinds.append("swa")
            self.layers.append(FeedForward(d))
            self.kinds.append("ffn")

        self.norm_out = nn.LayerNorm(d)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            # Leave the zero-init prompt projection alone.
            if not (m.weight.numel() and torch.count_nonzero(m.weight) == 0):
                nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, prompt: torch.Tensor | None = None) -> torch.Tensor:
        """``(B, N, d)`` -> ``(B, N, d)``, causal in ``N``."""
        for layer, kind in zip(self.layers, self.kinds):
            x = layer(x, prompt) if kind == "xattn" else layer(x)
        return self.norm_out(x)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        counts: dict[str, int] = {}
        for k in self.kinds:
            counts[k] = counts.get(k, 0) + 1
        layout = " ".join(f"{k}x{v}" for k, v in counts.items())
        return (
            f"Backbone[{self.cfg.rung}] d={self.cfg.d_model} "
            f"layers={self.cfg.n_layers} heads={self.cfg.n_heads} "
            f"window={self.cfg.window}\n  {layout}  "
            f"{self.n_parameters():,} params"
        )
