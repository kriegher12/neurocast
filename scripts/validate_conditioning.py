"""Verify the prompt and stimulus pathways.

Six claims:

1. **Stimulus causality, decode mode.** A stimulus change at patch s must leave
   every output before patch s untouched. In decode mode the stimulus is the
   unknown being inferred, so reading its future would leak the answer.
2. **Encode-mode lookahead is bounded** to exactly the configured lead.
3. **Brain causality survives** with both conditioning streams attached.
4. **The learned lag profile is a temporal response function**: trained on a
   response with a known delay, the lag bias must peak at that delay.
5. **Prompt memory is well-formed**: shapes, prompt-free episodes, and the
   nuisance swap touches only Tier 3.
6. **H3's mechanism, in miniature.** A model reads a subject's measurement
   statistics from the prompt -- on subjects it never saw in training. With the
   prompt it approaches the optimum; without it, it cannot; with another
   subject's prompt it does worse than with none. That last ordering is the
   point: the prompt is being *used*, not decoratively ignored.

Run:  .venv/Scripts/python.exe scripts/validate_conditioning.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurocast.models.backbone import Backbone, BackboneConfig, PromptCrossAttention  # noqa: E402
from neurocast.models.conditioning import PromptEncoder, StimulusCrossAttention  # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    return ok


def main() -> int:
    torch.manual_seed(0)
    ok = True
    G = 5

    print("=" * 80)
    print("1-3. Causality with conditioning streams attached")
    print("=" * 80)
    cfg = BackboneConfig("6m", window=16, global_every=5, prompt_every=6,
                         stim_every=4, d_stimulus=32, stim_lag_min=0, stim_lag_max=6,
                         n_groups=G)
    net = Backbone(cfg).eval()
    for layer, kind in zip(net.layers, net.kinds):
        if kind in ("stim", "xattn"):
            nn.init.normal_(layer.proj.weight, std=0.05)   # switch the pathways on
    print("  " + net.describe().replace("\n", "\n  "))

    b, n_patch = 2, 10
    x = torch.randn(b, n_patch * G, cfg.d_model)
    stim = torch.randn(b, n_patch, 32)
    prompt = torch.randn(b, 12, cfg.d_model)

    with torch.no_grad():
        base = net(x, prompt=prompt, stimulus=stim)
        s_star = 6
        stim2 = stim.clone()
        stim2[:, s_star] += 10.0 * torch.randn(b, 32)
        out = net(x, prompt=prompt, stimulus=stim2)
        per_patch = (base - out).abs().amax(dim=(0, 2)).reshape(n_patch, G).amax(1)
    before = float(per_patch[:s_star].max())
    at = float(per_patch[s_star])
    ok &= check(f"decode: stimulus change at patch {s_star} leaves patches <{s_star} untouched",
                before == 0.0, f"max |delta| {before:.3e}")
    ok &= check("  ...and does affect the patch where it lands", at > 1e-4, f"{at:.3e}")

    try:
        StimulusCrossAttention(64, 32, 4, lag_min=-2, mode="decode")
        ok &= check("decode mode refuses a negative lag (future stimulus)", False)
    except ValueError:
        ok &= check("decode mode refuses a negative lag (future stimulus)", True)

    enc = StimulusCrossAttention(64, 32, 4, lag_min=-2, lag_max=6, n_groups=1, mode="encode")
    nn.init.normal_(enc.proj.weight, std=0.05)
    with torch.no_grad():
        xe = torch.randn(1, 12, 64)
        se = torch.randn(1, 12, 32)
        e1 = enc(xe, se)
        se2 = se.clone()
        # Random, not a constant: LayerNorm on the stimulus removes a uniform
        # shift across dimensions, so "+= 10.0" would be an invisible change.
        se2[:, 8] += 10.0 * torch.randn(32)
        e2 = enc(xe, se2)
        d = (e1 - e2).abs().amax(dim=(0, 2))
    ok &= check("encode: lookahead reaches exactly lead=2 patches and no further",
                float(d[:6].max()) == 0.0 and float(d[6]) > 1e-4,
                f"patch 5 delta {float(d[5]):.1e}, patch 6 delta {float(d[6]):.1e}")

    with torch.no_grad():
        j = 23
        x2 = x.clone()
        x2[:, j] += 10.0 * torch.randn(b, cfg.d_model)
        o2 = net(x2, prompt=prompt, stimulus=stim)
        leak = float((base - o2).abs().amax(dim=(0, 2))[:j].max())
    ok &= check("brain causality holds with prompt AND stimulus attached", leak == 0.0,
                f"max |delta| before position {j}: {leak:.3e}")

    print()
    print("=" * 80)
    print("4. The learned lag profile recovers a known response delay")
    print("=" * 80)
    true_lag = 4
    torch.manual_seed(1)
    layer = StimulusCrossAttention(32, 16, 4, lag_min=0, lag_max=8, n_groups=1)
    readout = nn.Linear(32, 8)
    w_true = torch.randn(8, 16)
    opt = torch.optim.Adam(list(layer.parameters()) + list(readout.parameters()), lr=1e-2)
    for step in range(400):
        s = torch.randn(16, 24, 16)
        y = torch.zeros(16, 24, 8)
        y[:, true_lag:] = s[:, :-true_lag] @ w_true.T
        pred = readout(layer(torch.zeros(16, 24, 32), s))
        loss = ((pred - y) ** 2)[:, true_lag:].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    trf = layer.response_function().mean(0).detach()
    peak = int(trf.argmax())
    print("  learned TRF (lag bins, patches):  " +
          "  ".join(f"{i}:{v:.2f}" for i, v in enumerate(trf.tolist())))
    ok &= check("TRF peaks at the true response delay", peak == true_lag,
                f"peak at lag {peak}, truth {true_lag}, weight {float(trf[peak]):.2f}")

    print()
    print("=" * 80)
    print("5. Prompt memory")
    print("=" * 80)
    d_model, C, F_nuis = 64, 20, 10
    enc_p = PromptEncoder(d_model, summary_tokens=8, nuisance_features=F_nuis,
                          nuisance_tokens=4)
    chunks = torch.randn(2, 6, 40, d_model)
    nuis = torch.randn(2, C, F_nuis)
    mem = enc_p(chunks, minutes_before=torch.rand(2, 6) * 30, nuisance=nuis)
    ok &= check("memory = chunks x summary tokens + nuisance tokens",
                mem.shape == (2, 6 * 8 + 4, d_model), f"{tuple(mem.shape)}")
    ok &= check("a prompt-free episode yields no memory", enc_p(None) is None)
    donor = enc_p(chunks, nuisance=torch.randn(2, C, F_nuis))
    swapped = enc_p.swap_nuisance(mem, donor)
    ok &= check("nuisance swap replaces exactly Tier 3",
                torch.equal(swapped[:, :-4], mem[:, :-4])
                and torch.equal(swapped[:, -4:], donor[:, -4:]))

    print()
    print("=" * 80)
    print("6. H3 in miniature -- reading a NEW subject's statistics from context")
    print("=" * 80)
    torch.manual_seed(2)
    rng = np.random.default_rng(2)
    train_subj, test_subj = 12, 6
    log_gain = rng.normal(0.0, 0.6, train_subj + test_subj)

    class Reader(nn.Module):
        """One learned query token, prompt cross-attention, predicts log sigma."""

        def __init__(self):
            super().__init__()
            self.q = nn.Parameter(torch.zeros(1, 1, d_model))
            self.enc = PromptEncoder(d_model, nuisance_features=1, nuisance_tokens=2)
            self.xattn = PromptCrossAttention(d_model, 4)
            self.out = nn.Linear(d_model, 1)

        def forward(self, nuis_rows):
            b = nuis_rows.shape[0] if nuis_rows is not None else 1
            mem = self.enc(None, nuisance=nuis_rows) if nuis_rows is not None else None
            h = self.xattn(self.q.expand(b, -1, -1), mem)
            return self.out(h).squeeze(-1).squeeze(-1)

    def nuisance_rows(subj):
        """Per-channel nuisance = the subject's log gain, with measurement noise."""
        g = torch.tensor(log_gain[subj], dtype=torch.float32)
        return (g[:, None, None] + 0.1 * torch.randn(len(subj), 8, 1))

    def nll(log_sigma, xs):
        return (0.5 * math.log(2 * math.pi) + log_sigma
                + 0.5 * xs**2 * torch.exp(-2 * log_sigma)).mean()

    model = Reader()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(600):
        subj = rng.integers(0, train_subj, 64)
        xs = torch.randn(64) * torch.exp(torch.tensor(log_gain[subj], dtype=torch.float32))
        use_prompt = rng.random() > 0.2
        ls = model(nuisance_rows(subj) if use_prompt else None)
        loss = nll(ls, xs)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        subj = rng.integers(train_subj, train_subj + test_subj, 4000)   # UNSEEN subjects
        other = rng.integers(train_subj, train_subj + test_subj, 4000)
        other = np.where(other == subj, (other + 1 - train_subj) % test_subj + train_subj, other)
        xs = torch.randn(4000) * torch.exp(torch.tensor(log_gain[subj], dtype=torch.float32))
        optimum = float((0.5 * math.log(2 * math.pi * math.e)
                         + torch.tensor(log_gain[subj])).mean())
        matched = float(nll(model(nuisance_rows(subj)), xs))
        none = float(nll(model(None).expand(4000), xs))
        wrong = float(nll(model(nuisance_rows(other)), xs))

    print(f"  held-out subjects, NLL (nats):  optimum {optimum:.3f}")
    print(f"    matched prompt    {matched:.3f}   (+{matched - optimum:.3f} over optimum)")
    print(f"    no prompt         {none:.3f}   (+{none - optimum:.3f})")
    print(f"    WRONG subject     {wrong:.3f}   (+{wrong - optimum:.3f})")
    ok &= check("matched prompt nearly reaches the optimum on unseen subjects",
                matched - optimum < 0.05, f"gap {matched - optimum:.3f}")
    ok &= check("the prompt carries information weights cannot hold",
                none - matched > 0.10, f"no-prompt costs {none - matched:.3f} nats")
    ok &= check("a mismatched prompt is worse than none (it is actually read)",
                wrong > none, f"{wrong:.3f} > {none:.3f}")

    print()
    print("=" * 80)
    if ok:
        print("CONDITIONING VALIDATED: causal in both streams, TRF recovered, and a")
        print("new subject's statistics are read from context rather than weights.")
        return 0
    print("CONDITIONING FAILED validation.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
