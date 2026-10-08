"""Verify the backbone is causal and the objectives are sound.

The headline test is **causality**. A backbone that leaks one step of future
information manufactures precisely the result this project exists to measure
honestly: it would appear to forecast brain activity beautifully while actually
reading the answer. Nothing downstream -- not the Atlas, not the noisy-channel
decoder, not H1 -- survives that bug, and it does not announce itself.

So it is tested by perturbation, not by reading the mask code.

Also checked:
  * the AR shift aligns output i with target i+1 (the other dangerous off-by-one)
  * the mixture head is a proper density (integrates to 1)
  * sliding-window attention actually respects its window
  * prompt cross-attention starts as a no-op and is KV-only
  * the collapse monitor fires on a collapsed representation
  * all five arms run, produce finite losses, and reach the trunk
  * NO ARM'S TARGET REACHES ITS OWN INPUT -- checked by perturbing exactly the
    samples each arm must predict. An earlier training loop failed this for three
    of the five arms: masked and JEPA never masked their input, and the
    peripheral arm predicted a projection of its own input token.

Run:  .venv/Scripts/python.exe scripts/validate_backbone.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from neurocast.models.backbone import Backbone, BackboneConfig, CausalSelfAttention  # noqa: E402
from neurocast.models.heads import GaussianMixtureHead  # noqa: E402
from neurocast.models.neurocast import NeuroCast  # noqa: E402
from neurocast.objectives.arms import (  # noqa: E402
    CollapseMonitor,
    EMATeacher,
    OBJECTIVES,
    ar_shift,
    block_mask,
    build,
)
from neurocast.train.loop import arm_context, group_targets, valid_positions  # noqa: E402
from validate_tokenizer import eeg_montage, megin_montage  # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    return ok


def main() -> int:
    torch.manual_seed(0)
    ok = True

    cfg = BackboneConfig("6m", window=8, global_every=5, prompt_every=6)
    net = Backbone(cfg).eval()
    print(net.describe())

    b, n, d = 2, 24, cfg.d_model

    print()
    print("=" * 78)
    print("1. CAUSALITY -- the test that matters most")
    print("=" * 78)

    x = torch.randn(b, n, d)
    with torch.no_grad():
        out1 = net(x)
        for j in (5, 12, 23):
            x2 = x.clone()
            x2[:, j] += 10.0 * torch.randn(b, d)
            out2 = net(x2)
            delta = (out1 - out2).abs().amax(dim=(0, 2))

            past = float(delta[:j].max()) if j > 0 else 0.0
            here = float(delta[j])
            ok &= check(
                f"perturbing position {j:>2} leaves all earlier positions untouched",
                past == 0.0,
                f"max |delta| over positions <{j} = {past:.3e}",
            )
            ok &= check(
                f"  ...and does change position {j} itself",
                here > 1e-4,
                f"{here:.3e}",
            )

    print("       ^ zero leakage backwards: the model cannot see its own future.")

    print()
    print("=" * 78)
    print("2. Sliding window actually bounds the receptive field")
    print("=" * 78)

    solo = Backbone(BackboneConfig("6m", window=4, global_every=10_000,
                                   prompt_every=10_000)).eval()
    # 12 layers of window-4 attention: receptive field grows with depth, so use
    # a single-layer probe to test the window itself.
    layer = solo.layers[0]
    with torch.no_grad():
        y1 = layer(x)
        x3 = x.clone()
        x3[:, 0] += 10.0 * torch.randn(b, d)
        y2 = layer(x3)
        delta = (y1 - y2).abs().amax(dim=(0, 2))
    far = float(delta[6:].max())
    near = float(delta[:4].max())
    ok &= check("one window-4 layer does not propagate beyond its window",
                far == 0.0, f"max |delta| at positions >=6 is {far:.3e}")
    ok &= check("  ...but does affect positions inside the window",
                near > 1e-4, f"{near:.3e}")

    print()
    print("=" * 78)
    print("3. Prompt cross-attention: zero-init and KV-only")
    print("=" * 78)

    prompt = torch.randn(b, 16, d)
    with torch.no_grad():
        no_prompt = net(x, prompt=None)
        with_prompt = net(x, prompt=prompt)
    ok &= check("at init the prompt is a no-op (zero-init output projection)",
                torch.allclose(no_prompt, with_prompt, atol=1e-6),
                "model starts prompt-free and learns to use context")

    xattn = next(l for l, k in zip(net.layers, net.kinds) if k == "xattn")
    with torch.no_grad():
        xattn.proj.weight.normal_(0, 0.02)
        differs = not torch.allclose(net(x, prompt=None), net(x, prompt=prompt), atol=1e-6)
        xattn.proj.weight.zero_()
    ok &= check("once trained-in, the prompt changes the output", differs)

    print()
    print("=" * 78)
    print("4. AR shift -- the other dangerous off-by-one")
    print("=" * 78)

    h = torch.arange(n, dtype=torch.float32).reshape(1, n, 1).expand(b, n, 1)
    tgt = h.clone()
    hs, ts = ar_shift(h, tgt)
    ok &= check("shapes drop exactly one position",
                hs.shape[1] == n - 1 and ts.shape[1] == n - 1)
    ok &= check("output at i is paired with target at i+1",
                bool(torch.all(ts[0, :, 0] - hs[0, :, 0] == 1.0)),
                "h[i] predicts target[i+1], never target[i]")
    try:
        ar_shift(torch.randn(1, 1, 4), torch.randn(1, 1, 4))
        ok &= check("a length-1 sequence is rejected", False)
    except ValueError:
        ok &= check("a length-1 sequence is rejected", True)

    print()
    print("=" * 78)
    print("5. Mixture head is a proper density")
    print("=" * 78)

    head = GaussianMixtureHead(d_model=16, target_dim=1, n_components=5).eval()
    ctx = torch.randn(1, 16)
    grid = torch.linspace(-12, 12, 24_001).reshape(-1, 1)
    with torch.no_grad():
        lp = head.log_prob(ctx.expand(grid.shape[0], -1), grid)
        integral = float(torch.trapezoid(lp.exp(), grid.squeeze(-1)))
    ok &= check("density integrates to 1", abs(integral - 1.0) < 1e-3,
                f"integral = {integral:.6f}")

    with torch.no_grad():
        nll = float(head(ctx.expand(4096, -1), torch.randn(4096, 1)))
    import math

    expected = 0.5 * math.log(2 * math.pi) + 0.5
    ok &= check("NLL of standard-normal draws matches Gaussian entropy",
                abs(nll - expected) < 0.05, f"{nll:.4f} vs {expected:.4f}")

    # Worst-case bound. Force a maximally overconfident head (every component at
    # the log_sigma floor of -7) and score a target far from its mean.
    hot = GaussianMixtureHead(d_model=16, target_dim=4, n_components=5).eval()
    raw = GaussianMixtureHead(d_model=16, target_dim=4, n_components=5, defend=None).eval()
    with torch.no_grad():
        for hd in (hot, raw):
            hd.proj.weight.zero_()
            bias = torch.zeros(4, 5, 3)
            bias[..., 2] = -7.0                      # log_sigma -> sigma = e^-7
            hd.proj.bias.copy_(bias.reshape(-1))
        far = torch.full((1, 4), 3.0)
        nll_raw = float(-raw.log_prob(ctx, far))
        nll_def = float(-hot.log_prob(ctx, far))
        bg_nll = float(4 * (0.5 * math.log(2 * math.pi) + 0.5 * 9.0))
        cap = bg_nll - 4 * math.log(1e-3)
    print(f"       overconfident head, target 3 SD away: undefended NLL {nll_raw:,.0f}  "
          f"defended {nll_def:.1f}  (cap {cap:.1f})")
    ok &= check("defensive mixture bounds a catastrophically overconfident head",
                nll_def <= cap + 1e-4 and nll_raw > 100 * cap,
                "an eye blink after a quiet stretch cannot dominate the loss")

    print()
    print("=" * 78)
    print("6. Collapse monitor fires on a collapsed representation")
    print("=" * 78)

    mon = CollapseMonitor()
    healthy = mon.check(torch.randn(256, 64))
    collapsed = mon.check(torch.randn(256, 1).expand(256, 64) + 1e-4 * torch.randn(256, 64))
    print(f"       healthy   stable_rank_frac={healthy['stable_rank_frac']:.3f} "
          f"min_std={healthy['min_std']:.3f} offdiag={healthy['offdiag_corr']:.3f}")
    print(f"       collapsed stable_rank_frac={collapsed['stable_rank_frac']:.3f} "
          f"min_std={collapsed['min_std']:.3f} offdiag={collapsed['offdiag_corr']:.3f}")
    ok &= check("healthy representation passes", not healthy["collapsing"])
    ok &= check("collapsed representation is flagged", bool(collapsed["collapsing"]))

    print()
    print("=" * 78)
    print("7. All five arms run and reach the trunk")
    print("=" * 78)

    d_model, target_dim, n_periph = 32, 12, 3
    seq = 20
    kwargs = {
        "masked": dict(d_model=d_model, target_dim=target_dim),
        "jepa": dict(d_model=d_model),
        "ar_latent": dict(d_model=d_model),
        "ar_lik": dict(d_model=d_model, target_dim=target_dim),
        "peripheral": dict(d_model=d_model, n_peripheral=n_periph),
    }
    for name in OBJECTIVES:
        arm = build(name, **kwargs[name])
        trunk = torch.nn.Linear(d_model, d_model)
        h = trunk(torch.randn(2, seq, d_model))
        ctx = {
            "h": h,
            "target": torch.randn(2, seq, target_dim if name in ("masked", "ar_lik") else d_model),
            "mask": block_mask((2, seq), block_len=4, target_frac=0.4),
            "peripheral_target": torch.randn(2, seq, n_periph),
        }
        out = arm.loss(ctx)
        loss = out["loss"]
        loss.backward()
        grad = trunk.weight.grad
        finite = bool(torch.isfinite(loss)) and grad is not None and bool(torch.isfinite(grad).all())
        nonzero = grad is not None and float(grad.abs().sum()) > 0
        ok &= check(f"arm '{name:<10}' loss={float(loss):8.4f}",
                    finite and nonzero,
                    "finite loss, gradient reaches trunk")

    m = block_mask((4, 100), block_len=10, target_frac=0.4)
    frac = float(m.float().mean())
    ok &= check("block mask hits its target fraction",
                0.30 < frac < 0.55, f"{frac:.3f} masked")

    ok &= target_leakage()

    print()
    print("=" * 78)
    if ok:
        print("BACKBONE VALIDATED: causal, windowed, proper density, all five arms")
        print("differentiable through a shared trunk, and no arm sees its own target.")
        return 0
    print("BACKBONE FAILED validation.")
    return 1


def target_leakage() -> bool:
    """Section 8: perturb exactly what each arm predicts; its input must not move."""
    print()
    print("=" * 78)
    print("8. NO ARM'S TARGET REACHES ITS OWN INPUT")
    print("=" * 78)
    ok = True
    torch.manual_seed(0)
    montage = megin_montage(n_sites=16)
    quad = torch.from_numpy(montage.quadrants)
    G, P, n_patch = 5, 16, 6
    x = torch.randn(2, len(montage), n_patch * P)
    proj = torch.randn(P, 8) / P**0.5

    # The bidirectional switch must be real, and the default must stay causal.
    attn_bi = CausalSelfAttention(32, 4, causal=False).eval()
    xs = torch.randn(1, 10, 32)
    with torch.no_grad():
        xs2 = xs.clone()
        # Random, not constant: the pre-attention LayerNorm erases a uniform shift.
        xs2[:, 7] += 5.0 * torch.randn(32)
        back = float((attn_bi(xs) - attn_bi(xs2))[:, :7].abs().max())
    ok &= check("causal=False really is bidirectional (a later position moves earlier ones)",
                back > 1e-4, f"{back:.2e}")

    def perturb_cell(z: torch.Tensor, t: int, g: int) -> torch.Tensor:
        """Add noise to exactly the raw samples behind position (t, g)."""
        z2 = z.clone()
        idx = torch.nonzero(quad == g).flatten()
        z2[:, idx, t * P:(t + 1) * P] += 5.0 * torch.randn(z.shape[0], idx.numel(), P)
        return z2

    for arm in ("masked", "jepa"):
        model = NeuroCast(montage, rung="tiny", window=64, causal=False).eval()
        teacher = EMATeacher(model) if arm == "jepa" else None
        gen = torch.Generator().manual_seed(3)
        with torch.no_grad():
            c1 = arm_context(arm, model, x, proj=proj, teacher=teacher, generator=gen)
            p = int(torch.nonzero(c1["mask"][0]).flatten()[0])
            t, g = divmod(p, G)
            # Same mask for the perturbed input, and for the unmasked contrast.
            mask = c1["mask"]
            x2 = perturb_cell(x, t, g)
            h2 = model(x2, mask=mask)
            leak = float((c1["h"] - h2).abs().max())
            unmasked = float((model(x) - model(x2)).abs().max())
            tgt2 = (group_targets(x2, quad, G, P, proj) if arm == "masked"
                    else teacher(x2))
            moved = float((tgt2 - c1["target"])[:, p].abs().max())
        ok &= check(f"{arm:<7} masked cell (t={t}, g={g}): perturbing its raw samples "
                    "moves NOTHING the trunk outputs", leak == 0.0, f"max |delta| {leak:.1e}")
        ok &= check(f"{arm:<7}   ...the same perturbation is visible without the mask",
                    unmasked > 1e-4, f"{unmasked:.1e}  (the guard is not vacuous)")
        ok &= check(f"{arm:<7}   ...and it does change that cell's target",
                    moved > 1e-4, f"{moved:.1e}")

    # Peripheral: h is read at the last brain group, which precedes periph_t.
    model = NeuroCast(montage, rung="tiny", window=64).eval()
    t = 3
    with torch.no_grad():
        c1 = arm_context("peripheral", model, x, proj=proj)
        x2 = perturb_cell(x, t, G - 1)
        c2 = arm_context("peripheral", model, x2, proj=proj)
        leak = float((c1["h"] - c2["h"])[:, : t + 1].abs().max())
        moved = float((c1["peripheral_target"] - c2["peripheral_target"])[:, t].abs().max())
        own = float((model(x) - model(x2))[:, t * G + G - 1].abs().max())
    ok &= check("peripheral: predictions for patches <= t ignore periph_t",
                leak == 0.0, f"max |delta| {leak:.1e}")
    ok &= check("peripheral:   ...while the target at t does change", moved > 1e-4, f"{moved:.1e}")
    ok &= check("peripheral:   ...and reading h AT the peripheral position would have leaked it",
                own > 1e-4, f"{own:.1e}  (what the old loop did)")

    # Empty quadrant: an EEG cap has no peripherals.
    eeg = eeg_montage(16)
    v = valid_positions(torch.from_numpy(eeg.quadrants), G, 4)
    ok &= check("empty peripheral quadrant is excluded from observation targets",
                not bool(v[G - 1::G].any()) and bool(v.reshape(4, G)[:, :4].all()),
                f"{int(v.sum())}/{v.numel()} positions scored")
    head = GaussianMixtureHead(16, 6).eval()
    hh, tt = torch.randn(3, 16), torch.randn(3, 6)
    dm = torch.tensor([True, True, True, False, False, False])
    with torch.no_grad():
        a = head.log_prob(hh, tt, dm)
        tt2 = tt.clone()
        tt2[:, 3:] = 1e3
        b = head.log_prob(hh, tt2, dm)
    ok &= check("masked target coordinates contribute exactly 0 nats",
                bool(torch.equal(a, b)), "a constant coordinate cannot be fitted to a delta")
    try:
        NeuroCast(montage, rung="tiny").tokens(x, mask=torch.zeros(2, n_patch * G, dtype=torch.bool))
        ok &= check("a causal model refuses input masking", False)
    except ValueError:
        ok &= check("a causal model refuses input masking", True)
    return ok


if __name__ == "__main__":
    raise SystemExit(main())
