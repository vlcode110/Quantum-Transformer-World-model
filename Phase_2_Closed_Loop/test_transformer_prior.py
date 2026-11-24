"""Checks for the quantum transformer prior. Run before any training.

The prior keeps a rolling window of recurrent states as mutable state on a
module. That is the fragile part: a leak between episodes would not raise an
error, it would quietly let one episode attend over the previous one and
produce a plausible-looking result. Every check below targets that, plus the
basics of shape and gradient flow.

    python test_transformer_prior.py
    python test_transformer_prior.py --classical

Exit status 0 only if everything passes.
"""
import argparse
import sys

import numpy as np
import torch

import quantum_rssm as qr
from config import Config

results = []


def check(name, fn):
    try:
        detail = fn()
        results.append(True)
        print(f"  PASS  {name}" + (f"   ({detail})" if detail else ""))
    except Exception as e:
        results.append(False)
        print(f"  FAIL  {name}\n          {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classical", action="store_true",
                    help="test the classical twin instead of the circuit")
    ap.add_argument("--deter", type=int, default=64)
    ap.add_argument("--out", type=int, default=256)
    ap.add_argument("--batch", type=int, default=5)
    args = ap.parse_args()

    cfg = Config()
    cfg.prior_arch = "transformer"
    quantum = not args.classical
    print(f"quantum transformer prior ({'PQC' if quantum else 'classical twin'}), "
          f"deter={args.deter}, K={cfg.prior_context}, "
          f"d_model={cfg.d_model}, heads={cfg.n_heads}\n")

    core = qr.load_core_module(cfg.phase1_root)
    prior = qr.TransformerPrior(args.deter, args.out, core, cfg, quantum=quantum)

    B, D = args.batch, args.deter

    def c1():
        prior.reset()
        out = prior(torch.randn(B, D))
        assert out.shape == (B, args.out), f"got {tuple(out.shape)}"
        assert torch.isfinite(out).all(), "non-finite logits"
        return f"output {tuple(out.shape)}"
    check("1  output shape and finiteness", c1)

    def c2():
        """The window must actually hold K states, not just the latest."""
        prior.reset()
        for _ in range(cfg.prior_context + 3):
            prior(torch.randn(B, D))
        assert prior._buf is not None, "no buffer"
        assert prior._buf.shape == (B, cfg.prior_context, D), \
            f"buffer {tuple(prior._buf.shape)}"
        return f"window {tuple(prior._buf.shape)} after "\
               f"{cfg.prior_context + 3} steps"
    check("2  window is bounded at K", c2)

    def c3():
        """History must change the answer, or the attention is decorative.

        In eval mode: the encoder uses dropout, so in training mode two
        identical calls already differ and this check would pass on noise.
        """
        prior.eval()
        with torch.no_grad():
            x = torch.randn(B, D)
            prior.reset()
            fresh = prior(x)
            prior.reset()
            for _ in range(cfg.prior_context):
                prior(torch.randn(B, D))
            with_history = prior(x)
        prior.train()
        delta = float((fresh - with_history).abs().mean())
        assert delta > 1e-5, \
            "identical output with and without history — the prior is "\
            "ignoring its window"
        return f"mean |difference| {delta:.4f} (dropout disabled)"
    check("3  history affects the prediction", c3)

    def c4():
        """reset() must fully clear, so a new sequence starts clean.

        Also in eval mode, for the same reason: with dropout active the two
        runs differ regardless of whether the buffer was cleared, so the
        check would fail on a correct implementation.
        """
        prior.eval()
        with torch.no_grad():
            prior.reset()
            for _ in range(cfg.prior_context):
                prior(torch.ones(B, D))
            prior.reset()
            x = torch.randn(B, D)
            a = prior(x)
            prior.reset()
            b = prior(x)
        prior.train()
        assert torch.allclose(a, b, atol=1e-6), \
            f"output after reset depends on what came before it "\
            f"(max difference {float((a-b).abs().max()):.2e})"
        return "two fresh sequences agree exactly"
    check("4  reset() clears all history", c4)

    def c4b():
        """Confirm the nondeterminism in training mode really is dropout.

        If eval mode is deterministic and training mode is not, the only
        stochastic component is the encoder dropout — which is expected and
        must not be mistaken for the buffer leaking.
        """
        prior.train()
        prior.reset(); a = prior(torch.ones(B, D))
        prior.reset(); b = prior(torch.ones(B, D))
        train_differs = float((a - b).abs().max()) > 1e-6
        prior.eval()
        with torch.no_grad():
            prior.reset(); c = prior(torch.ones(B, D))
            prior.reset(); d = prior(torch.ones(B, D))
        prior.train()
        eval_same = torch.allclose(c, d, atol=1e-7)
        assert eval_same, "eval mode is nondeterministic — not dropout"
        assert train_differs, "training mode is deterministic — dropout is off"
        return "stochastic in train, deterministic in eval, as expected"
    check("4b nondeterminism is dropout, not state", c4b)

    def c5():
        """reset_rows() must clear only the flagged rows, and really clear them."""
        prior.reset()
        for _ in range(cfg.prior_context):
            prior(torch.randn(B, D))
        before = prior._buf.clone()
        mask = torch.zeros(B, dtype=torch.bool)
        mask[0] = mask[3] = True
        prior.reset_rows(mask)
        assert float(prior._buf[0].abs().sum()) == 0.0, "row 0 not cleared"
        assert float(prior._buf[3].abs().sum()) == 0.0, "row 3 not cleared"
        for r in (1, 2, 4):
            assert torch.equal(prior._buf[r], before[r]), \
                f"row {r} was cleared but should not have been"
        return "rows 0 and 3 cleared, 1/2/4 untouched"
    check("5  reset_rows() is selective and complete", c5)

    def c6():
        """A batch-size change must not reuse the previous window."""
        prior.reset()
        for _ in range(cfg.prior_context):
            prior(torch.randn(B, D))
        out = prior(torch.randn(B * 3, D))
        assert out.shape == (B * 3, args.out)
        assert prior._buf.shape[0] == B * 3
        return f"re-seeded for batch {B * 3}"
    check("6  batch-size change re-seeds the window", c6)

    def c7():
        """Gradients must reach the circuit, not just the surrounding layers."""
        prior.reset()
        out = prior(torch.randn(B, D))
        out.sum().backward()
        q = [p for n, p in prior.named_parameters()
             if "q." in n and "weights" in n and p.grad is not None]
        c = [p for n, p in prior.named_parameters()
             if not ("q." in n and "weights" in n) and p.grad is not None]
        if quantum:
            assert q, "no gradient on any circuit parameter"
            g = torch.cat([p.grad.flatten() for p in q])
            assert float(g.abs().sum()) > 0, "circuit gradients are all zero"
            detail = f"circuit |grad| {float(g.abs().mean()):.3e}, "\
                     f"{len(c)} classical tensors also have gradients"
        else:
            detail = f"{len(c)} tensors have gradients (classical twin)"
        assert c, "no gradient on the classical parameters"
        return detail
    check("7  gradients reach the circuit", c7)

    def c8():
        """Both arms must be close in size, or the comparison is confounded."""
        q_prior = qr.TransformerPrior(args.deter, args.out, core, cfg,
                                      quantum=True)
        c_prior = qr.TransformerPrior(args.deter, args.out, core, cfg,
                                      quantum=False)
        nq = sum(p.numel() for p in q_prior.parameters())
        nc = sum(p.numel() for p in c_prior.parameters())
        rel = abs(nq - nc) / max(nc, 1)
        assert rel < 0.15, f"parameter counts differ by {rel:.1%} ({nq} vs {nc})"
        return f"quantum {nq:,} ({q_prior.n_quantum_params} circuit), "\
               f"classical {nc:,}, {rel:.2%} apart"
    check("8  arms are capacity-matched", c8)

    print()
    ok = sum(results)
    if ok == len(results):
        print(f"ALL {len(results)} CHECKS PASSED.\n")
        print("Next:  python train_agent.py --task cartpole --quantum true "
              "--seed 0 --steps 2500")
        print("       (a short run first — confirm it trains before "
              "committing a night to it)")
    else:
        print(f"{len(results) - ok} of {len(results)} FAILED — do not train.")
    sys.exit(0 if ok == len(results) else 1)


if __name__ == "__main__":
    main()
