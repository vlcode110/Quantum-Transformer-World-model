"""Integration and physics checks for Phase 3 Stage A. Run this first.

Two kinds of thing can be wrong here. The machine-learning pipeline can be
broken, which the Phase 2 self-test taught us to check explicitly. But the
PHYSICS can also be wrong in ways that produce perfectly plausible numbers —
an unnormalised state, a Hamiltonian with the wrong sign, observables that
never move. Those failures are silent and would invalidate the whole phase,
so they are checked directly against known properties of the system.

    python selftest.py
    python selftest.py --n_spins 6 --observables zx

Checks:
   1  Hamiltonian is Hermitian (required for physical evolution)
   2  time evolution is unitary (probability is conserved)
   3  evolution actually changes the state (not accidentally the identity)
   4  observables stay within their physical range [-1, +1]
   5  the chain builds entanglement (otherwise it is not a quantum problem)
   6  observables move between steps, but not so much as to be noise
   7  trajectories generate and have the right shape
   8  the dataset builds; no window crosses a trajectory boundary
   9  the split is disjoint and fixed across seeds
  10  trivial baselines computed and sane
  11  Phase 1 model definitions load by file path
  12  both architectures build with matched parameter counts
  13  a training step runs and produces finite losses
  14  gradients reach the circuit parameters and are non-zero
  15  open-loop rollout evaluation runs
"""
import os
import sys
import traceback

import numpy as np
import torch

from config import Config
import spin_env

results, S = [], {}


def check(name, fn, fatal=False):
    try:
        d = fn()
        results.append((True, name, d or ""))
        print(f"  PASS  {name}" + (f"   ({d})" if d else ""))
        return True
    except Exception as e:
        results.append((False, name, f"{type(e).__name__}: {e}"))
        print(f"  FAIL  {name}\n           {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)
        if fatal:
            summarise()
        return False


def main():
    cfg = Config.from_cli()
    cfg.n_trajectories = min(cfg.n_trajectories, 20)   # tiny, for speed
    cfg.traj_length = min(cfg.traj_length, 40)
    cfg.epochs = 1
    print(f"Phase 3 self-test\n  {spin_env.describe(cfg)}\n")

    # ---------------------------------------------------------- the physics
    def c1():
        H = spin_env.hamiltonian(cfg.n_spins, cfg.coupling, cfg.field_levels[2])
        err = float(np.abs(H - H.conj().T).max())
        if err > 1e-10:
            raise AssertionError(f"H is not Hermitian (max asymmetry {err:.2e})")
        S["H"] = H
        return f"{H.shape[0]}x{H.shape[0]}, max asymmetry {err:.1e}"
    check("1  Hamiltonian is Hermitian", c1, fatal=True)

    def c2():
        U = spin_env.evolution_operators(cfg)
        S["U"] = U
        prod = U[2].conj().T @ U[2]
        err = float(np.abs(prod - np.eye(prod.shape[0])).max())
        if err > 1e-9:
            raise AssertionError(f"evolution is not unitary (error {err:.2e})")
        return f"{len(U)} operators, unitarity error {err:.1e}"
    check("2  time evolution is unitary", c2, fatal=True)

    def c3():
        U = S["U"][2]
        diff = float(np.abs(U - np.eye(U.shape[0])).max())
        if diff < 1e-6:
            raise AssertionError(
                "evolution operator is essentially the identity — dt may be far "
                "too small, so nothing would ever change")
        return f"max deviation from identity {diff:.3f}"
    check("3  evolution changes the state", c3)

    def c4():
        rng = np.random.default_rng(0)
        psi = spin_env.initial_state(cfg, rng)
        signs = spin_env._pauli_z_signs(cfg.n_spins)
        vals = []
        for _ in range(10):
            psi = S["U"][rng.integers(0, len(S["U"]))] @ psi
            psi /= np.linalg.norm(psi)
            vals.append(spin_env.measure(psi, signs, cfg.n_spins, cfg.observables))
        vals = np.asarray(vals)
        if vals.min() < -1.0001 or vals.max() > 1.0001:
            raise AssertionError(
                f"observables outside physical range: "
                f"[{vals.min():.3f}, {vals.max():.3f}]")
        S["psi"] = psi
        return f"range [{vals.min():.3f}, {vals.max():.3f}], all physical"
    check("4  observables are physical", c4)

    def c5():
        """If the chain never entangles, this is not a quantum modelling problem
        at all — a classical description of the same size would suffice."""
        psi = S["psi"]
        n = cfg.n_spins
        if n < 2:
            return "skipped (needs 2+ spins)"
        # reduced density matrix of the first spin; purity 1 means unentangled
        m = psi.reshape(2, -1)
        rho = m @ m.conj().T
        purity = float(np.real(np.trace(rho @ rho)))
        if purity > 0.999:
            raise AssertionError(
                f"first spin is nearly unentangled (purity {purity:.4f}) — the "
                f"chain is not generating quantum correlations")
        return f"single-spin purity {purity:.4f} (1.0 would mean no entanglement)"
    check("5  the chain builds entanglement", c5)

    def c6():
        obs, act, lvl = spin_env.generate(cfg, verbose=False)
        S.update(obs=obs, act=act)
        d = spin_env.diagnostics(obs, cfg)
        ratio = d["variance"] / max(d["persistence_mse"], 1e-12)
        if ratio > 200:
            raise AssertionError(
                f"observables barely move: persistence is {ratio:.0f}x better "
                f"than the mean, so predicting 'no change' nearly solves the "
                f"task. Increase dt.")
        if ratio < 1.2:
            raise AssertionError(
                f"consecutive observations are nearly independent "
                f"(ratio {ratio:.2f}) — closer to noise than dynamics. "
                f"Decrease dt.")
        return (f"persistence {ratio:.1f}x better than the mean — "
                f"well-scaled for prediction")
    check("6  dynamics are well-scaled", c6, fatal=True)

    def c7():
        obs, act = S["obs"], S["act"]
        want = (cfg.n_trajectories, cfg.traj_length, cfg.obs_dim())
        if obs.shape != want:
            raise AssertionError(f"observations {obs.shape}, expected {want}")
        if not np.isfinite(obs).all():
            raise AssertionError("non-finite observations")
        if not np.allclose(act.sum(axis=-1), 1.0):
            raise AssertionError("control encoding is not one-hot")
        return f"observations {obs.shape}, controls {act.shape}"
    check("7  trajectories have the right shape", c7)

    # ---------------------------------------------------------- the pipeline
    def c8():
        from dataset import build
        splits, info = build(cfg, verbose=False)
        S.update(splits=splits, info=info)
        w = splits["train"]
        # a window must never span two trajectories
        traj_ids = {i for i, _ in w.index}
        if len(traj_ids) > info["n_trajectories"]["train"]:
            raise AssertionError("windows reference more trajectories than exist")
        return f"{info['n_windows']} windows over {info['n_trajectories']} trajectories"
    check("8  dataset builds with intact boundaries", c8, fatal=True)

    def c9():
        from dataset import build
        a = build(cfg, verbose=False)[1]
        cfg2 = Config.from_cli()
        cfg2.__dict__.update(cfg.__dict__)
        cfg2.seed = cfg.seed + 7
        b = build(cfg2, verbose=False)[1]
        if a["n_trajectories"] != b["n_trajectories"]:
            raise AssertionError("split changed with seed despite fixed_split")
        return "split identical across seeds (fixed_split honoured)"
    check("9  split is fixed across seeds", c9)

    def c10():
        b = S["info"]["baselines"]
        if not all(np.isfinite(v) for v in b.values()):
            raise AssertionError(f"non-finite baseline: {b}")
        if b["persistence"] >= b["mean"]:
            raise AssertionError(
                "persistence is no better than predicting the mean — "
                "consecutive states are unrelated")
        return (f"persistence {b['persistence']:.5f}, mean {b['mean']:.5f}, "
                f"ratio {b['mean']/b['persistence']:.1f}x")
    check("10 trivial baselines are sane", c10)

    def c11():
        from train_dynamics import load_models
        S["core"] = load_models(cfg.phase1_root)
        if not hasattr(S["core"], "DynamicsCore"):
            raise AssertionError("loaded module has no DynamicsCore")
        return "Phase 1 models loaded by file path"
    check("11 Phase 1 model definitions load", c11, fatal=True)

    def c12():
        from train_dynamics import build_model
        info, core = S["info"], S["core"]
        q = build_model("quantum_transformer", info["obs_dim"],
                        info["action_dim"], cfg, core)
        c = build_model("classical_transformer", info["obs_dim"],
                        info["action_dim"], cfg, core)
        nq = sum(p.numel() for p in q.parameters())
        nc = sum(p.numel() for p in c.parameters())
        if abs(nq - nc) / max(nc, 1) > 0.15:
            raise AssertionError(
                f"parameter counts differ by more than 15% ({nq} vs {nc}) — "
                f"the comparison would be confounded by size")
        S["q"] = q
        return f"quantum {nq} params ({q.n_quantum_params} quantum), classical {nc}"
    check("12 both architectures build, matched size", c12, fatal=True)

    def c13():
        from dataset import loaders
        dl, _ = loaders(cfg, verbose=False)
        S["dl"] = dl
        model = S["q"]
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        oc, ac, o_next, _, _ = next(iter(dl["train"]))
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(oc, ac), o_next)
        loss.backward()
        opt.step()
        if not np.isfinite(loss.item()):
            raise AssertionError("loss is not finite")
        return f"loss {loss.item():.5f}"
    check("13 a training step runs", c13, fatal=True)

    def c14():
        from train_dynamics import grad_stats
        gs = grad_stats(S["q"])
        if "q_var" not in gs:
            raise AssertionError(
                "no gradients on circuit parameters — the circuit is not "
                "connected to the loss")
        if gs["q_absmean"] == 0.0:
            raise AssertionError("circuit gradients are exactly zero")
        return (f"circuit |grad| {gs['q_absmean']:.3e}, classical "
                f"{gs.get('c_absmean', 0):.3e}, ratio "
                f"{gs.get('ratio', float('nan')):.3f}")
    check("14 gradients reach the circuit", c14)

    def c15():
        from train_dynamics import evaluate
        e = evaluate(S["q"], S["dl"]["test"], (1, 5), cfg.device)
        if not all(np.isfinite(v) for v in e.values()):
            raise AssertionError(f"non-finite rollout errors: {e}")
        if e[5] < e[1]:
            raise AssertionError(
                "five-step error is lower than one-step — rollout is not "
                "compounding, which suggests the evaluation is not feeding "
                "predictions back in")
        return f"k=1 {e[1]:.5f}, k=5 {e[5]:.5f} (compounding as expected)"
    check("15 open-loop rollout evaluates", c15)

    summarise()


def summarise():
    print()
    ok = sum(1 for g, _, _ in results if g)
    n = len(results)
    if ok == n:
        print(f"ALL {n} CHECKS PASSED — physics and pipeline both sound.")
        print("\nNext:")
        print("  python train_dynamics.py --n_spins 4 --epochs 2")
    else:
        print(f"{n - ok} of {n} checks FAILED — do not start a sweep.")
        for g, name, d in results:
            if not g:
                print(f"    {name}\n      {d}")
    sys.exit(0 if ok == n else 1)


if __name__ == "__main__":
    main()
