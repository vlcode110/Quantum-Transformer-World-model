"""Train one architecture to predict spin-chain dynamics.

Reuses the Phase 1 model definitions rather than duplicating them, so the two
phases can never drift apart in what "the quantum transformer" means. Models
are loaded by explicit file path because both projects define config.py.

Usage
    python train_dynamics.py                          # both models, n=6
    python train_dynamics.py --n_spins 8 --seed 1
    python train_dynamics.py --epochs 2               # smoke test
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from config import Config
from dataset import loaders


# ------------------------------------------------------- Phase 1 model import
def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_models(phase1_root):
    here = os.path.dirname(os.path.abspath(__file__))
    core = os.path.join(here, phase1_root, "models", "quantum_core.py")
    if not os.path.exists(core):
        sys.exit(f"[phase3] Phase 1 models not found at {core} — "
                 f"fix `phase1_root` in config.py.")
    return _load(os.path.abspath(core), "_phase1_quantum_core")


def build_model(name, obs_dim, action_dim, cfg, core):
    """The same two architectures compared throughout the project."""
    if name == "quantum_transformer":
        return core.DynamicsCore(obs_dim, action_dim, cfg, quantum=True)
    if name == "classical_transformer":
        return core.DynamicsCore(obs_dim, action_dim, cfg, quantum=False)
    raise ValueError(f"unknown model {name!r}")


# ------------------------------------------------------------------ gradients
def grad_stats(model):
    q, c = [], []
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().flatten()
        (q if ("q." in n and "weights" in n) else c).append(g)
    out = {}
    if q:
        gq = torch.cat(q)
        out["q_var"] = float(gq.var()); out["q_absmean"] = float(gq.abs().mean())
    if c:
        gc = torch.cat(c)
        out["c_absmean"] = float(gc.abs().mean())
    if q and c and out["c_absmean"] > 0:
        out["ratio"] = out["q_absmean"] / out["c_absmean"]
    return out


# ----------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, loader, horizons, device):
    """One-step error plus open-loop rollout, where the model consumes its own
    predictions — the regime a world model is actually used in."""
    model.eval()
    errs = {k: [] for k in horizons}
    for oc, ac, o_next, o_fut, a_fut in loader:
        oc, ac = oc.to(device), ac.to(device)
        o_fut, a_fut = o_fut.to(device), a_fut.to(device)
        wo, wa, preds = oc.clone(), ac.clone(), []
        for k in range(max(horizons)):
            pred = model(wo, wa)
            preds.append(pred)
            wo = torch.cat([wo[:, 1:], pred.unsqueeze(1)], 1)
            wa = torch.cat([wa[:, 1:], a_fut[:, k:k + 1]], 1)
        preds = torch.stack(preds, 1)
        for k in horizons:
            e = ((preds[:, :k] - o_fut[:, :k]) ** 2).mean(dim=(1, 2))
            errs[k].append(e.cpu().numpy())
    return {k: float(np.concatenate(v).mean()) for k, v in errs.items()}


# ------------------------------------------------------------------- training
def train_one(name, cfg, dl, info, core, verbose=True):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = build_model(name, info["obs_dim"], info["action_dim"], cfg, core).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    run = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       cfg.run_dir, f"{name}_{cfg.tag()}")
    os.makedirs(run, exist_ok=True)
    cfg.save(os.path.join(run, "config.json"))

    history, grad_log = [], []
    best, best_state, since, step = float("inf"), None, 0, 0
    t0 = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        tot, n = 0.0, 0
        for oc, ac, o_next, _, _ in dl["train"]:
            oc, ac, o_next = oc.to(cfg.device), ac.to(cfg.device), o_next.to(cfg.device)
            opt.zero_grad()
            loss = loss_fn(model(oc, ac), o_next)
            loss.backward()
            if cfg.log_grad_var and step % cfg.grad_var_every == 0:
                gs = grad_stats(model)
                if gs:
                    grad_log.append({"step": step, "epoch": epoch, **gs})
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot += loss.item() * len(oc); n += len(oc); step += 1

        val = evaluate(model, dl["val"], (1,), cfg.device)[1]
        history.append({"epoch": epoch, "train": tot / max(n, 1), "val_1step": val})
        if verbose:
            print(f"  [{name}] epoch {epoch+1:>3d}/{cfg.epochs} "
                  f"train {tot/max(n,1):.5f}  val {val:.5f}")
        if val < best - 1e-12:
            best, since = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save({"model": best_state, "epoch": epoch}, os.path.join(run, "best.pt"))
        else:
            since += 1
            if since >= cfg.early_stop_patience:
                if verbose:
                    print(f"  [{name}] early stop at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    test = evaluate(model, dl["test"], cfg.rollout_horizons, cfg.device)
    b = info["baselines"]
    result = {
        "model": name, "n_spins": cfg.n_spins, "observables": cfg.observables,
        "seed": cfg.seed, "params": n_params,
        "quantum_params": getattr(model, "n_quantum_params", 0),
        "obs_dim": info["obs_dim"], "d_model": cfg.d_model,
        "n_qubits": cfg.n_qubits, "q_layers": cfg.q_layers,
        "entangler": cfg.entangler, "context": cfg.context,
        "best_val_1step": best,
        "test_rollout": {str(k): v for k, v in test.items()},
        # every error expressed against the trivial predictors, so a number is
        # never reported without a scale
        "vs_persistence": b["persistence"] / max(test[1], 1e-12),
        "r_squared": 1.0 - test[1] / max(b["variance"], 1e-12),
        "baselines": b,
        "wall_clock_s": time.time() - t0, "run_dir": run,
    }
    for f, d in (("history.json", history), ("grad_log.json", grad_log),
                 ("result.json", result)):
        with open(os.path.join(run, f), "w") as fh:
            json.dump(d, fh, indent=2)
    if verbose:
        roll = "  ".join(f"k={k}:{v:.5f}" for k, v in test.items())
        print(f"  [{name}] test  {roll}")
        print(f"  [{name}] {result['vs_persistence']:.1f}x persistence, "
              f"R2={result['r_squared']:.4f}  "
              f"({n_params} params, {result['quantum_params']} quantum, "
              f"{result['wall_clock_s']:.0f}s)")
    return result


def main():
    cfg = Config.from_cli()
    core = load_models(cfg.phase1_root)
    print(f"[phase3] {cfg.tag()}")
    dl, info = loaders(cfg)
    print(f"[phase3] {info['system']}\n")
    results = [train_one(m, cfg, dl, info, core) for m in cfg.models]

    q = next((r for r in results if r["model"] == "quantum_transformer"), None)
    c = next((r for r in results if r["model"] == "classical_transformer"), None)
    if q and c:
        print(f"\n[compare] n_spins={cfg.n_spins}, seed={cfg.seed}")
        for k in map(str, cfg.rollout_horizons):
            ratio = q["test_rollout"][k] / max(c["test_rollout"][k], 1e-12)
            better = "quantum" if ratio < 1 else "classical"
            print(f"  k={k:>2}: quantum {q['test_rollout'][k]:.5f}  "
                  f"classical {c['test_rollout'][k]:.5f}  "
                  f"ratio {ratio:.3f} ({better} ahead)")
        print("[compare] One seed at one system size proves nothing. The "
              "experiment is the SCALING — run several sizes and seeds, then "
              "analyze.py.")


if __name__ == "__main__":
    main()
