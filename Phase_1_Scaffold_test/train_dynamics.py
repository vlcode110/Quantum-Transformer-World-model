"""Train one dynamics model on offline latent trajectories.

Reports the two numbers the Phase 1 gate is written against:
  * 1-step prediction error on held-out episodes
  * k-step OPEN-LOOP rollout error (model re-consumes its own predictions)
and logs gradient variance so the barren-plateau baseline from Phase 0 extends
through Phase 1 rather than being a one-off measurement.

Usage:
    python train_dynamics.py --models quantum_transformer
    python train_dynamics.py --models classical_transformer --epochs 20
"""
import json
import os
import time
from dataclasses import fields

import numpy as np
import torch
import torch.nn as nn

from config import Config
from dataset import loaders
from models import build_model, IS_QUANTUM


# ------------------------------------------------------------ durable writes
def save_with_retry(obj, path, attempts=6, torch_save=True):
    """Write to iCloud-backed storage, retrying transient refusals.

    This project lives under ~/Documents, which is iCloud Drive. iCloud
    intermittently refuses opens with "cannot be opened" or EDEADLK while it
    is reconciling a file. A twelve-model sweep lost its third seed that way:
    four models had trained, the fourth could not write best.pt, and the
    whole process died with a traceback after six hours of compute.

    Retrying with a short backoff clears it. If it still fails, write beside
    the target under a .partial name and carry on rather than killing the
    sweep, since a missing checkpoint is recoverable and a lost sweep is not.
    """
    import shutil
    last = None
    for i in range(attempts):
        try:
            if torch_save:
                torch.save(obj, path)
            else:
                with open(path, "w") as f:
                    json.dump(obj, f, indent=2)
            return True
        except (OSError, RuntimeError) as e:
            last = e
            time.sleep(0.5 * (i + 1))
    fallback = os.path.join("/tmp", "qtwm_rescue",
                            os.path.basename(os.path.dirname(path)))
    try:
        os.makedirs(fallback, exist_ok=True)
        tmp = os.path.join(fallback, os.path.basename(path))
        if torch_save:
            torch.save(obj, tmp)
        else:
            with open(tmp, "w") as f:
                json.dump(obj, f, indent=2)
        try:
            shutil.copy2(tmp, path)
            print(f"  [io] recovered {path} via /tmp")
            return True
        except Exception:
            print(f"  [io] COULD NOT WRITE {path} ({type(last).__name__}); "
                  f"kept at {tmp} — copy it back before analysing")
            return False
    except Exception:
        print(f"  [io] COULD NOT WRITE {path} ({type(last).__name__}) and the "
              f"/tmp fallback also failed; continuing without it")
        return False


# --------------------------------------------------- gradient instrumentation
def grad_stats(model):
    """What we log, and why:
      q_var / q_absmean : variance and magnitude of gradients on QUANTUM
                          (circuit) parameters — the barren-plateau signal.
                          Compare its trajectory against the Phase 0 baseline.
      c_var             : the same for classical parameters, as a control: if
                          both collapse, the problem is the loss or the data,
                          not the circuit.
      ratio             : q_absmean / c_absmean — a dimensionless "is the
                          quantum part still learning?" indicator that is
                          comparable across model sizes.
    """
    q, c = [], []
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().flatten()
        (q if ("q." in n or "gates." in n) and "weights" in n else c).append(g)
    out = {}
    if q:
        gq = torch.cat(q)
        out["q_var"] = float(gq.var()); out["q_absmean"] = float(gq.abs().mean())
    if c:
        gc = torch.cat(c)
        out["c_var"] = float(gc.var()); out["c_absmean"] = float(gc.abs().mean())
    if q and c and out["c_absmean"] > 0:
        out["ratio"] = out["q_absmean"] / out["c_absmean"]
    return out


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate(model, loader, horizons, device):
    """1-step error plus open-loop k-step error.

    Open-loop means the model consumes ITS OWN predictions — the regime a world
    model is actually used in. Teacher-forced error looks far better and is not
    what the gate asks for."""
    model.eval()
    errs = {k: [] for k in horizons}
    for zc, ac, z_next, z_fut, a_fut in loader:
        zc, ac = zc.to(device), ac.to(device)
        z_fut, a_fut = z_fut.to(device), a_fut.to(device)
        window_z, window_a = zc.clone(), ac.clone()
        preds = []
        for k in range(max(horizons)):
            pred = model(window_z, window_a)
            preds.append(pred)
            # slide the window, feeding the prediction back in
            window_z = torch.cat([window_z[:, 1:], pred.unsqueeze(1)], 1)
            window_a = torch.cat([window_a[:, 1:], a_fut[:, k:k + 1]], 1)
        preds = torch.stack(preds, 1)                       # (B, H, z)
        for k in horizons:
            e = ((preds[:, :k] - z_fut[:, :k]) ** 2).mean(dim=(1, 2))
            errs[k].append(e.cpu().numpy())
    return {k: float(np.concatenate(v).mean()) for k, v in errs.items()}


# -------------------------------------------------------------------- training
def train_one(name, cfg, dl, info, verbose=True):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = build_model(name, info["z_dim"], info["action_dim"], cfg).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    run = os.path.join(os.path.abspath(cfg.run_dir), f"{name}_{cfg.tag()}")
    # Resumability. A sweep of 58 runs will be interrupted — by a crash, by
    # the machine sleeping, by the user. Skipping anything already finished
    # makes re-running the driver script free, so the correct response to an
    # interruption is simply to start it again.
    done_path = os.path.join(run, "result.json")
    if getattr(cfg, "skip_existing", False) and os.path.exists(done_path):
        try:
            prev = json.load(open(done_path))
            if verbose:
                print(f"  [{name}] already complete, skipping "
                      f"(k=1:{prev['test_rollout']['1']:.5f})")
            return prev
        except Exception:
            if verbose:
                print(f"  [{name}] result.json unreadable, re-running")
    os.makedirs(run, exist_ok=True)
    cfg.save(os.path.join(run, "config.json"))
    with open(os.path.join(run, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    history, grad_log = [], []
    best_val, best_state, since_best, step = float("inf"), None, 0, 0
    t0 = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        tot, n = 0.0, 0
        for zc, ac, z_next, _, _ in dl["train"]:
            zc, ac, z_next = zc.to(cfg.device), ac.to(cfg.device), z_next.to(cfg.device)
            opt.zero_grad()
            loss = loss_fn(model(zc, ac), z_next)
            loss.backward()
            if cfg.log_grad_var and step % cfg.grad_var_every == 0:
                gs = grad_stats(model)
                if gs:
                    grad_log.append({"step": step, "epoch": epoch, **gs})
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot += loss.item() * len(zc); n += len(zc); step += 1

        val = evaluate(model, dl["val"], (1,), cfg.device)[1]
        history.append({"epoch": epoch, "train_mse": tot / max(n, 1), "val_1step": val})
        if verbose:
            print(f"  [{name}] epoch {epoch+1:>3d}/{cfg.epochs} "
                  f"train {tot/max(n,1):.5f}  val_1step {val:.5f}")
        if val < best_val - 1e-9:
            best_val, since_best = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            save_with_retry({"model": best_state, "epoch": epoch, "val": val},
                            os.path.join(run, "best.pt"))
        else:
            since_best += 1
            if since_best >= cfg.early_stop_patience:
                if verbose:
                    print(f"  [{name}] early stop at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    test = evaluate(model, dl["test"], cfg.rollout_horizons, cfg.device)
    result = {
        "model": name, "task": cfg.task, "seed": cfg.seed,
        "params": n_params,
        "quantum_params": getattr(model, "n_quantum_params", 0),
        "is_quantum": IS_QUANTUM.get(name, False),
        "best_val_1step": best_val,
        "test_rollout": {str(k): v for k, v in test.items()},
        "n_qubits": cfg.n_qubits, "q_layers": cfg.q_layers,
        "encoding": cfg.encoding, "readout": cfg.readout,
        "entangler": getattr(cfg, "entangler", "brickwork"),
        "d_model": cfg.d_model, "n_layers": cfg.n_layers,
        "latent_mode": cfg.latent_mode, "latent_dim": info["latent_dim"],
        "context": cfg.context,
        "dataset_version": info["trajectory_version"],
        "wall_clock_s": time.time() - t0,
        "run_dir": run,
    }
    save_with_retry(history, os.path.join(run, "history.json"),
                    torch_save=False)
    save_with_retry(grad_log, os.path.join(run, "grad_log.json"),
                    torch_save=False)
    # result.json last, so its presence always means the run really finished
    # and --skip_existing can trust it.
    save_with_retry(result, os.path.join(run, "result.json"),
                    torch_save=False)
    if verbose:
        roll = "  ".join(f"k={k}:{v:.5f}" for k, v in test.items())
        print(f"  [{name}] test  {roll}   ({n_params} params, "
              f"{result['quantum_params']} quantum, "
              f"{result['wall_clock_s']:.0f}s)")
    return result


# Fields that tag() actually encodes. Anything NOT here is invisible in the
# directory name, so two runs differing only in such a field would collide.
TAGGED_FIELDS = {
    "task", "latent_mode", "latent_dim", "n_qubits", "q_layers", "encoding",
    "readout", "entangler", "context", "d_model", "n_layers", "seed",
}
# Fields that cannot affect a result and so need no guard.
BENIGN_FIELDS = {
    "dreamer_root", "phase0_root", "data_dir", "run_dir", "device", "models",
    "rollout_horizons", "gate_k", "gate_ratio", "log_grad_var",
    "grad_var_every", "skip_existing", "allow_untagged",
}


def guard_untagged(cfg):
    """Refuse to run if a result-affecting field is absent from tag().

    Two completed 80,000-step runs in Phase 2 were destroyed because a tag
    omitted one setting, so an ablation and a headline run resolved to the
    same directory. Both finished normally and left plausible files behind;
    the loss was invisible until the numbers were checked against their
    timestamps. The same hole exists here: tag() does not encode n_heads,
    quantum_layer_position, action_conditioning, epochs, or any dataset
    setting. Rather than change tag() -- which would orphan every existing
    run directory -- refuse to start when one of those is off its default,
    and say exactly which. Override with --allow_untagged true only after
    choosing a distinguishing --run_dir.
    """
    ref = Config()
    offenders = []
    for f in fields(Config):
        if f.name in TAGGED_FIELDS or f.name in BENIGN_FIELDS:
            continue
        a, b = getattr(cfg, f.name), getattr(ref, f.name)
        if a != b:
            offenders.append(f"{f.name}={a!r} (default {b!r})")
    if offenders and not getattr(cfg, "allow_untagged", False):
        raise SystemExit(
            "[guard] REFUSING TO RUN.\n"
            "  These settings differ from their defaults but do NOT appear in\n"
            "  the run directory name, so this run would be indistinguishable\n"
            "  from a default-settings run at the same seed and could overwrite\n"
            "  it:\n    " + "\n    ".join(offenders) + "\n"
            "  Either revert them, or pass a distinct --run_dir together with\n"
            "  --allow_untagged true.")
    if offenders:
        print("[guard] proceeding with untagged non-defaults: "
              + ", ".join(offenders))


def main():
    cfg = Config.from_cli()
    guard_untagged(cfg)
    print(f"[train] {cfg.tag()}")
    dl, info = loaders(cfg)
    results = [train_one(name, cfg, dl, info) for name in cfg.models]

    # gate check against the parameter-matched classical twin
    base = next((r for r in results if r["model"] == "classical_transformer"), None)
    quant = next((r for r in results if r["model"] == "quantum_transformer"), None)
    if base and quant:
        k = str(cfg.gate_k)
        one = quant["test_rollout"]["1"] <= base["test_rollout"]["1"]
        ratio = quant["test_rollout"][k] / max(base["test_rollout"][k], 1e-12)
        print(f"\n[gate] 1-step: quantum {quant['test_rollout']['1']:.5f} vs "
              f"classical {base['test_rollout']['1']:.5f} -> "
              f"{'PASS' if one else 'FAIL'}")
        print(f"[gate] {k}-step ratio {ratio:.2f}x (need <= {cfg.gate_ratio}) -> "
              f"{'PASS' if ratio <= cfg.gate_ratio else 'FAIL'}")
        print("[gate] NOTE: one seed is not evidence. Repeat over >=3 seeds and "
              "aggregate with analyze.py before believing either verdict.")


if __name__ == "__main__":
    main()
