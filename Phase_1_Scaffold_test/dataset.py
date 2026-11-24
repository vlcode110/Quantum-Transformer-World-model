"""Dataset construction: raw trajectories -> windowed next-latent prediction.

Responsibilities
  1. latent representation ('compressed' autoencoder | 'raw' categorical)
  2. EPISODE-LEVEL train/val/test splitting (no window may straddle a split)
  3. windowing into (z_{t-K+1..t}, a_{t-K+1..t}) -> z_{t+1}
  4. versioning: every artefact records the trajectory hash + the config that
     produced it, so a result can always be traced to its data.

Usage:
    python dataset.py                       # build from newest trajectory file
    python dataset.py --latent_mode raw
"""
import glob
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import Config


# --------------------------------------------------------------- autoencoder
class LatentAE(nn.Module):
    """Compresses Dreamer's [deter|stoch] feature to `latent_dim` continuous
    dims. DECISION (config.latent_mode='compressed'): keeps the quantum core's
    input width at one-feature-per-qubit. Swap for `raw` to skip it entirely."""

    def __init__(self, in_dim, latent_dim, hidden=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, latent_dim), nn.Tanh())   # bounded -> clean angles
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, in_dim))

    def forward(self, x):
        z = self.enc(x)
        return z, self.dec(z)


def train_autoencoder(feats, cfg, train_idx, verbose=True, deter_dim=None):
    """Fit ONLY on training episodes — fitting on everything leaks test
    information into the representation."""
    torch.manual_seed(cfg.seed)
    ae = LatentAE(feats.shape[1], cfg.latent_dim, cfg.ae_hidden).to(cfg.device)
    opt = torch.optim.Adam(ae.parameters(), lr=cfg.ae_lr)
    X = torch.as_tensor(feats[train_idx], device=cfg.device)
    for ep in range(cfg.ae_epochs):
        perm = torch.randperm(len(X), device=cfg.device)
        tot = 0.0
        for i in range(0, len(X), 256):
            xb = X[perm[i:i + 256]]
            opt.zero_grad()
            _, rec = ae(xb)
            loss = ((rec - xb) ** 2).mean()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        if verbose and (ep + 1) % 10 == 0:
            print(f"    AE epoch {ep+1}/{cfg.ae_epochs} recon {tot/len(X):.5f}")
    ae.eval()
    # Raw reconstruction MSE is uninterpretable on its own: half of the feature
    # vector is one-hot categorical, whose per-element variance is small, so a
    # "low" MSE can still mean the bottleneck discarded nearly everything.
    # Explained variance (1 - MSE/Var) is the number to judge, reported
    # separately for the deterministic and stochastic halves because they carry
    # different kinds of information.
    with torch.no_grad():
        X = torch.as_tensor(feats, device=cfg.device)
        rec = ae(X)[1]
        se = (rec - X) ** 2
        var = X.var(dim=0, unbiased=False)
        stats = {"recon_mse": float(se.mean()),
                 "explained_var": float(1 - se.mean() / (var.mean() + 1e-12))}
        if deter_dim is not None:
            for name, sl in (("deter", slice(0, deter_dim)),
                             ("stoch", slice(deter_dim, X.shape[1]))):
                m = float(se[:, sl].mean())
                v = float(var[sl].mean())
                stats[f"{name}_recon_mse"] = m
                stats[f"{name}_explained_var"] = float(1 - m / (v + 1e-12))
    return ae, stats


# ------------------------------------------------------------------ windowing
class LatentWindows(Dataset):
    """(context latents, context actions) -> next latent.

    Windows never cross an episode boundary, and `starts` are precomputed so
    __getitem__ stays trivial."""

    def __init__(self, z, a, episodes, context, horizon_max=1):
        self.z, self.a, self.context = z, a, context
        starts = []
        ep = np.asarray(episodes)
        for e in np.unique(ep):
            idx = np.nonzero(ep == e)[0]
            # need `context` history plus `horizon_max` future targets
            last = len(idx) - context - horizon_max
            starts.extend(idx[0] + np.arange(max(0, last + 1)))
        self.starts = np.asarray(starts, np.int64)
        self.horizon_max = horizon_max

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        c = self.context
        return (torch.as_tensor(self.z[s:s + c]),
                torch.as_tensor(self.a[s:s + c]),
                torch.as_tensor(self.z[s + c]),
                # future latents/actions for k-step open-loop evaluation
                torch.as_tensor(self.z[s + c:s + c + self.horizon_max]),
                torch.as_tensor(self.a[s + c:s + c + self.horizon_max]))


# ------------------------------------------------------------------- building
def newest_trajectory(cfg):
    pat = os.path.join(os.path.abspath(cfg.data_dir), f"traj_{cfg.task}_*.npz")
    files = sorted(glob.glob(pat), key=os.path.getmtime)
    if not files:
        raise SystemExit(
            f"[dataset] no trajectories at {pat}\n"
            f"          run: python collect_trajectories.py --task {cfg.task}")
    return files[-1]


def episode_split(episodes, fracs, seed):
    """Split by EPISODE id. Windowing inside an episode means a timestep-level
    split would put frame t in train and t+1 in test — silent leakage that
    makes every model look good."""
    uniq = np.unique(episodes)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(fracs[0] * n)
    n_va = int((fracs[0] + fracs[1]) * n)
    sets = (uniq[:n_tr], uniq[n_tr:n_va], uniq[n_va:])
    return [np.isin(episodes, s) for s in sets]


def cache_key(cfg, traj_version):
    """Identifies a built dataset by everything that affects its CONTENT.

    Model hyperparameters are deliberately excluded: every model in a
    comparison must see byte-identical latents, or the comparison is across
    different prediction targets rather than across architectures.
    """
    blob = json.dumps({
        "traj": traj_version, "latent_mode": cfg.latent_mode,
        "latent_dim": cfg.latent_dim, "ae_epochs": cfg.ae_epochs,
        "ae_hidden": cfg.ae_hidden, "ae_lr": cfg.ae_lr,
        "split_fracs": list(cfg.split_fracs),
        "split_by_episode": cfg.split_by_episode, "seed": cfg.seed,
    }, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def build(cfg, verbose=True):
    traj_path = newest_trajectory(cfg)
    d = np.load(traj_path)
    with open(traj_path.replace(".npz", ".json")) as f:
        traj_meta = json.load(f)
    episodes = d["episode"]

    # ---- cache: skip autoencoder training when nothing content-relevant moved
    ck = cache_key(cfg, traj_meta["version"])
    cache_path = os.path.join(os.path.abspath(cfg.data_dir), f"built_{ck}.npz")
    info_path = cache_path.replace(".npz", ".json")
    if os.path.exists(cache_path) and os.path.exists(info_path):
        c = np.load(cache_path)
        with open(info_path) as f:
            info = json.load(f)
        horizon = max(cfg.rollout_horizons)
        splits = {}
        for name in ("train", "val", "test"):
            m = c[f"mask_{name}"]
            splits[name] = LatentWindows(c["z"][m], c["a"][m], episodes[m],
                                         cfg.context, horizon)
        if verbose:
            print(f"[dataset] cache hit {ck} — AE reused "
                  f"(deter explained var {info.get('ae_deter_explained_var', 0):.1%})")
        # windows depend on context/horizon, which are NOT in the cache key
        info["n_windows"] = {k: len(v) for k, v in splits.items()}
        info["context"], info["horizon_max"] = cfg.context, horizon
        return splits, info

    tr_mask, va_mask, te_mask = episode_split(
        episodes, cfg.split_fracs, cfg.seed) if cfg.split_by_episode else None

    feats = np.concatenate([d["deter"], d["stoch"]], axis=1)  # (N, deter+stoch)
    info = {"trajectory": os.path.basename(traj_path),
            "trajectory_version": traj_meta["version"],
            "latent_mode": cfg.latent_mode}

    if cfg.latent_mode == "compressed":
        if verbose:
            print(f"[dataset] fitting autoencoder {feats.shape[1]} -> "
                  f"{cfg.latent_dim} on training episodes only")
        ae, ae_stats = train_autoencoder(feats, cfg, np.nonzero(tr_mask)[0],
                                         verbose, deter_dim=d["deter"].shape[1])
        with torch.no_grad():
            z = ae.enc(torch.as_tensor(feats, device=cfg.device)).cpu().numpy()
        info.update({f"ae_{k}": v for k, v in ae_stats.items()})
        info["latent_dim"] = cfg.latent_dim
        recon = ae_stats["recon_mse"]
        # keep the AE so rollouts can be decoded back to Dreamer features
        ae_path = os.path.join(os.path.abspath(cfg.data_dir),
                               f"ae_{cfg.task}_{traj_meta['version']}"
                               f"_d{cfg.latent_dim}.pt")
        torch.save(ae.state_dict(), ae_path)
        info["autoencoder"] = os.path.basename(ae_path)
        if verbose:
            ev = ae_stats["explained_var"]
            print(f"[dataset] AE recon MSE {recon:.5f}  explained variance "
                  f"{ev:.1%}  (deter {ae_stats.get('deter_explained_var', 0):.1%}, "
                  f"stoch {ae_stats.get('stoch_explained_var', 0):.1%})")
            if ae_stats.get("deter_explained_var", 1) < 0.8:
                print(f"[dataset] WARNING: the DETERMINISTIC state is poorly "
                      f"reconstructed. That half carries the smooth dynamics, "
                      f"so a {cfg.latent_dim}-dim bottleneck is probably too "
                      f"narrow — raise latent_dim (and n_qubits with it) or "
                      f"train the AE longer. Low stoch explained variance is "
                      f"expected and not a problem: those are sampled "
                      f"one-hots and partly irreducible noise.")
            print(f"[dataset] AE -> {ae_path}")
    elif cfg.latent_mode == "raw":
        # DECISION: predict Dreamer's categorical stochastic latent directly.
        # Faithful to DreamerV3, but the readout head carries most of the
        # dimensionality — note this when interpreting quantum contribution.
        z = d["stoch"].astype(np.float32)
        info["latent_dim"] = z.shape[1]
    else:
        raise SystemExit(f"[dataset] unknown latent_mode {cfg.latent_mode!r}")

    a = d["action"].astype(np.float32)
    z = z.astype(np.float32)
    horizon = max(cfg.rollout_horizons)

    splits = {}
    for name, mask in zip(("train", "val", "test"), (tr_mask, va_mask, te_mask)):
        splits[name] = LatentWindows(z[mask], a[mask], episodes[mask],
                                     cfg.context, horizon)
    info.update({
        "n_windows": {k: len(v) for k, v in splits.items()},
        "n_episodes": {k: int(len(np.unique(episodes[m])))
                       for k, m in zip(("train", "val", "test"),
                                       (tr_mask, va_mask, te_mask))},
        "context": cfg.context, "horizon_max": horizon,
        "action_dim": int(a.shape[1]), "z_dim": int(z.shape[1]),
    })
    if verbose:
        print(f"[dataset] windows: {info['n_windows']}  "
              f"episodes: {info['n_episodes']}")

    np.savez_compressed(cache_path, z=z, a=a, mask_train=tr_mask,
                        mask_val=va_mask, mask_test=te_mask)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    if verbose:
        print(f"[dataset] cached as {ck} — subsequent runs skip AE training")
    return splits, info


def loaders(cfg, verbose=True):
    splits, info = build(cfg, verbose)
    dl = {k: DataLoader(v, batch_size=cfg.batch_size,
                        shuffle=(k == "train"), drop_last=(k == "train"))
          for k, v in splits.items()}
    return dl, info


if __name__ == "__main__":
    cfg = Config.from_cli()
    _, info = build(cfg)
    print(json.dumps(info, indent=2))
