"""Turning spin-chain trajectories into a prediction dataset.

Much simpler than Phase 1's equivalent, because there is no compression step:
the recorded observables ARE the state the model predicts. That removes the
autoencoder and with it a whole class of doubt about whether compression was
hiding the effect.

Two lessons from Phase 1 are built in:
  * windows never cross a trajectory boundary, and the split is by whole
    trajectory, so no training window can overlap a test window
  * the split is FIXED across seeds by default, so seeds vary only model
    initialisation. Phase 1 varied the partition too, which inflated
    seed-to-seed variance and cost statistical power
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import Config
import spin_env


class Windows(Dataset):
    """(history of observations, history of controls) -> next observation."""

    def __init__(self, obs, act, context, horizon_max):
        self.obs, self.act = obs, act           # (n_traj, T, d), (n_traj, T, a)
        self.context, self.horizon_max = context, horizon_max
        n_traj, T = obs.shape[0], obs.shape[1]
        last = T - context - horizon_max
        if last < 0:
            raise ValueError(
                f"trajectories of length {T} are too short for context "
                f"{context} plus horizon {horizon_max}")
        self.index = [(i, s) for i in range(n_traj) for s in range(last + 1)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, s = self.index[k]
        c, h = self.context, self.horizon_max
        return (torch.from_numpy(self.obs[i, s:s + c]),
                torch.from_numpy(self.act[i, s:s + c]),
                torch.from_numpy(self.obs[i, s + c]),
                torch.from_numpy(self.obs[i, s + c:s + c + h]),
                torch.from_numpy(self.act[i, s + c:s + c + h]))


def path_for(cfg):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        cfg.data_dir, cfg.data_tag() + ".npz")


def ensure_data(cfg, verbose=True):
    """Generate the trajectories if they are not already on disk.

    Data generation is cheap and fully determined by the physical settings, so
    it is cached by a tag encoding those settings — a different chain length or
    time step produces a different file rather than silently reusing the wrong
    one."""
    path = path_for(cfg)
    if os.path.exists(path):
        d = np.load(path)
        if verbose:
            print(f"[data] reusing {os.path.basename(path)}")
        return d["obs"], d["act"], d["lvl"]
    if verbose:
        print(f"[data] generating: {spin_env.describe(cfg)}")
    obs, act, lvl = spin_env.generate(cfg, verbose)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, obs=obs, act=act, lvl=lvl)
    with open(path.replace(".npz", ".json"), "w") as f:
        json.dump({"description": spin_env.describe(cfg),
                   **spin_env.diagnostics(obs, cfg)}, f, indent=2)
    if verbose:
        print(f"[data] saved {os.path.basename(path)}")
    return obs, act, lvl


def trivial_baselines(obs_test):
    """Scale references, computed on the test trajectories.

    Reported for exactly the reason Phase 1 learned the hard way: an error of
    0.003 means nothing until you know what 'predict no change' scores.
    """
    cur, nxt = obs_test[:, :-1], obs_test[:, 1:]
    persistence = float(((cur - nxt) ** 2).mean())
    prev = obs_test[:, :-2]
    linear = float(((2 * obs_test[:, 1:-1] - prev - obs_test[:, 2:]) ** 2).mean())
    flat = obs_test.reshape(-1, obs_test.shape[-1])
    mean_pred = float(((flat.mean(axis=0) - flat) ** 2).mean())
    variance = float(flat.var(axis=0).mean())
    return {"persistence": persistence, "linear": linear,
            "mean": mean_pred, "variance": variance}


def build(cfg, verbose=True):
    obs, act, lvl = ensure_data(cfg, verbose)
    n_traj = obs.shape[0]

    # split by whole trajectory; fixed across seeds unless told otherwise
    split_seed = 0 if cfg.fixed_split else cfg.seed
    order = np.random.default_rng(split_seed).permutation(n_traj)
    n_tr = int(cfg.split_fracs[0] * n_traj)
    n_va = int((cfg.split_fracs[0] + cfg.split_fracs[1]) * n_traj)
    parts = {"train": order[:n_tr], "val": order[n_tr:n_va], "test": order[n_va:]}

    horizon = max(cfg.rollout_horizons)
    splits = {k: Windows(obs[v], act[v], cfg.context, horizon)
              for k, v in parts.items()}

    info = {
        "system": spin_env.describe(cfg),
        "n_spins": cfg.n_spins, "obs_dim": int(obs.shape[-1]),
        "action_dim": int(act.shape[-1]),
        "n_trajectories": {k: int(len(v)) for k, v in parts.items()},
        "n_windows": {k: len(v) for k, v in splits.items()},
        "context": cfg.context, "horizon_max": horizon,
        "fixed_split": cfg.fixed_split,
        "baselines": trivial_baselines(obs[parts["test"]]),
        "diagnostics": spin_env.diagnostics(obs, cfg),
    }
    if verbose:
        b = info["baselines"]
        print(f"[data] {info['n_windows']} windows, "
              f"{info['n_trajectories']} trajectories")
        print(f"[data] baselines — persistence {b['persistence']:.5f}, "
              f"linear {b['linear']:.5f}, mean {b['mean']:.5f}")
        print(f"[data] persistence is {b['mean']/max(b['persistence'],1e-12):.1f}x "
              f"better than predicting the mean")
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
