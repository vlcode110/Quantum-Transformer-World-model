"""Roll out a trained Phase 0 Dreamer agent and dump latent trajectories.

Produces a versioned dataset: raw Dreamer features [deter | stoch] per step,
actions, rewards, and episode boundaries. Nothing is windowed or split here —
that happens in dataset.py, so re-splitting never requires re-collecting.

Usage:
    python collect_trajectories.py                       # cartpole, 200 episodes
    python collect_trajectories.py --task pendulum       # BLOCKED until trained
    python collect_trajectories.py --n_episodes 50       # quick pass
"""
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

from config import Config


def _load_module_from_file(path, module_name):
    """Load a module by explicit path under a unique name.

    Necessary because this project and the DreamerV3 project BOTH define a
    top-level `config.py`. A plain `import config` after a sys.path tweak
    returns whichever one is already in sys.modules — ours — and fails with a
    confusing AttributeError. Loading by file path sidesteps the name entirely.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        sys.exit(f"[collect] could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _import_dreamer(dreamer_root):
    """Import the Phase 0 package without copying it (single source of truth)."""
    root = os.path.abspath(dreamer_root)
    if not os.path.isdir(root):
        sys.exit(f"[collect] Dreamer root not found: {root}\n"
                 f"          Fix `dreamer_root` in config.py.")
    cfg_path = os.path.join(root, "config.py")
    if not os.path.exists(cfg_path):
        sys.exit(f"[collect] {cfg_path} not found — is `dreamer_root` right?")
    sys.path.insert(0, root)
    try:
        from dreamer.agent import Dreamer            # package name is unique
    except ImportError as e:
        sys.exit(f"[collect] could not import Dreamer from {root}: {e}")
    finally:
        sys.path.pop(0)
    dreamer_config = _load_module_from_file(cfg_path, "_dreamer_task_config")
    return Dreamer, dreamer_config


def load_agent(cfg):
    Dreamer, dcfg_mod = _import_dreamer(cfg.dreamer_root)
    ckpt = os.path.join(os.path.abspath(cfg.dreamer_root), "runs",
                        f"{cfg.task}_seed{cfg.seed}", "checkpoint.pt")
    if not os.path.exists(ckpt):
        sys.exit(f"[collect] no checkpoint at {ckpt}\n"
                 f"          {cfg.task} has not finished training yet — this is "
                 f"the expected block for pendulum.")
    import gymnasium as gym
    dcfg = dcfg_mod.get_config(cfg.task)
    env = gym.make(dcfg["env_id"])
    obs_dim = env.observation_space.shape[0]
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    action_dim = env.action_space.n if discrete else env.action_space.shape[0]
    agent = Dreamer(obs_dim, action_dim, discrete, dcfg, cfg.device)
    blob = torch.load(ckpt, map_location=cfg.device, weights_only=False)
    # full training checkpoints wrap the weights; legacy ones are a bare dict
    agent.load_state_dict(blob["model"] if "model" in blob else blob)
    if "step" in blob:
        print(f"[collect] checkpoint from step {blob['step']}")
    agent.eval()
    return agent, env, discrete, action_dim, dcfg


def to_env_action(action, env, discrete):
    if discrete:
        return int(np.argmax(action))
    low, high = env.action_space.low, env.action_space.high
    return low + (action + 1.0) * 0.5 * (high - low)


@torch.no_grad()
def collect(cfg):
    agent, env, discrete, action_dim, dcfg = load_agent(cfg)
    rng = np.random.default_rng(cfg.seed)

    deters, stochs, actions, rewards, ep_ids, is_firsts, terminals = \
        [], [], [], [], [], [], []
    returns = []

    for ep in range(cfg.n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1e9)))
        state, done, total, first = None, False, 0.0, True
        while not done:
            action, state = agent.policy(obs, state,
                                         mode=(cfg.policy_mode == "greedy"))
            # DECISION (config.policy_mode): exploration noise widens the state
            # manifold the dynamics model must cover. Pure greedy trajectories
            # are easy for every model and hide differences between them.
            if cfg.policy_mode == "mixed" and rng.random() < cfg.epsilon:
                if discrete:
                    action = np.eye(action_dim, dtype=np.float32)[
                        rng.integers(action_dim)]
                else:
                    action = rng.uniform(-1, 1, action_dim).astype(np.float32)
                # CRITICAL: the RSSM's next step conditions on the PREVIOUS
                # action, which the policy stashed in `state`. Overriding the
                # executed action without rewriting it there desynchronises the
                # latent state from the environment and the policy collapses
                # toward random play.
                deter, stoch, _ = state
                state = (deter, stoch,
                         torch.as_tensor(action, dtype=torch.float32,
                                         device=cfg.device).unsqueeze(0))
            deter, stoch, _ = state
            deters.append(deter.squeeze(0).cpu().numpy())
            stochs.append(stoch.squeeze(0).cpu().numpy())
            actions.append(np.asarray(action, np.float32))
            ep_ids.append(ep)
            is_firsts.append(first)
            first = False

            obs, r, term, trunc, _ = env.step(to_env_action(action, env, discrete))
            rewards.append(np.float32(r))
            terminals.append(bool(term))
            total += r
            done = term or trunc
        returns.append(total)
        if (ep + 1) % 20 == 0:
            print(f"  episode {ep+1}/{cfg.n_episodes}  "
                  f"mean return {np.mean(returns):.1f}  steps {len(ep_ids)}")

    env.close()
    data = {
        "deter": np.asarray(deters, np.float32),
        "stoch": np.asarray(stochs, np.float32),
        "action": np.asarray(actions, np.float32),
        "reward": np.asarray(rewards, np.float32),
        "episode": np.asarray(ep_ids, np.int32),
        "is_first": np.asarray(is_firsts, bool),
        "is_terminal": np.asarray(terminals, bool),
    }
    # Cast every numpy scalar to a Python type: gymnasium returns action_space.n
    # as numpy.int64, which json.dump refuses to serialise.
    meta = {
        "task": cfg.task, "n_episodes": int(cfg.n_episodes),
        "policy_mode": cfg.policy_mode, "epsilon": float(cfg.epsilon),
        "seed": int(cfg.seed), "discrete": bool(discrete),
        "action_dim": int(action_dim),
        "stoch_shape": [int(dcfg["stoch"]), int(dcfg["classes"])],
        "deter_dim": int(dcfg["deter"]),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "n_steps": int(len(ep_ids)),
        "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_checkpoint": os.path.join(cfg.dreamer_root, "runs",
                                          f"{cfg.task}_seed{cfg.seed}",
                                          "checkpoint.pt"),
    }
    return data, meta


def version_hash(meta):
    """Content hash over the generating parameters — datasets are immutable and
    identified by how they were made, so results can always be traced back."""
    keys = ["task", "n_episodes", "policy_mode", "epsilon", "seed"]
    blob = json.dumps({k: meta[k] for k in keys}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def main():
    cfg = Config.from_cli()
    print(f"[collect] task={cfg.task} episodes={cfg.n_episodes} "
          f"policy={cfg.policy_mode}")
    data, meta = collect(cfg)
    vh = version_hash(meta)
    meta["version"] = vh
    out_dir = os.path.abspath(cfg.data_dir)
    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, f"traj_{cfg.task}_{vh}.npz")
    np.savez_compressed(npz, **data)
    with open(npz.replace(".npz", ".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[collect] {meta['n_steps']} steps from {cfg.n_episodes} episodes")
    print(f"[collect] mean return {meta['mean_return']:.1f} "
          f"± {meta['std_return']:.1f}")
    # Sanity check: exploration should degrade the policy, not destroy it. A
    # near-random return means the agent is not actually driving the data, and
    # the resulting "dynamics" dataset would mostly describe failure modes.
    # (task -> approximate random-policy return, threshold to warn below)
    RANDOM_BASELINE = {"cartpole": (22.0, 100.0), "pendulum": (-1200.0, -600.0)}
    if cfg.task in RANDOM_BASELINE:
        base, floor = RANDOM_BASELINE[cfg.task]
        if meta["mean_return"] < floor:
            print(f"[collect] WARNING: mean return {meta['mean_return']:.1f} is "
                  f"near random play (~{base:g}). A trained agent with "
                  f"{cfg.epsilon:.0%} exploration should score far higher. "
                  f"Check the checkpoint and consider lowering --epsilon "
                  f"before trusting this dataset.")
    print(f"[collect] saved -> {npz}")
    print(f"[collect] version {vh}  (referenced by every downstream run)")


if __name__ == "__main__":
    main()
