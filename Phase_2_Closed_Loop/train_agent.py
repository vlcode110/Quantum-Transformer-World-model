"""Train a DreamerV3 agent whose world-model prior is a quantum circuit.

Self-contained: imports the Phase 0 Dreamer and replay buffer and the Phase 1
circuit builder, but modifies neither project.

Usage:
    python train_agent.py --quantum true                    # quantum agent
    python train_agent.py --quantum false                   # matched control
    python train_agent.py --partial_obs true                # hypothesis test
    python train_agent.py --steps 2000 --warmup_steps 300   # smoke test

Always run the control with IDENTICAL settings. The compute budget here is
reduced from DreamerV3's defaults (see config.py), so published DreamerV3
numbers are not a valid comparison.
"""
import csv
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

from config import Config, resolve_steps, TASK_GATE
from envs import make_env, describe, to_env_action
from quantum_rssm import swap_prior, prior_grad_stats


# --------------------------------------------------------------- Phase 0 import
def _load_module_from_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def import_dreamer(dreamer_root):
    """Import Dreamer and ReplayBuffer without disturbing the Phase 0 project.

    `config.py` exists in both projects, so the DreamerV3 task config is loaded
    by file path under a unique name rather than by import.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, dreamer_root))
    if not os.path.isdir(root):
        # DreamerV3 is vendored into this repository, but older local
        # checkouts kept it as a sibling of ~/Documents. Try the historical
        # locations before giving up, so neither layout needs path editing.
        for alt in ("../../../DreamerV3", "../../../../DreamerV3"):
            cand = os.path.abspath(os.path.join(here, alt))
            if os.path.isdir(cand):
                root = cand
                break
        else:
            sys.exit(f"[phase2] DreamerV3 not found at {root} nor at the "
                     f"legacy out-of-tree locations — fix `dreamer_root`.")
    cfg_path = os.path.join(root, "config.py")
    sys.path.insert(0, root)
    try:
        from dreamer.agent import Dreamer          # package name is unique
        from dreamer.buffer import ReplayBuffer
    except ImportError as e:
        sys.exit(f"[phase2] could not import Dreamer from {root}: {e}")
    finally:
        sys.path.pop(0)
    dreamer_cfg = _load_module_from_file(cfg_path, "_dreamer_task_config")
    return Dreamer, ReplayBuffer, dreamer_cfg


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate(agent, env_id, cfg, discrete, episodes, seed=0):
    env = make_env(env_id, cfg)
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        state, done, total = None, False, 0.0
        while not done:
            action, state = agent.policy(obs, state, mode=True)
            obs, r, term, trunc, _ = env.step(to_env_action(action, env, discrete))
            total += r
            done = term or trunc
        returns.append(total)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


# -------------------------------------------------------------- checkpointing
def save_checkpoint(path, agent, step, info):
    """Atomic write: temp file then rename, so an interrupted save cannot
    leave a truncated checkpoint."""
    ckpt = {"model": agent.state_dict(), "step": step, "info": info,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state()}
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


# -------------------------------------------------------------------- training

# --------------------------------------------------------- durable writes
def durable(write_fn, path, attempts=8):
    """Run write_fn(path), retrying transient iCloud refusals.

    This project lives under ~/Documents, which is iCloud Drive. iCloud
    intermittently refuses opens with ETIMEDOUT, EDEADLK or "cannot be
    opened" while reconciling a file. One 80,000-step run was lost exactly
    that way: it had SOLVED the task at 20,000 steps and died at 22,500 on
    `TimeoutError: [Errno 60]` writing grad_log.json -- eight hours of
    compute discarded because a 40 kB diagnostic file could not be opened.

    grad_log.json is written at every evaluation, so a single run makes 32
    attempts at this; over a ten-run sweep the chance of at least one
    failure is high. Retry with backoff, then fall back to /tmp and copy
    back, and only then give up -- never by raising, because losing the
    whole run to a failed auxiliary write is the worst available outcome.
    """
    import shutil
    last = None
    for i in range(attempts):
        try:
            write_fn(path)
            return True
        except (OSError, RuntimeError) as e:
            last = e
            time.sleep(0.5 * (i + 1))
    try:
        rescue = os.path.join("/tmp", "qtwm_rescue",
                              os.path.basename(os.path.dirname(path)))
        os.makedirs(rescue, exist_ok=True)
        tmp = os.path.join(rescue, os.path.basename(path))
        write_fn(tmp)
        try:
            shutil.copy2(tmp, path)
            print(f"[phase2][io] recovered {os.path.basename(path)} via /tmp")
            return True
        except Exception:
            print(f"[phase2][io] COULD NOT WRITE {path} "
                  f"({type(last).__name__}); kept at {tmp}")
            return False
    except Exception:
        print(f"[phase2][io] COULD NOT WRITE {path} ({type(last).__name__}) "
              f"and the /tmp fallback also failed; continuing")
        return False


def durable_json(obj, path):
    def _w(p):
        with open(p, "w") as f:
            json.dump(obj, f, indent=2)
    return durable(_w, path)


def main():
    cfg = Config.from_cli()
    steps = resolve_steps(cfg)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    Dreamer, ReplayBuffer, dreamer_cfg = import_dreamer(cfg.dreamer_root)
    dcfg = dreamer_cfg.get_config(cfg.task)

    # Apply the Phase 2 compute budget. These override DreamerV3's defaults and
    # are the reason the classical control must use the same settings.
    dcfg["batch_size"] = cfg.batch_size
    dcfg["seq_len"] = cfg.seq_len
    dcfg["horizon"] = cfg.horizon
    dcfg["train_every"] = cfg.train_every
    dcfg["warmup_steps"] = cfg.warmup_steps

    env_id = dcfg["env_id"]
    env = make_env(env_id, cfg, seed=cfg.seed)
    obs_dim = env.observation_space.shape[0]
    import gymnasium as gym
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    action_dim = env.action_space.n if discrete else env.action_space.shape[0]

    agent = Dreamer(obs_dim, action_dim, discrete, dcfg, cfg.device)
    agent, info = swap_prior(agent, cfg, dcfg)

    run = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       cfg.run_dir, cfg.tag())
    # Never write into a directory that already holds a finished run. A tag
    # collision once overwrote two 80,000-step runs with 40,000-step ablations
    # and left a plausible result.json behind, so the loss was invisible until
    # the file timestamps were checked. Refusing outright is the only guard
    # that does not depend on the naming scheme being correct.
    existing = os.path.join(run, "result.json")
    if os.path.exists(existing):
        try:
            with open(existing) as f:
                prior = json.load(f)
            detail = (f"{prior.get('env_steps', '?')} steps, "
                      f"final {prior.get('final_eval_return', '?')}")
        except Exception:
            detail = "unreadable"
        sys.exit(
            f"[phase2] REFUSING TO OVERWRITE\n"
            f"  {run}\n"
            f"  already contains a completed run ({detail}).\n\n"
            f"  Move or delete it first, or change a setting so the tag "
            f"differs. Nothing\n  has been modified.")
    os.makedirs(run, exist_ok=True)
    cfg.save(os.path.join(run, "config.json"))
    with open(os.path.join(run, "model_info.json"), "w") as f:
        json.dump({**info, "obs_dim": obs_dim, "action_dim": int(action_dim),
                   "env": describe(env_id, cfg)}, f, indent=2)

    csv_path = os.path.join(run, "log.csv")
    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)
    if os.stat(csv_path).st_size == 0:
        writer.writerow(["step", "eval_return", "eval_std", "train_return",
                         "wm_recon", "wm_kl", "prior_q_var", "prior_ratio"])

    print(f"[phase2] {cfg.tag()}")
    print(f"[phase2] {describe(env_id, cfg)}  obs_dim={obs_dim}")
    print(f"[phase2] prior={info['prior']}  prior params={info['prior_params']}"
          f"  quantum params={info['quantum_params']}"
          f"  total={info['total_params']}")
    print(f"[phase2] budget: batch {cfg.batch_size}x{cfg.seq_len}, "
          f"horizon {cfg.horizon}, grad step every {cfg.train_every} env steps")

    buffer = ReplayBuffer(dcfg["buffer_capacity"], obs_dim, action_dim)
    obs, _ = env.reset(seed=cfg.seed)
    prev_action = np.zeros(action_dim, np.float32)
    state, is_first = None, True
    ep_return, ep_returns, metrics, grad_log = 0.0, [], {}, []
    t0 = time.time()
    solved_at = None

    for step in range(1, steps + 1):
        # ---- act
        if step <= cfg.warmup_steps:
            if discrete:
                action = np.eye(action_dim, dtype=np.float32)[env.action_space.sample()]
            else:
                action = np.random.uniform(-1, 1, action_dim).astype(np.float32)
        else:
            action, state = agent.policy(obs, state)
        next_obs, reward, term, trunc, _ = env.step(
            to_env_action(action, env, discrete))
        ep_return += reward
        buffer.add(obs, prev_action, reward, is_first, term)
        obs, prev_action, is_first = next_obs, action, False

        if term or trunc:
            ep_returns.append(ep_return)
            obs, _ = env.reset()
            prev_action = np.zeros(action_dim, np.float32)
            state, is_first, ep_return = None, True, 0.0

        # ---- learn
        if (step > cfg.warmup_steps and step % cfg.train_every == 0
                and len(buffer) > cfg.batch_size * cfg.seq_len):
            batch = buffer.sample(cfg.batch_size, cfg.seq_len, cfg.device)
            metrics = agent.train_step(batch)
            if cfg.log_grad_var and step % (cfg.train_every * cfg.grad_var_every) == 0:
                gs = prior_grad_stats(agent)
                if gs:
                    grad_log.append({"step": step, **gs})
                    metrics.update(gs)

        # ---- evaluate
        if step % cfg.eval_every == 0 or step == steps:
            ev, es = evaluate(agent, env_id, cfg, discrete,
                              cfg.eval_episodes, seed=10_000 + step)
            recent = float(np.mean(ep_returns[-10:])) if ep_returns else float("nan")
            writer.writerow([step, ev, es, recent,
                             metrics.get("wm/recon", ""), metrics.get("wm/kl", ""),
                             metrics.get("prior_q_var", ""),
                             metrics.get("prior_ratio", "")])
            csv_file.flush()
            gate = TASK_GATE[cfg.task]
            if ev >= gate and solved_at is None:
                solved_at = step
            print(f"step {step:>7d} | eval {ev:8.1f} +/- {es:5.1f} | "
                  f"train {recent:8.1f} | "
                  f"{(step)/(time.time()-t0):5.1f} steps/s"
                  + (f" | SOLVED at {solved_at}" if solved_at else ""))
            durable(lambda p: save_checkpoint(p, agent, step, info),
                    os.path.join(run, "checkpoint.pt"))
            durable_json(grad_log, os.path.join(run, "grad_log.json"))

    ev, es = evaluate(agent, env_id, cfg, discrete, 20, seed=999)
    result = {**info, "task": cfg.task, "quantum": cfg.quantum,
              "partial_obs": cfg.partial_obs, "seed": cfg.seed,
              "env_steps": steps, "final_eval_return": ev,
              "final_eval_std": es, "solved_at_step": solved_at,
              "gate": TASK_GATE[cfg.task],
              "gate_met": bool(ev >= TASK_GATE[cfg.task]),
              "wall_clock_s": time.time() - t0, "run_dir": run}
    # result.json is the completion marker: reevaluate.py treats its absence
    # as "still training" and the overwrite guard treats its presence as
    # "finished". It is therefore the one write that must not be lost, so it
    # is attempted hardest and its failure is reported loudly.
    if not durable_json(result, os.path.join(run, "result.json")):
        print("[phase2] WARNING: result.json was not written. This run will "
              "be treated as incomplete and re-run. Recover it from "
              "/tmp/qtwm_rescue if you do not want to repeat it.")
    csv_file.close()
    env.close()

    print(f"\n[phase2] final eval over 20 episodes: {ev:.1f} +/- {es:.1f}")
    print(f"[phase2] gate {TASK_GATE[cfg.task]}: "
          f"{'MET' if result['gate_met'] else 'not met'}"
          + (f" (first crossed at {solved_at} steps)" if solved_at else ""))
    print(f"[phase2] -> {run}")
    print("[phase2] NOTE: one seed is not evidence, and the classical control "
          "must be run with identical settings before any comparison.")


if __name__ == "__main__":
    main()
