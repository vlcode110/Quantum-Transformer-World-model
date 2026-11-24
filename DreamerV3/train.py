"""Train small DreamerV3 on CartPole-v1 or Pendulum-v1.

Usage:
    python train.py --task cartpole
    python train.py --task pendulum
    python train.py --task cartpole --steps 300 --warmup 100  # smoke test
"""
import argparse
import csv
import os
import time

import gymnasium as gym
import numpy as np
import torch

from config import get_config
from dreamer.agent import Dreamer
from dreamer.buffer import ReplayBuffer


def save_checkpoint(path, agent, buffer, step, cfg, save_buffer=True):
    """Full training state, not just weights: optimizer moments and the replay
    buffer matter as much as the parameters for a seamless resume.

    Written to a temp file then renamed — an atomic swap, so a crash (or an
    iCloud stall) mid-write can never leave a truncated checkpoint."""
    ckpt = {
        "model": agent.state_dict(),
        "wm_opt": agent.wm.opt.state_dict(),
        "actor_opt": agent.ac.actor_opt.state_dict(),
        "critic_opt": agent.ac.critic_opt.state_dict(),
        "step": step,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
    }
    if save_buffer:
        n = len(buffer)
        ckpt["buffer"] = {
            "obs": buffer.obs[:n], "action": buffer.action[:n],
            "reward": buffer.reward[:n], "is_first": buffer.is_first[:n],
            "is_terminal": buffer.is_terminal[:n],
            "idx": buffer.idx, "full": buffer.full,
        }
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, agent, buffer, device):
    """Returns the step to resume from. Tolerates weights-only checkpoints
    written by earlier versions of this script."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model" not in ckpt:            # legacy: bare state_dict
        agent.load_state_dict(ckpt)
        print(f"[resume] legacy weights-only checkpoint — optimizer state and "
              f"replay buffer are lost; pass --start_step to set the counter "
              f"and the buffer will refill over the warmup window.")
        return 0
    agent.load_state_dict(ckpt["model"])
    agent.wm.opt.load_state_dict(ckpt["wm_opt"])
    agent.ac.actor_opt.load_state_dict(ckpt["actor_opt"])
    agent.ac.critic_opt.load_state_dict(ckpt["critic_opt"])
    torch.set_rng_state(ckpt["torch_rng"].cpu().to(torch.uint8))
    np.random.set_state(ckpt["numpy_rng"])
    if "buffer" in ckpt:
        b = ckpt["buffer"]
        n = len(b["obs"])
        buffer.obs[:n] = b["obs"]; buffer.action[:n] = b["action"]
        buffer.reward[:n] = b["reward"]; buffer.is_first[:n] = b["is_first"]
        buffer.is_terminal[:n] = b["is_terminal"]
        buffer.idx, buffer.full = b["idx"], b["full"]
        print(f"[resume] restored {n} buffer transitions")
    print(f"[resume] continuing from step {ckpt['step']}")
    return ckpt["step"]


def make_env(env_id, seed=None):
    env = gym.make(env_id)
    if seed is not None:
        env.reset(seed=seed)
    return env


def to_env_action(action, env, discrete):
    if discrete:
        return int(np.argmax(action))
    low, high = env.action_space.low, env.action_space.high
    return low + (action + 1.0) * 0.5 * (high - low)


@torch.no_grad()
def evaluate(agent, env_id, discrete, episodes, seed=0):
    env = make_env(env_id)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["cartpole", "pendulum"], default="cartpole")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--logdir", default=None)
    ap.add_argument("--resume", nargs="?", const="auto", default=None,
                    help="resume training; bare --resume uses "
                         "<logdir>/checkpoint.pt")
    ap.add_argument("--start_step", type=int, default=None,
                    help="step counter to resume from (needed only for "
                         "legacy weights-only checkpoints)")
    ap.add_argument("--no_save_buffer", action="store_true",
                    help="smaller checkpoints; resume then refills the buffer")
    args = ap.parse_args()

    cfg = get_config(args.task)
    if args.steps:
        cfg["steps"] = args.steps
    if args.warmup:
        cfg["warmup_steps"] = args.warmup

    if args.device == "auto":
        device = ("cuda" if torch.cuda.is_available() else
                  "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logdir = args.logdir or os.path.join("runs", f"{args.task}_seed{args.seed}")
    os.makedirs(logdir, exist_ok=True)

    env = make_env(cfg["env_id"], seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    action_dim = env.action_space.n if discrete else env.action_space.shape[0]

    agent = Dreamer(obs_dim, action_dim, discrete, cfg, device)
    buffer = ReplayBuffer(cfg["buffer_capacity"], obs_dim, action_dim)

    start_step = 0
    if args.resume:
        ckpt_path = (os.path.join(logdir, "checkpoint.pt")
                     if args.resume == "auto" else args.resume)
        if not os.path.exists(ckpt_path):
            raise SystemExit(f"[resume] no checkpoint at {ckpt_path}")
        start_step = load_checkpoint(ckpt_path, agent, buffer, device)
        if args.start_step is not None:
            start_step = args.start_step
        if len(buffer) == 0:
            # No buffer to resume from: collect fresh random-ish data before
            # gradient steps restart, otherwise the first updates train on
            # almost nothing and can undo a good policy.
            cfg["warmup_steps"] = start_step + cfg["warmup_steps"]
            print(f"[resume] empty buffer — refilling until step "
                  f"{cfg['warmup_steps']}")

    csv_path = os.path.join(logdir, "log.csv")
    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)
    if os.stat(csv_path).st_size == 0:
        writer.writerow(["step", "eval_return", "eval_std", "episode_return",
                         "wm_recon", "wm_kl", "ac_return"])

    obs, _ = env.reset(seed=args.seed)
    prev_action = np.zeros(action_dim, np.float32)
    state, is_first = None, True
    ep_return, ep_returns = 0.0, []
    metrics = {}
    t0 = time.time()
    print(f"task={args.task} device={device} steps={cfg['steps']} logdir={logdir}")

    for step in range(start_step + 1, cfg["steps"] + 1):
        # --- act ---
        if step <= cfg["warmup_steps"]:
            if discrete:
                a_idx = env.action_space.sample()
                action = np.eye(action_dim, dtype=np.float32)[a_idx]
            else:
                action = np.random.uniform(-1, 1, action_dim).astype(np.float32)
        else:
            action, state = agent.policy(obs, state)
        env_action = to_env_action(action, env, discrete)
        next_obs, reward, term, trunc, _ = env.step(env_action)
        ep_return += reward

        buffer.add(obs, prev_action, reward, is_first, term)
        obs, prev_action, is_first = next_obs, action, False

        if term or trunc:
            ep_returns.append(ep_return)
            obs, _ = env.reset()
            prev_action = np.zeros(action_dim, np.float32)
            state, is_first, ep_return = None, True, 0.0

        # --- train ---
        if step > cfg["warmup_steps"] and step % cfg["train_every"] == 0 \
                and len(buffer) > cfg["batch_size"] * cfg["seq_len"]:
            batch = buffer.sample(cfg["batch_size"], cfg["seq_len"], device)
            metrics = agent.train_step(batch)

        # --- eval / log ---
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            eval_mean, eval_std = evaluate(agent, cfg["env_id"], discrete,
                                           cfg["eval_episodes"], seed=1000 + step)
            recent = np.mean(ep_returns[-10:]) if ep_returns else float("nan")
            writer.writerow([step, eval_mean, eval_std, recent,
                             metrics.get("wm/recon", ""), metrics.get("wm/kl", ""),
                             metrics.get("ac/return", "")])
            csv_file.flush()
            fps = max(step - start_step, 1) / (time.time() - t0)
            print(f"step {step:>7d} | eval {eval_mean:8.1f} ± {eval_std:5.1f} | "
                  f"train_ep {recent:8.1f} | fps {fps:5.0f} | "
                  f"recon {metrics.get('wm/recon', float('nan')):.3f} "
                  f"kl {metrics.get('wm/kl', float('nan')):.2f}")
            save_checkpoint(os.path.join(logdir, "checkpoint.pt"), agent,
                            buffer, step, cfg,
                            save_buffer=not args.no_save_buffer)
            if eval_mean >= cfg["solved"]:
                print(f"SOLVED: eval return {eval_mean:.1f} >= {cfg['solved']}")

    csv_file.close()
    env.close()


if __name__ == "__main__":
    main()
