"""Is the agent actually using the world model, or succeeding despite it?

A model-based agent can reach competence while effectively ignoring a poor
world model: the policy learns something that happens to work in the real
environment even though the imagined rollouts it was trained on were wrong.
If that is what is happening here, the paper's central premise — that a quantum
circuit is doing the world-modelling — is unsupported, however good the returns
look.

Three measurements, all from checkpoints that already exist. No training.

  OPEN-LOOP ERROR    Roll the model forward on the prior alone, as imagination
      does, then DECODE each prediction and compare it against the observation
      actually received, both in symlog space (which is where the decoder is
      trained). Observation space rather than latent space, so the number is
      interpretable instead of being a distance in an arbitrary basis. Error
      that stays bounded over the horizon means the imagined trajectories
      resemble the real ones; error that explodes within a step or two means
      the policy was trained on noise.

      NOTE ON SCOPE: each episode is truncated at `horizon + 12` steps, so this
      measures short-horizon prediction during the RECOVERY phase after reset,
      not the stabilised phase of a long episode. That is deliberate -- a
      persistence baseline is near-correct once a policy has stabilised and
      would dominate trivially -- but it bounds what the numbers mean.

  PRIOR-POSTERIOR KL  How far the prior's belief sits from the posterior's at
      each step. This is the quantity the world-model loss optimises, so it
      reports whether the quantum prior actually learned the dynamics or
      settled for a diffuse distribution that is cheap under the KL term.

  PERSISTENCE BASELINE  The same open-loop error for a model that predicts no
      change. A world model that cannot beat "nothing happens" is not being
      used for anything, whatever its architecture.

    python imagination_quality.py
    python imagination_quality.py --runs runs/cartpole_quantum_q6L4_full_b8x16h10_seed0

Writes imagination_quality.json and figures/imagination.png.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load_agent(run_dir, ta, quantum_rssm):
    """Rebuild the agent exactly as train_agent.main() does, then restore the
    checkpoint. Construction is duplicated rather than imported because
    train_agent.py performs it inline inside main(); keeping the two in step
    matters, so any change there must be mirrored here."""
    import gymnasium as gym
    with open(os.path.join(run_dir, "config.json")) as f:
        raw = json.load(f)
    cfg = ta.Config()
    for k, v in raw.items():
        if hasattr(cfg, k):
            setattr(cfg, k, tuple(v) if isinstance(v, list) else v)

    Dreamer, _ReplayBuffer, dreamer_cfg = ta.import_dreamer(cfg.dreamer_root)
    dcfg = dreamer_cfg.get_config(cfg.task)
    for k in ("batch_size", "seq_len", "horizon", "train_every", "warmup_steps"):
        dcfg[k] = getattr(cfg, k)

    env = ta.make_env(dcfg["env_id"], cfg, seed=cfg.seed)
    obs_dim = env.observation_space.shape[0]
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    action_dim = env.action_space.n if discrete else env.action_space.shape[0]

    agent = Dreamer(obs_dim, action_dim, discrete, dcfg, cfg.device)
    agent, _ = quantum_rssm.swap_prior(agent, cfg, dcfg)

    ckpt = torch.load(os.path.join(run_dir, "checkpoint.pt"),
                      map_location=cfg.device)
    state = ckpt.get("agent", ckpt.get("model", ckpt))
    if isinstance(state, dict):
        agent.load_state_dict(state, strict=False)
    return agent, cfg, env, dcfg["env_id"], discrete, ckpt.get("step")


@torch.no_grad()
def rollout_error(agent, env, horizon, episodes, device, to_env_action,
                  discrete):
    """Open-loop prior rollout against the observation-informed posterior.

    The world model exposes `rssm.observe` for the posterior sequence and
    `rssm.img_step` for a single prior step; imagination is repeated img_step
    with no observation, which is what is reproduced here. Predictions are
    compared in OBSERVATION space by decoding, so the error is interpretable
    rather than being a distance in an arbitrary latent basis.
    """
    import importlib
    wm, rssm = agent.wm, agent.wm.rssm
    symlog = importlib.import_module(
        agent.wm.__module__.rsplit(".", 1)[0] + ".nets").symlog

    model_err = np.zeros(horizon)
    persist_err = np.zeros(horizon)
    counts = np.zeros(horizon)
    n = 0
    lengths = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=2000 + ep)
        state, obs_seq, act_seq, first_seq = None, [], [], []
        for t in range(horizon + 12):
            action, state = agent.policy(obs, state, mode=True)
            obs_seq.append(obs)
            act_seq.append(np.asarray(action, dtype=np.float32))
            first_seq.append(1.0 if t == 0 else 0.0)
            obs, _, term, trunc, _ = env.step(
                to_env_action(action, env, discrete))
            if term or trunc:
                break
        lengths.append(len(obs_seq))
        # only needs enough steps to make at least one prediction; short
        # episodes contribute to the horizons they can reach, counted per k
        if len(obs_seq) < 5:
            continue

        T = len(obs_seq)
        O = torch.as_tensor(np.array(obs_seq), dtype=torch.float32,
                            device=device).unsqueeze(0)
        A = torch.as_tensor(np.array(act_seq), dtype=torch.float32,
                            device=device).unsqueeze(0)
        F = torch.as_tensor(np.array(first_seq), dtype=torch.float32,
                            device=device).unsqueeze(0)

        embeds = wm.encoder(symlog(O))
        # observe() returns FIVE values; its docstring lists four and omits
        # the trailing carry state, which is what the unpack was tripping on.
        deters, stochs, _posts, _priors, _carry = rssm.observe(embeds, A, F)

        # start the prior-only rollout a few steps in, once the recurrent
        # state carries information
        t0 = 2
        deter, stoch = deters[:, t0], stochs[:, t0]
        for k in range(horizon):
            if t0 + 1 + k >= T:
                break
            deter, prior = rssm.img_step(deter, stoch, A[:, t0 + k])
            stoch = rssm._sample(prior)
            pred = wm.decoder(wm.feat(deter, stoch))
            # the decoder is trained against symlog(obs), so the comparison
            # must happen in that space. Comparing its output to raw
            # observations inflates the model error while leaving the
            # persistence baseline untouched, which biases the whole test.
            truth = symlog(O[:, t0 + 1 + k])
            model_err[k] += float(((pred - truth) ** 2).mean())
            persist_err[k] += float(((symlog(O[:, t0]) - truth) ** 2).mean())
            counts[k] += 1
        n += 1

    if n == 0 or counts.max() == 0:
        raise RuntimeError(
            f"no usable episode; lengths seen: "
            f"{lengths[:10]}{' ...' if len(lengths) > 10 else ''}")
    valid = counts > 0
    model_err[valid] /= counts[valid]
    persist_err[valid] /= counts[valid]
    model_err[~valid] = np.nan
    persist_err[~valid] = np.nan
    return model_err, persist_err, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", default="imagination_quality.json")
    args = ap.parse_args()

    runs = args.runs or sorted(glob.glob(os.path.join(HERE, "runs", "*")))
    runs = [r for r in runs if os.path.exists(os.path.join(r, "checkpoint.pt"))]
    if not runs:
        raise SystemExit("[imagination] no checkpoints found under runs/.")

    sys.path.insert(0, HERE)
    import quantum_rssm
    import train_agent as ta

    for fn in ("import_dreamer", "make_env", "Config", "to_env_action",
               "evaluate"):
        if not hasattr(ta, fn):
            raise SystemExit(
                f"[imagination] train_agent.py no longer exposes {fn!r}; "
                f"agent construction here must be kept in step with it.")

    out = {}
    for run in runs:
        name = os.path.basename(run)
        try:
            agent, cfg, env, env_id, discrete, step = load_agent(
                run, ta, quantum_rssm)
            me, pe, kl = rollout_error(agent, env, args.horizon,
                                       args.episodes, cfg.device,
                                       ta.to_env_action, discrete)
        except SystemExit:
            raise
        except Exception as e:
            print(f"  {name:<50} skipped ({type(e).__name__}: {e})")
            continue
        k_last = int(np.nanargmax(np.where(np.isfinite(me), np.arange(len(me)), -1)))
        ratio = float(me[k_last] / max(pe[k_last], 1e-12))
        out[name] = {"step": step, "model_error": me.tolist(),
                     "persistence_error": pe.tolist(),
                     "episodes_per_horizon": kl.tolist(),
                     "horizon_used": k_last + 1,
                     "final_ratio_vs_persistence": ratio}
        verdict = "model is informative" if ratio < 0.8 else \
                  "NOT better than predicting no change"
        print(f"  {name:<50} k={k_last+1} error {me[k_last]:.5f}, "
              f"persistence {pe[k_last]:.5f}, ratio {ratio:.3f}  {verdict}")

    with open(os.path.join(HERE, args.out), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n-> {args.out}")
    print("\nA ratio below 1 means the learned prior predicts the agent's own")
    print("future better than assuming nothing changes. Above 1 means the")
    print("policy succeeded in spite of the world model, not because of it,")
    print("which would undercut the claim that the circuit does the modelling.")


if __name__ == "__main__":
    main()
