"""Environments, including the partial-observability variant.

Hiding velocity is the cheapest available test of the hypothesis Phase 1
generated: that quantum attention contributes when the dynamics are
history-dependent, and contributes nothing when the current state already
determines the next one.

With velocities hidden the agent cannot act well from a single observation —
it must integrate several to infer how fast things are moving. That makes the
latent dynamics genuinely non-Markovian while leaving every other part of the
pipeline unchanged.
"""
import gymnasium as gym
import numpy as np

# Indices retained under partial observability, per environment.
#   CartPole-v1: [cart position, cart velocity, pole angle, pole angular vel]
#   Pendulum-v1: [cos(theta), sin(theta), theta_dot]
DEFAULT_KEEP = {
    "CartPole-v1": (0, 2),      # positions only
    "Pendulum-v1": (0, 1),      # angle only, angular velocity hidden
}


class HideDimensions(gym.ObservationWrapper):
    """Expose only selected observation dimensions.

    Implemented as an ObservationWrapper rather than by zeroing the hidden
    entries, so the agent's input width genuinely shrinks and there is no way
    for it to exploit a constant channel.
    """

    def __init__(self, env, keep):
        super().__init__(env)
        self.keep = np.asarray(keep, dtype=np.int64)
        low = env.observation_space.low[self.keep]
        high = env.observation_space.high[self.keep]
        self.observation_space = gym.spaces.Box(low=low, high=high,
                                                dtype=env.observation_space.dtype)

    def observation(self, obs):
        return np.asarray(obs, dtype=np.float32)[self.keep]


def make_env(env_id, cfg, seed=None):
    """Build the environment, applying partial observability if configured."""
    env = gym.make(env_id)
    if cfg.partial_obs:
        keep = tuple(cfg.partial_obs_keep) or DEFAULT_KEEP.get(env_id)
        if keep is None:
            raise ValueError(
                f"No default partial-observation indices for {env_id}; "
                f"set --partial_obs_keep explicitly.")
        env = HideDimensions(env, keep)
    if seed is not None:
        env.reset(seed=seed)
    return env


def describe(env_id, cfg):
    """One-line description for logs, so a run's observation setup is never
    ambiguous after the fact."""
    if not cfg.partial_obs:
        return f"{env_id} (full observation)"
    keep = tuple(cfg.partial_obs_keep) or DEFAULT_KEEP.get(env_id, ())
    return f"{env_id} (partial: keeping dims {list(keep)}, velocities hidden)"


def to_env_action(action, env, discrete):
    """Map the agent's action representation onto the environment's."""
    if discrete:
        return int(np.argmax(action))
    low, high = env.action_space.low, env.action_space.high
    return low + (action + 1.0) * 0.5 * (high - low)
