"""Configs for small DreamerV3 on classic control."""

BASE = {
    # model sizes (small config for vector observations)
    "hidden": 256,
    "layers": 2,
    "embed": 128,
    "deter": 256,
    "stoch": 16,
    "classes": 16,
    "bins": 255,
    # optimization
    "model_lr": 3e-4,
    "actor_lr": 1e-4,
    "critic_lr": 1e-4,
    "grad_clip": 100.0,
    # actor-critic
    "horizon": 15,
    "gamma": 0.997,
    "lam": 0.95,
    "ent_scale": 3e-4,
    "slow_tau": 0.02,
    # data
    "batch_size": 16,
    "seq_len": 32,
    "buffer_capacity": 200_000,
    "warmup_steps": 1_000,
    "train_every": 2,       # one gradient step every N env steps
    "eval_every": 5_000,
    "eval_episodes": 5,
}

TASKS = {
    "cartpole": {
        "env_id": "CartPole-v1",
        "steps": 60_000,
        "solved": 475.0,       # published-level: max return is 500
    },
    "pendulum": {
        "env_id": "Pendulum-v1",
        "steps": 150_000,
        "solved": -200.0,      # good agents reach roughly -150 to -200
        "ent_scale": 1e-4,
    },
}


def get_config(task):
    cfg = dict(BASE)
    cfg.update(TASKS[task])
    return cfg
