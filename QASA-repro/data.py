"""Synthetic univariate signals in the spirit of QASA's benchmark suite:
clean periodic (classical should win), noisy periodic, and chaotic
(quantum layer should help, per the paper's task-conditional claim)."""
import numpy as np


def sine(n=3000, period=50):
    t = np.arange(n)
    return np.sin(2 * np.pi * t / period)


def noisy_sine(n=3000, period=50, sigma=0.3, seed=0):
    rng = np.random.default_rng(seed)
    return sine(n, period) + sigma * rng.standard_normal(n)


def mackey_glass(n=3000, tau=17, beta=0.2, gamma=0.1, exp=10, seed=0):
    rng = np.random.default_rng(seed)
    warmup = 500
    x = np.zeros(n + warmup + tau)
    x[:tau] = 1.2 + 0.1 * rng.standard_normal(tau)
    for t in range(tau, len(x) - 1):
        x[t + 1] = x[t] + beta * x[t - tau] / (1 + x[t - tau] ** exp) - gamma * x[t]
    return x[warmup + tau:]


SIGNALS = {"sine": sine, "noisy_sine": noisy_sine, "mackey_glass": mackey_glass}


def windows(series, lookback=24):
    """(N, lookback) inputs, (N,) next-step targets, normalized by train stats."""
    n_train = int(0.7 * len(series))
    mu, sd = series[:n_train].mean(), series[:n_train].std() + 1e-8
    s = (series - mu) / sd
    X = np.stack([s[i:i + lookback] for i in range(len(s) - lookback)])
    y = s[lookback:]
    n_tr = int(0.7 * len(X))
    n_va = int(0.85 * len(X))
    return ((X[:n_tr], y[:n_tr]), (X[n_tr:n_va], y[n_tr:n_va]),
            (X[n_va:], y[n_va:]))
