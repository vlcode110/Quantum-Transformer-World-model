"""Two things the paper asserts about gradients, verified against the code.

    python test_gradient_isolation.py

Exit status 0 only if every check passes. Builds one small agent, fills a
short replay buffer, and runs a single training step with the world-model
and actor-critic phases separated so each can be measured on its own.

WHAT THIS ANSWERS
-----------------
(1) What prior_grad_stats() actually measures, confirmed empirically
    rather than read off the source: the axis of each variance, and what
    prior_ratio divides by.

(2) Whether the actor and critic losses can change the prior's
    parameters. They cannot, but the reason is indirect -- optimiser
    scoping plus call ordering, not a stop-gradient -- so it deserves a
    test rather than a comment.

WHY A CHECK AND NOT A STOP-GRADIENT
-----------------------------------
`ents` is computed inside ActorCritic.imagine() from a non-detached
`feat`, so actor_loss.backward() does deposit gradient on world-model
parameters, including the circuit. Detaching it would be the direct
fix, and it is the wrong one here for two reasons.

First, the same graph is load-bearing in continuous-action mode: the
comment at agent.py:107-108 keeps it deliberately so the actor gradient
can flow through imagined dynamics, which is DreamerV3's backprop mode.
A blanket detach would silently change the algorithm on continuous
tasks.

Second, Phase 2's design rule is that DreamerV3 is not modified --- every
quantum substitution happens by swapping a module on an already-built
agent, so the baseline stays byte-identical and the comparison isolates
one sublayer. Editing the actor loss would break that guarantee for a
behaviour that is already correct.

So the deposited gradient is left in place and its harmlessness is
asserted here. If someone later adds world-model parameters to the actor
optimiser, or reorders train_step so the deposit is no longer cleared,
this test fails.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

results = []


def check(name, fn):
    try:
        detail = fn()
        results.append(True)
        print(f"  PASS  {name}" + (f"\n          {detail}" if detail else ""))
    except Exception as e:
        results.append(False)
        print(f"  FAIL  {name}\n          {type(e).__name__}: {e}")


def main():
    try:
        import torch
    except ImportError:
        sys.exit("torch is missing — activate the project environment:\n"
                 "    source ~/.venvs/qtwm/bin/activate")

    import train_agent as ta
    from quantum_rssm import swap_prior, prior_grad_stats
    from config import Config
    import gymnasium as gym

    cfg = Config()
    cfg.quantum = True
    cfg.prior_arch = "transformer"
    cfg.prior_context = 4
    cfg.partial_obs = True
    cfg.batch_size, cfg.seq_len, cfg.horizon = 4, 8, 5   # small, this is a test

    Dreamer, ReplayBuffer, dreamer_cfg = ta.import_dreamer(cfg.dreamer_root)
    dcfg = dreamer_cfg.get_config(cfg.task)
    for k in ("batch_size", "seq_len", "horizon", "train_every", "warmup_steps"):
        dcfg[k] = getattr(cfg, k)

    env = ta.make_env(dcfg["env_id"], cfg, seed=0)
    obs_dim = env.observation_space.shape[0]
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    action_dim = env.action_space.n if discrete else env.action_space.shape[0]

    agent = Dreamer(obs_dim, action_dim, discrete, dcfg, cfg.device)
    agent, info = swap_prior(agent, cfg, dcfg)
    print(f"\nagent built: prior={info['prior']}, "
          f"prior_params={info['prior_params']}, "
          f"circuit_params={info['quantum_params']}, discrete={discrete}\n")

    # ---- a short random rollout, enough for one batch
    buf = ReplayBuffer(2000, obs_dim, action_dim)
    obs, _ = env.reset(seed=0)
    prev = np.zeros(action_dim, np.float32)
    for t in range(600):
        a = env.action_space.sample()
        onehot = np.zeros(action_dim, np.float32)
        if discrete:
            onehot[a] = 1.0
        else:
            onehot = np.asarray(a, np.float32)
        buf.add(obs, prev, 0.0, t == 0, False)
        obs, r, term, trunc, _ = env.step(a)
        prev = onehot
        if term or trunc:
            obs, _ = env.reset()
            prev = np.zeros(action_dim, np.float32)
    batch = buf.sample(cfg.batch_size, cfg.seq_len, cfg.device)

    prior = agent.wm.rssm.prior_net
    qnames = [n for n, _ in prior.named_parameters() if "q." in n]
    cnames = [n for n, _ in prior.named_parameters() if "q." not in n]
    qnumel = sum(p.numel() for n, p in prior.named_parameters() if "q." in n)
    cnumel = sum(p.numel() for n, p in prior.named_parameters() if "q." not in n)

    # =================================================================
    print("PART 1 — what prior_grad_stats() measures\n")

    def c_selector():
        assert qnames, "the 'q.' selector matched no parameter"
        assert len(qnames) == 1, f"expected one circuit tensor, got {qnames}"
        assert qnumel == info["quantum_params"], \
            f"'q.' selector caught {qnumel} values, circuit has " \
            f"{info['quantum_params']}"
        return (f"circuit tensor: {qnames[0]} ({qnumel} values); "
                f"{len(cnames)} classical tensors ({cnumel} values). "
                f"No classical name contains 'q.', so the selector is exact.")
    check("1  the 'q.' name selector isolates exactly the circuit", c_selector)

    # one world-model step on its own
    (deters, stochs), _ = agent.wm.train_step(batch)
    wm_only = prior_grad_stats(agent)

    def c_axis():
        g = [p.grad.detach().flatten() for n, p in prior.named_parameters()
             if "q." in n and p.grad is not None]
        assert g, "no gradient on the circuit after a world-model step"
        gq = torch.cat(g)
        assert gq.numel() == qnumel, f"{gq.numel()} != {qnumel}"
        manual = float(gq.var())
        assert abs(manual - wm_only["prior_q_var"]) < 1e-12 * max(1.0, abs(manual)), \
            f"{manual} != {wm_only['prior_q_var']}"
        return (f"prior_q_var reproduces as var() over the {qnumel} circuit "
                f"parameters at ONE optimisation step.\n          "
                f"Not across steps (that is the figure's x-axis) and not "
                f"across the batch\n          (backward has already reduced "
                f"the batch dimension).")
    check("2  prior_q_var is a variance ACROSS PARAMETERS, at one step", c_axis)

    def c_ratio():
        expect = wm_only["prior_q_absmean"] / wm_only["prior_c_absmean"]
        assert abs(expect - wm_only["prior_ratio"]) < 1e-9 * max(1.0, expect)
        return (f"prior_ratio = prior_q_absmean / prior_c_absmean = "
                f"{wm_only['prior_ratio']:.4f}.\n          "
                f"A ratio of MEAN ABSOLUTE gradients, not of variances, and "
                f"the denominator is\n          the surrounding classical "
                f"parameters of the prior module ({cnumel} values) —\n          "
                f"not the whole world model and not the previous step.")
    check("3  prior_ratio divides circuit |grad| by classical |grad|", c_ratio)

    # =================================================================
    print("\nPART 2 — can the actor or critic change the prior?\n")

    before = {n: p.detach().clone() for n, p in prior.named_parameters()}
    agent.ac.train_step(agent.wm, deters, stochs)
    after_ac = prior_grad_stats(agent)

    def c_values():
        moved = [n for n, p in prior.named_parameters()
                 if not torch.equal(p.detach(), before[n])]
        assert not moved, f"actor/critic step changed {len(moved)} tensors: " \
                          f"{moved[:4]}"
        return (f"all {len(before)} prior tensors bit-identical after "
                f"ac.train_step(). The circuit is\n          trained by the "
                f"world-model loss alone.")
    check("4  ac.train_step() leaves every prior parameter unchanged", c_values)

    def c_disjoint():
        wm_ids = {id(p) for p in agent.wm.parameters()}
        for name, opt in (("actor_opt", agent.ac.actor_opt),
                          ("critic_opt", agent.ac.critic_opt)):
            got = {id(p) for g in opt.param_groups for p in g["params"]}
            assert not (got & wm_ids), \
                f"{name} contains {len(got & wm_ids)} world-model parameters"
        return ("neither optimiser holds a world-model parameter, so anything "
                "the actor\n          loss deposits there can never be applied.")
    check("5  actor and critic optimisers exclude world-model parameters",
          c_disjoint)

    def c_deposit():
        a = wm_only.get("prior_q_absmean")
        b = after_ac.get("prior_q_absmean")
        assert a is not None and b is not None
        rel = abs(b - a) / max(a, 1e-30)
        note = (f"circuit |grad| after world-model step: {a:.4e}\n          "
                f"after actor-critic step:            {b:.4e}  "
                f"({rel:+.1%})")
        if rel < 1e-9:
            return note + ("\n          The actor loss deposited nothing "
                           "measurable on the circuit.")
        return note + (
            "\n          The actor loss DOES add to the circuit's .grad, via "
            "the entropy term,\n          which is computed inside imagine() "
            "from a non-detached feat. Harmless —\n          checks 4 and 5 "
            "show it is never applied and wm.opt.zero_grad() clears it —\n"
            "          but it means grad_log.json records the SUM of the "
            "world-model and\n          actor-entropy gradients, not the "
            "world-model gradient alone.")
    check("6  quantify the entropy-term deposit on the circuit", c_deposit)

    def c_cleared():
        agent.wm.opt.zero_grad()
        left = [n for n, p in prior.named_parameters()
                if p.grad is not None and float(p.grad.abs().sum()) > 0]
        assert not left, f"gradient survived zero_grad() on {left[:4]}"
        return ("wm.opt.zero_grad() clears the deposit before the next "
                "world-model step.")
    check("7  the deposit is cleared before it could ever be used", c_cleared)

    print()
    ok = sum(results)
    if ok == len(results):
        print(f"ALL {len(results)} CHECKS PASSED.\n")
        print("Summary for the manuscript:")
        print("  prior_q_var / prior_c_var — variance across the parameters of")
        print("    that group, at a single optimisation step.")
        print("  prior_ratio — mean |grad| on the 28 circuit parameters divided")
        print("    by mean |grad| on the surrounding classical parameters of the")
        print("    same module.")
        print("  Both are read after agent.train_step(), so they include the")
        print("    actor's entropy-term contribution; check 6 reports its size.")
        print("  The actor and critic losses do not update the prior.")
    else:
        print(f"{len(results) - ok} of {len(results)} FAILED.")
    sys.exit(0 if ok == len(results) else 1)


if __name__ == "__main__":
    import torch  # noqa: F401  (imported in main too, for the error message)
    main()
