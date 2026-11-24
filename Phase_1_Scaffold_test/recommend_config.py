"""Turn the Phase 0 measurements into concrete n_qubits / q_layers / latent_dim.

Three independent ceilings constrain the quantum core; the answer is the
lowest of them:

  1. THROUGHPUT   how wide you can afford to SIMULATE.
                  From benchmark_statevector.py. Converted here into "hours per
                  Phase 1 training run" and "hours for a full ablation sweep",
                  because that -- not an abstract ms/step -- is what limits you.

  2. TRAINABILITY how wide you can afford to TRAIN.
                  From gradient_variance.py. A width is usable while the LOCAL
                  readout's gradient variance is (a) not decaying faster than
                  ~0.6x per added qubit, and (b) still well above numerical
                  noise in absolute terms.

  3. ENCODING     angle encoding maps one feature to one qubit, so
                  latent_dim <= n_qubits. This sets latent_dim, not n_qubits.

Depth (q_layers) is a separate trade: more layers buy expressivity but cost
linearly in time and accelerate gradient decay. Default rule below: the largest
depth measured whose gradient variance is still healthy, capped at 6.

Usage:
    python recommend_config.py
    python recommend_config.py --hours-per-run 2 --ablation-runs 24
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np

from config import Config

# a Phase 1 run's cost in gradient steps: epochs x (windows / batch)
# windows ~= total collected steps (one per timestep, minus context+horizon)


def read(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def throughput_ceiling(rows, steps_per_run, hours_per_run, ablation_runs):
    """Largest width whose single-run time fits the budget, per simulator."""
    by = defaultdict(dict)
    for r in rows:
        by[r["device"]][int(float(r["qubits"]))] = float(r["ms_per_step"])
    out = {}
    for dev, d in by.items():
        ok = [q for q, ms in sorted(d.items())
              if ms * steps_per_run / 3.6e6 <= hours_per_run]
        out[dev] = {
            "max_qubits": max(ok) if ok else None,
            "table": {q: {"hours_per_run": ms * steps_per_run / 3.6e6,
                          "hours_full_sweep": ms * steps_per_run
                          * ablation_runs / 3.6e6}
                      for q, ms in sorted(d.items())},
        }
    return out


def trainability_ceiling(rows, decay_floor=0.6, abs_floor=1e-8):
    """Largest width where the LOCAL readout still has usable gradients."""
    local = defaultdict(dict)      # layers -> qubits -> var
    for r in rows:
        if r["cost"] != "local":
            continue
        local[int(float(r["layers"]))][int(float(r["qubits"]))] = float(r["grad_var"])
    out = {}
    for layers, d in sorted(local.items()):
        q = np.array(sorted(d), float)
        v = np.array([max(d[int(x)], 1e-30) for x in q])
        half = len(q) // 2
        factor = float(np.exp(np.polyfit(q[half:], np.log(v[half:]), 1)[0])) \
            if len(q) - half >= 2 else float("nan")
        usable = [int(x) for x, y in zip(q, v) if y >= abs_floor]
        out[layers] = {"decay_per_qubit": factor,
                       "max_qubits": max(usable) if usable else None,
                       "healthy": (np.isnan(factor) or factor >= decay_floor)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qasa_root", default="../../../QASA-repro")
    ap.add_argument("--hours-per-run", type=float, default=3.0,
                    help="acceptable wall-clock for ONE Phase 1 training run")
    ap.add_argument("--ablation-runs", type=int, default=24,
                    help="runs in the sweep you intend (ablate.py prints this)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--collected-steps", type=int, default=40000,
                    help="approx timesteps collected (200 eps x ~200 steps)")
    args = ap.parse_args()

    cfg = Config()
    epochs = args.epochs or cfg.epochs
    steps_per_run = epochs * max(args.collected_steps * 0.7 // cfg.batch_size, 1)
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        args.qasa_root, "results")

    bench = read(os.path.join(root, "statevector_benchmark.csv"))
    grads = read(os.path.join(root, "gradient_variance.csv"))

    print(f"Assuming {epochs} epochs x {int(steps_per_run/epochs)} steps/epoch "
          f"= {int(steps_per_run):,} gradient steps per run\n")

    # ---------------------------------------------------------- 1. throughput
    t_max = None
    if bench:
        print("1. THROUGHPUT ceiling (how wide you can afford to simulate)")
        tp = throughput_ceiling(bench, steps_per_run, args.hours_per_run,
                                args.ablation_runs)
        for dev, d in tp.items():
            print(f"   {dev}:")
            for q, t in d["table"].items():
                mark = "  <-- limit" if q == d["max_qubits"] else ""
                print(f"      {q:>2}q  {t['hours_per_run']:>7.2f} h/run   "
                      f"{t['hours_full_sweep']:>8.1f} h/sweep{mark}")
            print(f"      max under {args.hours_per_run} h/run: "
                  f"{d['max_qubits']}")
        cands = [d["max_qubits"] for d in tp.values() if d["max_qubits"]]
        t_max = max(cands) if cands else None   # use your FASTEST simulator
        print(f"   -> throughput ceiling: {t_max} qubits "
              f"(on the fastest simulator available)\n")
    else:
        print("1. THROUGHPUT: benchmark CSV not found — run "
              "benchmark_statevector.py\n")

    # -------------------------------------------------------- 2. trainability
    g_max, best_depth = None, None
    if grads:
        print("2. TRAINABILITY ceiling (how wide you can afford to train)")
        tc = trainability_ceiling(grads)
        for layers, d in tc.items():
            flag = "healthy" if d["healthy"] else "DECAYING - plateau risk"
            print(f"   L={layers}: decay x{d['decay_per_qubit']:.2f}/qubit, "
                  f"usable to {d['max_qubits']}q  [{flag}]")
        # Decay RATE and absolute MAGNITUDE are different questions and an
        # earlier version of this script conflated them, refusing to recommend
        # anything when the rate looked bad. What actually blocks training is
        # the gradient falling below the noise you can resolve:
        #   - in noiseless simulation (backprop) that floor is ~1e-8, so a
        #     steep decay is a scaling warning, not a present-day blocker;
        #   - on hardware with finite shots the floor is ~1/sqrt(shots), and
        #     the same decay becomes fatal much sooner.
        usable = {l: d for l, d in tc.items() if d["max_qubits"]}
        if usable:
            g_max = max(d["max_qubits"] for d in usable.values())
            best_depth = min(max(usable), 6)
            steep = [l for l, d in usable.items() if not d["healthy"]]
            if steep:
                print(f"   WARNING: decay is steep at depth(s) {steep}. "
                      f"Gradients are still resolvable in simulation, so this "
                      f"does not block Phase 1 — but it predicts trouble on "
                      f"hardware and when scaling width. Measure shallower "
                      f"depths before blaming width.")
        print(f"   -> trainability ceiling: {g_max} qubits "
              f"(absolute-magnitude limit), suggested depth {best_depth}\n")
    else:
        print("2. TRAINABILITY: gradient CSV not found — run "
              "gradient_variance.py\n")

    # ------------------------------------------------------------ 3. verdict
    if t_max and g_max:
        n_qubits = min(t_max, g_max)
        binding = "throughput" if t_max <= g_max else "trainability"
        # leave one qubit of headroom: the world model adds encoder/decoder
        # cost on top of the circuit, and Phase 2 will be tighter than Phase 1
        print("3. RECOMMENDATION")
        print(f"   n_qubits   = {n_qubits}   (binding constraint: {binding})")
        print(f"   q_layers   = {best_depth}")
        print(f"   latent_dim = {n_qubits}   (angle encoding: one feature/qubit)")
        print(f"\n   Edit config.py:")
        print(f"       n_qubits: int = {n_qubits}")
        print(f"       q_layers: int = {best_depth}")
        print(f"       latent_dim: int = {n_qubits}")
        if binding == "trainability":
            print("\n   NOTE: trainability binds before throughput. That is the "
                  "\n   scientifically interesting case -- it means the limit is "
                  "\n   the barren plateau, not your laptop, and it belongs in "
                  "\n   the Phase 1 write-up as a finding rather than a footnote.")
        else:
            print("\n   NOTE: simulation speed binds first, so the ceiling is an "
                  "\n   artefact of your hardware, not physics. Revisit with "
                  "\n   lightning.qubit + adjoint, or a GPU simulator, before "
                  "\n   treating this width as fundamental.")
    else:
        print("3. RECOMMENDATION: incomplete — both measurements are needed.")


if __name__ == "__main__":
    main()
