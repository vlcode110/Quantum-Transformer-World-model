"""Ablation sweep runner.

The grid below is the Phase 1 ablation list from the research plan. It is
written as one dict so adding an axis is a one-line change; `--grid` selects a
named preset, `--dry-run` prints the plan without spending compute.

Usage:
    python ablate.py --dry_run                  # see the plan and its cost
    python ablate.py --grid encoding            # one axis
    python ablate.py --grid full --seeds 0 1 2  # everything
"""
import argparse
import itertools
import json
import os
import time

from config import Config
from dataset import loaders
from train_dynamics import train_one

# DECISION: each entry is an ablation axis. Values are swept one axis at a time
# against the defaults in config.py (OFAT), not as a full cross product — a full
# cross product of these is thousands of runs and buys little.
GRIDS = {
    "encoding":  {"encoding": ["angle", "angle_reupload", "amplitude"]},
    "width":     {"n_qubits": [4, 6, 8, 10]},
    "depth":     {"q_layers": [2, 4, 6, 8]},
    "readout":   {"readout": ["local", "global"]},
    # Entanglement topology is a measured axis now, not an unexamined default:
    # brickwork vs ring_cascade changed local gradient variance by 3 orders of
    # magnitude at 10 qubits in the Phase 0 probe. Worth confirming that the
    # trainability difference actually shows up in task performance.
    "entangler": {"entangler": ["brickwork", "ring_cascade"]},
    "position":  {"quantum_layer_position": [0, -1]},
    "context":   {"context": [4, 8, 16]},
    # MEASURED: on CartPole the 1-step error floor (~0.0010) is reached by
    # ~18k parameters; 79k buys nothing (0.00099). Comparing architectures AT
    # the floor is uninformative — everything ties. The discriminating regime
    # is the capacity-limited one below it (1.9k params -> 0.0035), so this
    # axis sweeps model width to build an error-vs-capacity curve per
    # architecture. If the quantum curve sits left of the classical one, that
    # is the parameter-efficiency claim in metric 7.3, measured rather than
    # asserted.
    "capacity":  {"d_model": [8, 12, 16, 24, 32]},
    "action":    {"action_conditioning": ["token", "concat", "circuit"]},
    "latent":    {"latent_mode": ["compressed", "raw"]},
}
GRIDS["full"] = {k: v for g in GRIDS.values() for k, v in g.items()}

# axes that change the DATASET (not just the model) — these force a rebuild
DATA_AXES = {"context", "latent_mode"}


def plan(grid_name, models, seeds):
    grid = GRIDS[grid_name]
    runs = []
    for axis, values in grid.items():
        for v in values:
            for m in models:
                for s in seeds:
                    runs.append({"axis": axis, "value": v, "model": m, "seed": s})
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="encoding", choices=list(GRIDS))
    ap.add_argument("--models", nargs="+",
                    default=["quantum_transformer", "classical_transformer"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--out", default="results/ablation.jsonl")
    args, rest = ap.parse_known_args()

    runs = plan(args.grid, args.models, args.seeds)
    print(f"[ablate] grid '{args.grid}': {len(runs)} runs")
    for r in runs[:12]:
        print(f"   {r['axis']}={r['value']:<16} {r['model']:<22} seed {r['seed']}")
    if len(runs) > 12:
        print(f"   ... and {len(runs)-12} more")
    if args.dry_run:
        print("\n[ablate] dry run — nothing executed.\n"
              "         Time one run first (train_dynamics.py prints wall_clock_s),\n"
              "         multiply, and decide before launching.")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cache = {}
    t0 = time.time()
    for i, r in enumerate(runs, 1):
        cfg = Config.from_cli(rest)
        setattr(cfg, r["axis"], r["value"])
        cfg.seed = r["seed"]
        key = (cfg.context, cfg.latent_mode, cfg.seed)
        if key not in cache:                    # rebuild data only when needed
            cache[key] = loaders(cfg, verbose=False)
        dl, info = cache[key]
        print(f"\n[{i}/{len(runs)}] {r['axis']}={r['value']} "
              f"{r['model']} seed={r['seed']}")
        try:
            res = train_one(r["model"], cfg, dl, info, verbose=False)
            res["ablation_axis"], res["ablation_value"] = r["axis"], str(r["value"])
            with open(args.out, "a") as f:
                f.write(json.dumps(res) + "\n")
            print(f"      1-step {res['test_rollout']['1']:.5f}  "
                  f"{cfg.gate_k}-step {res['test_rollout'][str(cfg.gate_k)]:.5f}  "
                  f"({res['wall_clock_s']:.0f}s)")
        except Exception as e:                  # keep the sweep alive
            print(f"      FAILED: {type(e).__name__}: {e}")
            with open(args.out, "a") as f:
                f.write(json.dumps({"failed": True, **r, "error": str(e)}) + "\n")
    print(f"\n[ablate] {len(runs)} runs in {(time.time()-t0)/60:.1f} min "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
