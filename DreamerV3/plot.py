"""Plot learning curves from runs/*/log.csv."""
import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    steps, rets = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            rets.append(float(row["eval_return"]))
    return steps, rets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="runs/*/log.csv")
    ap.add_argument("--out", default="learning_curves.png")
    args = ap.parse_args()

    plt.figure(figsize=(8, 5))
    for path in sorted(glob.glob(args.pattern)):
        steps, rets = load(path)
        label = os.path.basename(os.path.dirname(path))
        plt.plot(steps, rets, label=label, marker="o", markersize=3)
    plt.xlabel("Environment steps")
    plt.ylabel("Eval return (mean of 5 episodes)")
    plt.title("Small DreamerV3 — classic control")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
