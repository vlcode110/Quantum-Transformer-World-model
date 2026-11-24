"""Consolidate every result in the project into one file, for audit.

Written because the paper's central argument is that quantum ML results are
reported without enough care, which makes it unacceptable for any number in it
to come from a console log remembered second-hand. Two figures quoted during
this project have already turned out wrong. So every statistic gets recomputed
from the run directories, and this script is step one: gather the raw material
into a single place.

It reads rather than writes, apart from the one output file, so it cannot
damage any existing run.

    python collect_all.py

Writes AUDIT_RAW.json next to itself. Long training histories are reduced to
summary statistics so the file stays small; per-run final metrics are kept in
full because those are what the paper quotes.

If files live in iCloud and have been evicted, reads fail with OSError
("Resource deadlock avoided"). The script retries, and reports anything it
could not open rather than silently omitting it — a partial audit that looks
complete would be worse than no audit.
"""
import csv
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")
DOCS = os.path.join(HOME, "Documents")

# (label, absolute glob) — every place this project stores a result
SOURCES = [
    ("dreamer",  os.path.join(DOCS, "DreamerV3", "runs", "*", "*.json")),
    ("phase1",   os.path.join(HERE, "Phase_1_Scaffold_test", "runs", "*", "result.json")),
    ("phase1_ablation", os.path.join(HERE, "Phase_1_Scaffold_test", "results", "*.jsonl")),
    ("phase2",   os.path.join(HERE, "Phase_2_Closed_Loop", "runs", "*", "*.json")),
    ("phase3",   os.path.join(HERE, "Phase_3_Quantum_Native", "runs", "*", "result.json")),
    ("qasa_csv", os.path.join(DOCS, "QASA-repro", "results", "*.csv")),
]

MAX_ATTEMPTS = 4
failures = []


def read_text(path):
    """Read with retries: an evicted iCloud file often materialises on retry."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            with open(path, "r") as f:
                return f.read()
        except OSError as e:
            if attempt == MAX_ATTEMPTS - 1:
                failures.append({"path": path, "error": f"{type(e).__name__}: {e}"})
                return None
            time.sleep(0.6 * (attempt + 1))
    return None


def summarise_series(v):
    """A training curve is evidence of convergence, not a quoted number, so it
    is kept as shape and endpoints rather than in full."""
    nums = [x for x in v if isinstance(x, (int, float))]
    if not nums:
        return {"n": len(v), "note": "non-numeric"}
    return {"n": len(nums), "first": nums[0], "last": nums[-1],
            "min": min(nums), "max": max(nums)}


def compact(obj, depth=0):
    """Drop bulk, keep every scalar. Scalars are what get quoted."""
    if isinstance(obj, dict):
        return {k: compact(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > 12 and all(isinstance(x, (int, float)) for x in obj):
            return summarise_series(obj)
        if len(obj) > 12 and all(isinstance(x, dict) for x in obj):
            return {"n_records": len(obj), "first": compact(obj[0], depth + 1),
                    "last": compact(obj[-1], depth + 1)}
        return [compact(x, depth + 1) for x in obj[:12]]
    return obj


def load_json_file(path):
    txt = read_text(path)
    if txt is None:
        return None
    try:
        return compact(json.loads(txt))
    except Exception as e:
        failures.append({"path": path, "error": f"parse: {type(e).__name__}: {e}"})
        return None


def load_jsonl_file(path):
    txt = read_text(path)
    if txt is None:
        return None
    out = []
    for i, line in enumerate(txt.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception as e:
            failures.append({"path": path, "error": f"line {i}: {e}"})
    return out


def load_csv_file(path):
    txt = read_text(path)
    if txt is None:
        return None
    rows = list(csv.DictReader(txt.splitlines()))
    return {"n_rows": len(rows), "fields": list(rows[0].keys()) if rows else [],
            "rows": rows}


def main():
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "python": sys.version.split()[0], "sources": {}}

    for label, pattern in SOURCES:
        paths = sorted(glob.glob(pattern))
        entries = {}
        for p in paths:
            key = os.path.join(os.path.basename(os.path.dirname(p)),
                               os.path.basename(p))
            if p.endswith(".jsonl"):
                d = load_jsonl_file(p)
            elif p.endswith(".csv"):
                d = load_csv_file(p)
            else:
                d = load_json_file(p)
            if d is not None:
                entries[key] = d
        out["sources"][label] = {"pattern": pattern,
                                 "files_found": len(paths),
                                 "files_read": len(entries),
                                 "entries": entries}
        print(f"  {label:<18} {len(entries):>4} read / {len(paths):>4} found")

    out["failures"] = failures
    dest = os.path.join(HERE, "AUDIT_RAW.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=1, default=str)

    size = os.path.getsize(dest) / 1e6
    print(f"\n-> {dest}  ({size:.1f} MB)")
    if failures:
        print(f"\n{len(failures)} file(s) could not be read. The audit will be "
              f"incomplete until these materialise:")
        seen = set()
        for f_ in failures[:8]:
            d = os.path.basename(os.path.dirname(f_["path"]))
            if d not in seen:
                seen.add(d)
                print(f"    {d}")
        if len(failures) > 8:
            print(f"    ... and {len(failures) - 8} more")
        print("\nIf these are iCloud-evicted, force them down first:")
        print("  find ~/Documents/PhD/qtwm ~/Documents/DreamerV3 -name '*.json' "
              "-exec cat {} + > /dev/null")
        print("then run this script again.")
    else:
        print("\nAll files read. Nothing missing.")


if __name__ == "__main__":
    main()
