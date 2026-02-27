"""
Compute take and pair counts per scenario for the baseline set.

Outputs (CSV):
  - baseline_takes_per_scenario.csv   (scenario, n_takes)
  - baseline_pairs_per_scenario.csv   (scenario, n_pairs)
  - baseline_take_pair_summary.csv    (scenario, take_uid, n_pairs)

Usage:
    python src/scripts/data/analyze_baseline_counts.py
    python src/scripts/data/analyze_baseline_counts.py --root /path/to/data --output-dir /path/to/output
"""

import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _baseline_common import (
    load_baseline_pairs,
    load_all_annotations,
    build_uid_scenario_map,
)


def main():
    parser = argparse.ArgumentParser(description="Baseline pair/take counts per scenario")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3],
                        help="Root dir containing baseline_egoexo_pairs.json and relations_*.json")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for CSVs (default: <root>/baseline_analysis)")
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = (args.output_dir or root / "baseline_analysis").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Root: {root}")
    print(f"Output: {out_dir}")

    pairs = load_baseline_pairs(root)
    annotations = load_all_annotations(root)
    uid_scenario = build_uid_scenario_map(annotations)

    # Per-UID pair counts
    uid_pair_count: dict[str, int] = defaultdict(int)
    for p in pairs:
        uid = p[0].split("//")[1]
        uid_pair_count[uid] += 1

    # Build per-take summary
    take_rows = []
    for uid, n_pairs in sorted(uid_pair_count.items()):
        scenario = uid_scenario.get(uid, "unknown")
        take_rows.append({"scenario": scenario, "take_uid": uid, "n_pairs": n_pairs})

    # Aggregate takes per scenario
    scenario_takes: dict[str, int] = defaultdict(int)
    scenario_pairs: dict[str, int] = defaultdict(int)
    for row in take_rows:
        scenario_takes[row["scenario"]] += 1
        scenario_pairs[row["scenario"]] += row["n_pairs"]

    # Write baseline_takes_per_scenario.csv
    path_takes = out_dir / "baseline_takes_per_scenario.csv"
    with open(path_takes, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "n_takes"])
        w.writeheader()
        for s in sorted(scenario_takes):
            w.writerow({"scenario": s, "n_takes": scenario_takes[s]})
    print(f"Wrote {path_takes} ({len(scenario_takes)} rows)")

    # Write baseline_pairs_per_scenario.csv
    path_pairs = out_dir / "baseline_pairs_per_scenario.csv"
    with open(path_pairs, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "n_pairs"])
        w.writeheader()
        for s in sorted(scenario_pairs):
            w.writerow({"scenario": s, "n_pairs": scenario_pairs[s]})
    print(f"Wrote {path_pairs} ({len(scenario_pairs)} rows)")

    # Write baseline_take_pair_summary.csv
    path_summary = out_dir / "baseline_take_pair_summary.csv"
    with open(path_summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "take_uid", "n_pairs"])
        w.writeheader()
        for row in sorted(take_rows, key=lambda r: (r["scenario"], r["take_uid"])):
            w.writerow(row)
    print(f"Wrote {path_summary} ({len(take_rows)} rows)")


if __name__ == "__main__":
    main()
