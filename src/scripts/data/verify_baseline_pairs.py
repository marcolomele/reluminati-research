"""
Verify baseline_egoexo_pairs.json and baseline_exoego_pairs.json:
  - Check entry counts and mirror property
  - Report unique UIDs and takes-per-scenario breakdown
  - Spot-check first entries

Usage:
    python src/scripts/data/verify_baseline_pairs.py
"""

import json
from collections import defaultdict
from pathlib import Path

SCENARIO_KEYWORDS = {
    "basketball": [
        "basketball", "drills", "layup", "mikan", "jump shooting",
        "mid-range", "reverse layup",
    ],
    "bike repair": [
        "wheel", "tire", "flat", "tube", "chain", "lubricate",
        "derailleur", "derailueur", "bike",
    ],
    "cooking": [
        "cooking", "making", "omelet", "scrambled", "eggs", "noodles",
        "pasta", "salad", "greek", "cucumber", "tomato", "sesame",
        "ginger", "milk tea", "chai tea", "chai", "coffee", "latte",
    ],
    "music": [
        "piano", "violin", "guitar", "suzuki", "playing", "scales",
        "arpeggios", "freeplaying",
    ],
    "health": [
        "first aid", "cpr", "covid", "antigen", "rapid", "test",
    ],
    "dance": ["dance", "dancing"],
    "soccer": ["soccer", "football"],
    "rock climbing": ["rock climbing", "climbing"],
}

RELATIONS_FILES = [
    "relations_test.json",
    "relations_train.json",
    "relations_val.json",
]


def classify_scenario(title: str) -> str | None:
    title_lower = title.lower()
    for scenario, keywords in SCENARIO_KEYWORDS.items():
        for kw in keywords:
            if kw in title_lower:
                return scenario
    return None


def main():
    root = Path(__file__).resolve().parents[3]

    # Load relations for UID -> scenario lookup
    uid_scenario_title: dict[str, str] = {}
    for fname in RELATIONS_FILES:
        data = json.loads((root / fname).read_text())
        for uid, ann in data["annotations"].items():
            uid_scenario_title[uid] = ann["scenario"]

    egoexo = json.loads((root / "baseline_egoexo_pairs.json").read_text())
    exoego = json.loads((root / "baseline_exoego_pairs.json").read_text())

    print(f"baseline_egoexo_pairs.json: {len(egoexo)} entries")
    print(f"baseline_exoego_pairs.json: {len(exoego)} entries")
    print(f"Same length: {len(egoexo) == len(exoego)}")

    # Check mirror property
    mirror_ok = all(
        ego[0] == exo[2] and ego[1] == exo[3]
        and ego[2] == exo[0] and ego[3] == exo[1]
        for ego, exo in zip(egoexo, exoego)
    )
    print(f"Mirror property holds: {mirror_ok}")

    # Unique UIDs and scenario breakdown
    uids: set[str] = set()
    for p in egoexo:
        uids.add(p[0].split("//")[1])

    print(f"\nUnique UIDs in baseline: {len(uids)}")

    scenario_takes: dict[str, int] = defaultdict(int)
    uid_cat: dict[str, str | None] = {}
    for uid in uids:
        title = uid_scenario_title.get(uid, "UNKNOWN")
        cat = classify_scenario(title)
        scenario_takes[cat] += 1
        uid_cat[uid] = cat

    print("\nTakes per scenario:")
    for s in sorted(scenario_takes, key=lambda x: x or ""):
        print(f"  {s}: {scenario_takes[s]}")

    # Pairs per scenario
    scenario_pairs: dict[str | None, int] = defaultdict(int)
    for p in egoexo:
        uid = p[0].split("//")[1]
        scenario_pairs[uid_cat[uid]] += 1

    print("\nPairs per scenario:")
    for s in sorted(scenario_pairs, key=lambda x: x or ""):
        print(f"  {s}: {scenario_pairs[s]}")

    # Spot-check first 2 entries
    print("\n--- Spot check: first 2 egoexo entries ---")
    for i in range(min(2, len(egoexo))):
        print(f"  egoexo[{i}]: {egoexo[i]}")
        print(f"  exoego[{i}]: {exoego[i]}")
        print()


if __name__ == "__main__":
    main()
