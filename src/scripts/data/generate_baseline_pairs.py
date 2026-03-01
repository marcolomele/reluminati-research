"""
Generate baseline_egoexo_pairs.json and baseline_exoego_pairs.json
by sampling 10 takes per scenario from the full pairs data.

Usage:
    python src/scripts/data/generate_baseline_pairs.py
"""

import json
import random
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

PAIRS_FILES = [
    "test_egoexo_pairs.json",
    "train_egoexo_pairs.json",
    "val_egoexo_pairs.json",
]

RELATIONS_FILES = [
    "relations_test.json",
    "relations_train.json",
    "relations_val.json",
]

SEED = 42
SAMPLES_PER_SCENARIO = 10


def classify_scenario(title: str) -> str | None:
    title_lower = title.lower()
    for scenario, keywords in SCENARIO_KEYWORDS.items():
        for kw in keywords:
            if kw in title_lower:
                return scenario
    return None


def main():
    root = Path(__file__).resolve().parents[3]

    # Step 1: Load pairs and build uid -> list of pairs
    uid_pairs: dict[str, list] = defaultdict(list)
    for fname in PAIRS_FILES:
        for pair in json.loads((root / fname).read_text()):
            uid = pair[0].split("//")[1]
            uid_pairs[uid].append(pair)

    print(f"Total unique UIDs in pairs files: {len(uid_pairs)}")

    # Step 2: Load relations, classify each UID into a scenario category
    uid_scenario: dict[str, str] = {}
    unmapped_titles: set[str] = set()

    for fname in RELATIONS_FILES:
        data = json.loads((root / fname).read_text())
        for uid, ann in data["annotations"].items():
            cat = classify_scenario(ann["scenario"])
            if cat:
                uid_scenario[uid] = cat
            else:
                unmapped_titles.add(ann["scenario"])

    if unmapped_titles:
        print(f"WARNING: unmapped scenario titles: {unmapped_titles}")
    print(f"UIDs with classified scenario: {len(uid_scenario)}")

    # Step 3: Group UIDs (present in pairs) by scenario, sample 10 per scenario
    scenario_uids: dict[str, list[str]] = defaultdict(list)
    for uid in uid_pairs:
        if uid in uid_scenario:
            scenario_uids[uid_scenario[uid]].append(uid)

    print("\nScenarios found in pairs data:")
    for scenario in sorted(scenario_uids):
        print(f"  {scenario}: {len(scenario_uids[scenario])} UIDs")

    random.seed(SEED)
    sampled_uids: set[str] = set()
    for scenario, uids in sorted(scenario_uids.items()):
        sample = random.sample(uids, min(SAMPLES_PER_SCENARIO, len(uids)))
        sampled_uids.update(sample)
        print(f"  Sampled {len(sample)} from '{scenario}'")

    print(f"\nTotal sampled UIDs: {len(sampled_uids)}")

    # Step 4: Collect all pairs for sampled UIDs
    egoexo = [p for uid in sorted(sampled_uids) for p in uid_pairs[uid]]
    print(f"Total egoexo pairs: {len(egoexo)}")

    # Step 5: Generate mirror (exo-ego)
    exoego = [[p[2], p[3], p[0], p[1]] for p in egoexo]

    out_egoexo = root / "baseline_egoexo_pairs.json"
    out_exoego = root / "baseline_exoego_pairs.json"

    with open(out_egoexo, "w") as f:
        json.dump(egoexo, f)
    with open(out_exoego, "w") as f:
        json.dump(exoego, f)

    print(f"\nWrote {out_egoexo}")
    print(f"Wrote {out_exoego}")


if __name__ == "__main__":
    main()
