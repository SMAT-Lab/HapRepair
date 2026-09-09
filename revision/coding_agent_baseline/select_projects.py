#!/usr/bin/env python3
"""Select ten projects with a seeded 2 x 2 LOC-by-alert-density design."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SEED = 20260802
STRATA = ("low_loc__low_density", "low_loc__high_density", "high_loc__low_density", "high_loc__high_density")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-run", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "selected_projects.json")
    args = parser.parse_args()
    if args.count < len(STRATA) * 2:
        raise SystemExit("At least two projects per stratum are required")

    scan_path = args.scan_run.resolve()
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    candidates = [
        item
        for item in scan["projects"]
        if item["status"].startswith("scanned")
        and item["nonblank_source_loc"] > 0
        and item["initial_alerts"] > 10
        and item["alerts_per_kloc"] is not None
    ]
    if len(candidates) < args.count:
        raise SystemExit(f"Only {len(candidates)} eligible projects are available")
    loc_median = statistics.median(item["nonblank_source_loc"] for item in candidates)
    density_median = statistics.median(item["alerts_per_kloc"] for item in candidates)

    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in STRATA}
    for item in candidates:
        loc_group = "low_loc" if item["nonblank_source_loc"] <= loc_median else "high_loc"
        density_group = "low_density" if item["alerts_per_kloc"] <= density_median else "high_density"
        stratum = f"{loc_group}__{density_group}"
        grouped[stratum].append({**item, "stratum": stratum})
    insufficient = {name: len(items) for name, items in grouped.items() if len(items) < 2}
    if insufficient:
        raise SystemExit(f"Insufficient candidates for balanced selection: {insufficient}")

    rng = random.Random(args.seed)
    for items in grouped.values():
        items.sort(key=lambda item: item["name"])
        rng.shuffle(items)

    allocation = {name: 2 for name in STRATA}
    remaining = args.count - sum(allocation.values())
    strata_with_capacity = [name for name in STRATA if len(grouped[name]) > allocation[name]]
    rng.shuffle(strata_with_capacity)
    while remaining:
        progressed = False
        for name in strata_with_capacity:
            if allocation[name] < len(grouped[name]):
                allocation[name] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise SystemExit("Unable to allocate the requested project count")

    selected = []
    for stratum in STRATA:
        selected.extend(grouped[stratum][: allocation[stratum]])
    selected.sort(key=lambda item: (item["stratum"], item["name"]))
    output = {
        "schema_version": 1,
        "experiment": "EXP-AGENT-10",
        "selection_frozen": True,
        "selection_timing": "before HapRepair or coding-agent repair outcomes",
        "seed": args.seed,
        "requested_count": args.count,
        "eligible_candidate_count": len(candidates),
        "loc_median": loc_median,
        "density_median_alerts_per_kloc": density_median,
        "allocation": allocation,
        "scan_manifest": str(scan_path),
        "scan_manifest_sha256": sha256_file(scan_path),
        "projects": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {len(selected)} projects -> {args.output}")
    for item in selected:
        print(
            f"{item['stratum']:24s} {item['name']:45s} "
            f"LOC={item['nonblank_source_loc']:7d} alerts={item['initial_alerts']:5d} "
            f"density={item['alerts_per_kloc']:.2f}"
        )


if __name__ == "__main__":
    main()
