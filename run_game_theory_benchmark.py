"""Collect the JEV game-theory benchmark and write a tidy result table.

Re-running is cheap and safe: answers are cached on disk by request content, so
an interrupted run resumes and an unchanged design costs nothing.
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from jev_bench import games as G
from jev_bench.client import DecisionsClient, atomic_write, digest
from jev_bench.design import build_trials

MODEL = "typesafe/jev-1.13"
RUN_ROOT = Path("jev_bench_runs")

META_COLUMNS = [
    "trial_id", "block", "family", "T", "R", "P", "S", "frame", "coop_key",
    "instruction", "coop_first", "horizon", "rounds", "delta", "opponent",
    "repetition", "dominant", "best_response", "grim_threshold",
    "greed", "fear", "transform", "payoff_set", "base_family", "record_key",
]


def main() -> Path:
    trials = build_trials()
    design_id = digest([t["record_key"] for t in trials])[:16]
    run_dir = RUN_ROOT / f"game-theory-{design_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run directory: {run_dir}")

    client = DecisionsClient(run_dir, model=MODEL, requests_per_minute=900,
                             max_workers=10)
    client.collect(trials)

    rows = []
    for trial in trials:
        record = json.loads(client.record_path(trial).read_text()) \
            if client.record_path(trial).exists() else {"status": "missing"}
        row = {key: trial.get(key) for key in META_COLUMNS}
        row["status"] = record.get("status", "missing")
        if row["status"] == "ok":
            row.update(G.read_cooperation(record["answers"], trial["coop_key"]))
            row["latency_seconds"] = record.get("latency_seconds")
            row["cost"] = (record.get("usage") or {}).get("cost")
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(run_dir / "trials.csv", index=False)

    ok = frame[frame.status == "ok"]
    print(f"\n{len(ok):,} of {len(frame):,} trials collected "
          f"({len(ok) / len(frame):.2%})")
    if len(ok) < len(frame):
        print(frame[frame.status != "ok"].status.value_counts().to_string())

    # Cost is billed per distinct request, not per trial row.
    distinct = ok.drop_duplicates("record_key")
    print(f"distinct requests {len(distinct):,}  "
          f"total cost ${distinct['cost'].sum():.4f}  "
          f"median latency {ok['latency_seconds'].median():.2f}s")

    atomic_write(run_dir / "manifest.json", {
        "benchmark": "game_theory_2x2",
        "version": "1.0",
        "model": MODEL,
        "design_id": design_id,
        "trials": len(frame),
        "distinct_requests": int(frame["record_key"].nunique()),
        "collected_ok": int(len(ok)),
        "total_cost_usd": float(distinct["cost"].sum()),
        "blocks": frame["block"].value_counts().to_dict(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
    })
    print(f"\nwrote {run_dir / 'trials.csv'}")
    return run_dir


if __name__ == "__main__":
    main()
