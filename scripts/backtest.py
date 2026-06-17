"""Replay history through one or more candidate models and write CSVs.

Usage:

    uv run python scripts/backtest.py \\
        --since 2026-06-10T00:00:00Z \\
        --until 2026-06-17T00:00:00Z \\
        --candidates default,arf,sgd \\
        --out data/backtest

Writes:
    {out}/{candidate}.timeline.csv   rolling AUC/logloss/brier over wall-clock
    {out}/{candidate}.preds.csv      every prediction (rev_id, edit_ts, score)
    {out}/summary.csv                one row per candidate, final metrics
"""

import argparse
import asyncio
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import candidates as cand_mod
from src.backtest.replay import replay
from src.common import store


def _parse_ts(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


async def _main(args):
    pool = await store.open_pool()
    try:
        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        names = [n.strip() for n in args.candidates.split(",") if n.strip()]
        summary_rows = []
        for name in names:
            if name not in cand_mod.CANDIDATES:
                print(f"unknown candidate: {name}", file=sys.stderr)
                continue
            print(f"replaying {name}...")
            result = await replay(
                pool,
                cand_mod.CANDIDATES[name],
                since=_parse_ts(args.since),
                until=_parse_ts(args.until),
                window=args.window,
            )
            with (outdir / f"{name}.timeline.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["ts", "n", "rocauc", "logloss", "brier"])
                w.writeheader()
                w.writerows(result.timeline)
            with (outdir / f"{name}.preds.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["rev_id", "edit_ts", "score"])
                w.writeheader()
                w.writerows(result.per_prediction)
            summary_rows.append(
                {
                    "candidate": name,
                    "n_predicted": result.n_predicted,
                    "n_learned": result.n_learned,
                    "final_rocauc": result.final_rocauc,
                }
            )
            print(f"  n_predicted={result.n_predicted} n_learned={result.n_learned} "
                  f"final_rocauc={result.final_rocauc}")
        with (outdir / "summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["candidate", "n_predicted", "n_learned", "final_rocauc"]
            )
            w.writeheader()
            w.writerows(summary_rows)
        print(f"wrote {outdir}/")
    finally:
        await pool.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--since", default=(datetime.now(tz=timezone.utc).isoformat()))
    p.add_argument("--until", default=(datetime.now(tz=timezone.utc).isoformat()))
    p.add_argument("--candidates", default="default")
    p.add_argument("--window", type=int, default=2000)
    p.add_argument("--out", default="data/backtest")
    asyncio.run(_main(p.parse_args()))


if __name__ == "__main__":
    main()
