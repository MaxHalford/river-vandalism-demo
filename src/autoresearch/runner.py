"""Subprocess runner for backtesting a candidate model.

Spawned by the orchestrator. Sets RLIMITs, validates the candidate code
against the AST allowlist, imports it, then replays the pre-fetched event
list through `replay_rows`. Writes the result to a JSON file the parent reads.

Invoke:
    python -m src.autoresearch.runner <code_file> <data_pickle> <out_json>
"""

from __future__ import annotations

import json
import pickle
import resource
import sys
import traceback

from src.autoresearch.sandbox import validate
from src.backtest.replay import Candidate, replay_rows


def _set_limits():
    one_gib = 1024 * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (one_gib, one_gib))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    except (ValueError, OSError):
        pass


def _write(out_path: str, payload: dict) -> None:
    with open(out_path, "w") as f:
        json.dump(payload, f)


def main():
    code_path, data_path, out_path = sys.argv[1:4]
    _set_limits()

    code = open(code_path).read()
    errors = validate(code)
    if errors:
        _write(out_path, {"status": "rejected_sandbox", "errors": errors})
        return

    try:
        ns: dict = {"__name__": "candidate", "__builtins__": __builtins__}
        exec(compile(code, code_path, "exec"), ns)
    except Exception:
        _write(out_path, {"status": "rejected_runtime", "error": traceback.format_exc()})
        return

    make_model = ns.get("make_model")
    if not callable(make_model):
        _write(out_path, {"status": "rejected_sandbox", "errors": ["missing callable make_model()"]})
        return
    featurize = ns.get("featurize")  # may be None

    try:
        with open(data_path, "rb") as f:
            rows = pickle.load(f)
        cand = Candidate(name="candidate", factory=make_model)
        if featurize is not None:
            # The replay engine calls featurize(row); the candidate signature
            # takes (event, state). For backtest we don't have streaming state,
            # so we pass an empty dict. The row carries the raw event fields.
            cand.featurize = lambda row, _f=featurize: _f(dict(row), {})
        result = replay_rows(rows, cand)
        _write(
            out_path,
            {
                "status": "ok",
                "final_rocauc": result.final_rocauc,
                "n_predicted": result.n_predicted,
                "n_learned": result.n_learned,
                "timeline": result.timeline,
                "per_prediction": result.per_prediction,
            },
        )
    except Exception:
        _write(out_path, {"status": "rejected_runtime", "error": traceback.format_exc()})


if __name__ == "__main__":
    main()
