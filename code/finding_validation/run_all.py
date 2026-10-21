#!/usr/bin/env python3
"""One-command reproduction: rescore when needed, evaluate, report, and test."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def scoring_ready(path: Path, target: str) -> bool:
    if not path.is_file():
        return False
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return bool(rows) and all(
        row.get("scorer_version") == target
        and str(row.get("score_correct", "")).strip().lower() in {"true", "false", "0", "1", "0.0", "1.0"}
        for row in rows
    )


def directories_needing_rescore(config: dict[str, Any]) -> list[Path]:
    data_root = Path(config["data_root"])
    allowed = set(config["allowed_budgets"])
    target = config["target_scorer_version"]
    directories = []
    for runtime, runtime_config in config["runtimes"].items():
        for task in config["tasks"]:
            root = data_root / runtime / config["device"] / task / runtime_config["model_dir"]
            for directory in sorted(root.glob("*/output_budget_*")) if root.exists() else []:
                if directory.name.removeprefix("output_budget_") not in allowed:
                    continue
                preferred = directory / "summary.rescored_v2.csv"
                source = preferred if preferred.exists() else directory / "summary.csv"
                if source.exists() and not scoring_ready(source, target):
                    directories.append(directory)
    return directories


def run(command: list[str], *, env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> None:
    here = Path(__file__).resolve().parent
    repository = here.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=here / "config.json")
    parser.add_argument("--skip-rescore", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    env = os.environ.copy()

    if not args.skip_rescore:
        needs_rescore = directories_needing_rescore(config)
        if needs_rescore:
            rescore_script = repository / "code/QNN/scripts/rescore_math_benchmarks.py"
            run([sys.executable, str(rescore_script), *map(str, needs_rescore)], env=env)
        else:
            print("All scoped scoring files already use the target scorer.")

    run([sys.executable, str(here / "run.py"), "--config", str(args.config)], env=env)
    run([sys.executable, str(here / "report.py"), "--config", str(args.config)], env=env)
    run([sys.executable, str(here / "test_run.py")], env=env)


if __name__ == "__main__":
    main()
