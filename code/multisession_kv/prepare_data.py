#!/usr/bin/env python3
"""Inventory local HammerBench and BFCL V4 sources without modifying them."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def load(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError(f"JSON root is not an array: {path}")
        return value
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def hammer_stats(path: Path) -> dict[str, Any]:
    rows = load(path)
    sessions: dict[tuple[str, str], list[int]] = defaultdict(list)
    malformed = 0
    for row in rows:
        try:
            kind, conversation, turn = str(row["id"]).rsplit("_", 2)
            sessions[(kind, conversation)].append(int(turn))
        except Exception:
            malformed += 1
    complete = {
        key: turns
        for key, turns in sessions.items()
        if sorted(turns) == list(range(1, max(turns) + 1))
    }
    return {
        "format": "json_array",
        "rows": len(rows),
        "sessions": len(sessions),
        "complete_sessions": len(complete),
        "complete_multi_turn_sessions": sum(len(turns) >= 2 for turns in complete.values()),
        "turn_histogram": dict(sorted(Counter(len(turns) for turns in complete.values()).items())),
        "malformed_rows": malformed,
    }


def bfcl_stats(path: Path) -> dict[str, Any]:
    rows = load(path)
    turn_counts = []
    field_counts: Counter[str] = Counter()
    for row in rows:
        field_counts.update(row.keys())
        question = row.get("question", [])
        turn_counts.append(len(question) if isinstance(question, list) else 0)
    return {
        "format": "jsonl",
        "rows": len(rows),
        "unique_ids": len({str(row.get("id")) for row in rows}),
        "turn_histogram": dict(sorted(Counter(turn_counts).items())),
        "field_presence": dict(sorted(field_counts.items())),
    }


def file_record(path: Path, stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "local_mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        **stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    hammer_repo = Path(cfg["hammerbench"]["repository"]).resolve()
    bfcl_repo = Path(cfg["bfcl"]["repository"]).resolve()
    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "download_time": None,
        "download_time_note": "not recoverable from existing local checkouts",
        "hammerbench": {
            "repository": str(hammer_repo),
            "commit": commit(hammer_repo),
            "license": "Apache-2.0",
            "license_evidence": str(hammer_repo / "LICENSE"),
            "multi_turn": file_record(
                Path(cfg["hammerbench"]["dataset"]).resolve(),
                hammer_stats(Path(cfg["hammerbench"]["dataset"]).resolve()),
            ),
        },
        "bfcl_v4": {
            "repository": str(bfcl_repo),
            "commit": commit(bfcl_repo),
            "license": "Apache-2.0",
            "license_evidence": str(bfcl_repo / "pyproject.toml"),
            "base": file_record(
                Path(cfg["bfcl"]["base"]).resolve(),
                bfcl_stats(Path(cfg["bfcl"]["base"]).resolve()),
            ),
            "long_context": file_record(
                Path(cfg["bfcl"]["long_context"]).resolve(),
                bfcl_stats(Path(cfg["bfcl"]["long_context"]).resolve()),
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
