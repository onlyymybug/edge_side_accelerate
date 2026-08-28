#!/usr/bin/env python3
"""Re-score an existing HammerBench run without rerunning phone inference."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


QWEN_CODE = Path(__file__).resolve().parents[1] / "qwen"
sys.path.insert(0, str(QWEN_CODE))

import run_hammerbench as hammer  # noqa: E402
import run_qnn_math_benchmarks as qrun  # noqa: E402


BACKUP_SUFFIX = ".pre_official_metrics_v3"
BACKUP_FILES = (
    "manifest.json",
    "rows.jsonl",
    "summary_metrics.csv",
    "evaluation_summary.csv",
    "deadline_summary.csv",
    "thermal_summary_metrics.csv",
    "inference_results.json",
)


def atomic_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def backup_outputs(root: Path) -> None:
    for name in BACKUP_FILES:
        source = root / name
        backup = root / f"{name}{BACKUP_SUFFIX}"
        if source.is_file() and not backup.exists():
            shutil.copy2(source, backup)


def completion_for(
    root: Path,
    sample_id: str,
    inference_by_id: dict[str, str],
    runner_model_profile: str,
) -> str:
    sample_root = root / hammer.TASK_NAME.lower() / sample_id
    generated_path = sample_root / "generated_text.txt"
    prompt_path = sample_root / "prompt.txt"
    if generated_path.is_file() and prompt_path.is_file():
        completion = qrun.completion_from_output(
            generated_path.read_text(encoding="utf-8", errors="replace"),
            prompt_path.read_text(encoding="utf-8", errors="replace"),
            runner_model_profile=runner_model_profile,
        )
        (sample_root / "completion.txt").write_text(completion, encoding="utf-8")
        return completion
    completion_path = sample_root / "completion.txt"
    if completion_path.is_file():
        return completion_path.read_text(encoding="utf-8", errors="replace")
    return inference_by_id.get(sample_id, "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="allow re-scoring fewer rows than requested by the manifest",
    )
    args = parser.parse_args()

    root = args.result_dir.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        parser.error(f"HammerBench manifest is missing: {manifest_path}")
    manifest = qrun.load_json(manifest_path)
    dataset_path = Path(str(manifest["dataset"])).resolve()
    language = str(manifest["language"])
    split = str(manifest["split"])
    samples = hammer.load_samples(dataset_path, language, split)
    selected_types = set(manifest.get("types", []))
    if selected_types:
        samples = [
            sample for sample in samples if sample["hammer_type"] in selected_types
        ]
    limit = manifest.get("limit")
    if limit is not None:
        samples = samples[: int(limit)]

    rows_by_id = hammer.load_journal(root / "rows.jsonl")
    if not rows_by_id and (root / "summary_metrics.csv").is_file():
        rows_by_id = {
            str(row["sample_id"]): row
            for row in qrun.load_csv(root / "summary_metrics.csv")
        }
    expected = int(manifest.get("requested_samples", len(samples)))
    if not args.allow_partial and len(rows_by_id) != expected:
        parser.error(
            f"run is incomplete: found {len(rows_by_id)} rows, expected {expected}; "
            "use --allow-partial only for an intentional checkpoint"
        )

    inference_by_id: dict[str, str] = {}
    inference_path = root / "inference_results.json"
    if inference_path.is_file():
        inference_by_id = {
            str(row.get("id", "")): str(row.get("predict", ""))
            for row in qrun.load_json(inference_path)
        }
    runner_model_profile = (
        "bitnet"
        if str(manifest.get("qnn_version", "")) == "2.28"
        else str(manifest.get("decoder_model_version", "qwen")).split("_")[0]
    )

    samples_by_id = {str(sample["sample_id"]): sample for sample in samples}
    rescored_by_id: dict[str, dict[str, object]] = {}
    for sample_id, sample in samples_by_id.items():
        if sample_id not in rows_by_id:
            continue
        completion = completion_for(
            root, sample_id, inference_by_id, runner_model_profile
        )
        row = dict(rows_by_id[sample_id])
        row.update(hammer.score_output(hammer.TASK_NAME, sample, completion, {}))
        rescored_by_id[sample_id] = row
    rows = hammer.ordered_rows(rescored_by_id, samples)

    backup_outputs(root)
    atomic_jsonl(root / "rows.jsonl", rows)
    deadlines_ms = [int(value) for value in manifest.get("deadlines_ms", [])]
    if not deadlines_ms:
        deadlines_ms = [
            int(value) for value in str(qrun.DEFAULT_DEADLINES_MS).split(",")
        ]
    hammer.write_outputs(
        root,
        rows,
        samples_by_id,
        deadlines_ms,
        int(manifest.get("dataset_selected_samples", expected)),
    )
    manifest["rescored_at"] = datetime.now().astimezone().isoformat()
    manifest["scoring"] = {
        "version": hammer.SCORING_VERSION,
        "function": "exact name match",
        "arguments": "HammerBench Rouge-L F1 >= 0.7 policy",
        "irrelevant": f"output contains the official rejection: {hammer.REJECTION_TEXT}",
        "end_to_end": (
            "irrelevant_correct for single-turn ir-*; otherwise "
            "function_correct and arguments_correct"
        ),
        "multi_turn_progress_rate": "correct prefix turns / conversation turns",
        "multi_turn_success_rate": "all snapshots in a conversation are correct",
    }
    hammer.atomic_json(manifest_path, manifest)

    correct = sum(hammer.truth(row.get("score_correct")) for row in rows)
    irrelevant = [row for row in rows if hammer.truth(row.get("irrelevant_expected"))]
    irrelevant_correct = sum(
        hammer.truth(row.get("irrelevant_correct")) for row in irrelevant
    )
    print(f"Re-scored {len(rows)} rows in {root}")
    print(f"Benchmark accuracy: {correct}/{len(rows)} = {correct / len(rows):.6f}")
    if irrelevant:
        print(
            "Function-irrelevant accuracy: "
            f"{irrelevant_correct}/{len(irrelevant)} = "
            f"{irrelevant_correct / len(irrelevant):.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
