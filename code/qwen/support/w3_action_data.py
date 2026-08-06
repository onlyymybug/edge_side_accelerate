import json
from pathlib import Path
from typing import Any

def read_action_samples(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Action samples file not found: {path}")

    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in ["sample_id", "request", "gold_action", "gold_arguments"]:
                if key not in row:
                    raise ValueError(f"Missing '{key}' in {path}:{line_no}")
            if not isinstance(row["gold_arguments"], dict):
                raise ValueError(f"'gold_arguments' must be an object in {path}:{line_no}")
            row.setdefault("dataset", "tiny_schema_v1")
            row.setdefault("difficulty", "unknown")
            samples.append(row)
    return samples

def build_action_prompt(
    sample: dict[str, Any],
    prompt_template: str,
    action_schema: dict[str, Any],
) -> str:
    schema_json = json.dumps(action_schema, ensure_ascii=False, indent=2, sort_keys=True)
    return prompt_template.format(
        schema_json=schema_json,
        request=sample["request"],
    )