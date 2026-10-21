from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
PACE_ROOT = HERE.parents[1]
CODE_ROOT = PACE_ROOT / "code"
DEFAULT_CONFIG = HERE / "config.json"
PLAN_DIR = HERE / "plans"


def read_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def truth(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def is_length_stop(row: dict[str, str]) -> bool:
    return truth(row.get("seq_len_stop")) or row.get("finish_reason") == "length"


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def stable_id(*parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def load_module(name: str, path: Path):
    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def file_sha256(path_text: str) -> str:
    path = Path(path_text)
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_blocks(tokens: int, width: int) -> int:
    return math.ceil(tokens / width)


def exact_length_chatml(tokenizer, target: int, chatml_builder, *, add_special_tokens: bool = False) -> str:
    """Construct benign ChatML input with exactly target non-special tokens."""
    stem = "Repeat silently and answer OK. Data:"
    for count in range(max(1, target * 3)):
        prompt = chatml_builder(stem + (" x" * count))
        size = len(tokenizer.encode(prompt, add_special_tokens=add_special_tokens).ids)
        if size == target:
            return prompt
        if size > target + 4:
            break
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 .,-_"
    queue = [stem]
    seen = {stem}
    for _ in range(20000):
        text = queue.pop(0)
        prompt = chatml_builder(text)
        size = len(tokenizer.encode(prompt, add_special_tokens=add_special_tokens).ids)
        if size == target:
            return prompt
        if size < target:
            for char in alphabet:
                candidate = text + char
                if candidate not in seen:
                    seen.add(candidate)
                    queue.append(candidate)
        if not queue:
            break
    raise ValueError(f"cannot construct exact {target}-token ChatML prompt")


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_csv(path) if path.exists() else []
    all_rows: list[dict[str, Any]] = [*existing, row]
    fields: list[str] = []
    for item in all_rows:
        for key in item:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    temporary.replace(path)
