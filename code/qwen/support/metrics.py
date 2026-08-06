from datetime import datetime
from typing import Any, Iterable

from .constants import NA


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def to_float_or_none(x: Any) -> float | None:
    try:
        if x is None or x == NA or x == "":
            return None
        return float(x)
    except Exception:
        return None


def fmt(x: Any, digits: int = 3) -> Any:
    if x is None or x == NA:
        return NA
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, float):
        return round(x, digits)
    return x


def percentile(values: Iterable[float], q: float) -> Any:
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return NA
    if len(xs) == 1:
        return round(xs[0], 3)
    pos = (len(xs) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return round(xs[lo] + frac * (xs[hi] - xs[lo]), 3)


def mean(values: Iterable[Any]) -> Any:
    vals = [to_float_or_none(v) for v in values]
    vals = [v for v in vals if v is not None]
    return fmt(sum(vals) / len(vals)) if vals else NA


def bool_rate(values: Iterable[Any]) -> Any:
    vals: list[bool] = []
    for v in values:
        if isinstance(v, bool):
            vals.append(v)
        elif isinstance(v, str):
            if v.lower() == "true":
                vals.append(True)
            elif v.lower() == "false":
                vals.append(False)
    return fmt(sum(vals) / len(vals), 6) if vals else NA
