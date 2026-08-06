import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any
from .constants import NA


_INTEGER_TOKEN = r"(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)"
_NUMBER_PATTERN = (
    r"[-+]?[$]?(?:"
    + _INTEGER_TOKEN
    + r"\s*/\s*[-+]?"
    + _INTEGER_TOKEN
    + r"|"
    + _INTEGER_TOKEN
    + r"(?:\.[0-9]+)?|\.[0-9]+)[$]?"
)
_FINAL_ANSWER_PATTERN = re.compile(
    r"The\s+answer\s+is\s+("
    + _NUMBER_PATTERN
    + r")\s*\.\s*(?=<\|im_end\|>|<\|endoftext\|>|$)",
    re.IGNORECASE,
)
_GSM8K_GOLD_PATTERN = re.compile(
    r"####\s*(" + _NUMBER_PATTERN + r")\s*$",
    re.IGNORECASE,
)
_PLAIN_NUMBER_PATTERN = re.compile(
    r"^[+-]?(?:[0-9]+/[+-]?[0-9]+|[0-9]+(?:\.[0-9]+)?|\.[0-9]+)$"
)


@dataclass(frozen=True)
class ParsedAnswer:
    answer: Any
    method: str


def _canonical_decimal(x: Decimal) -> str:
    if x == 0:
        return "0"
    if x == x.to_integral_value():
        return str(x.quantize(Decimal(1)))
    return format(x.normalize(), "f").rstrip("0").rstrip(".")


def _normalize_numeric_answer(value: Any) -> Any:
    if value is None or value == NA:
        return NA

    s = str(value).strip().replace("$", "").replace(",", "")
    s = re.sub(r"\s+", "", s)

    if not _PLAIN_NUMBER_PATTERN.fullmatch(s):
        return NA

    try:
        if "/" in s:
            frac = Fraction(s)
            dec = Decimal(frac.numerator) / Decimal(frac.denominator)
        else:
            dec = Decimal(s)
        return _canonical_decimal(dec)
    except (InvalidOperation, ZeroDivisionError, ValueError):
        return NA


def parse_final_answer(text: str) -> ParsedAnswer:
    """Parse the last exact ``The answer is <number>.`` marker."""
    if not text or not text.strip():
        return ParsedAnswer(NA, "empty")

    matches = _FINAL_ANSWER_PATTERN.findall(text)
    if matches:
        return ParsedAnswer(
            _normalize_numeric_answer(matches[-1]),
            "exact_answer_phrase",
        )

    return ParsedAnswer(NA, "missing_answer_marker")


def parse_gold_answer(value: Any) -> ParsedAnswer:
    """Normalize a dataset gold value without requiring the model output marker."""
    if value is None or value == NA or not str(value).strip():
        return ParsedAnswer(NA, "gold_empty")

    text = str(value).strip()
    gsm8k_matches = _GSM8K_GOLD_PATTERN.findall(text)
    if gsm8k_matches:
        answer = _normalize_numeric_answer(gsm8k_matches[-1])
        method = "gold_gsm8k_hash" if answer != NA else "gold_invalid"
        return ParsedAnswer(answer, method)

    answer = _normalize_numeric_answer(text)
    method = "gold_value" if answer != NA else "gold_invalid"
    return ParsedAnswer(answer, method)


def answers_equal(pred: Any, gold: Any) -> bool:
    if pred == NA or gold == NA:
        return False
    pred_norm = _normalize_numeric_answer(pred)
    gold_norm = _normalize_numeric_answer(gold)
    if pred_norm == NA or gold_norm == NA:
        return False
    if pred_norm == gold_norm:
        return True

    try:
        return abs(Decimal(str(pred_norm)) - Decimal(str(gold_norm))) <= Decimal("1e-6")
    except Exception:
        return str(pred_norm).strip().lower() == str(gold_norm).strip().lower()
