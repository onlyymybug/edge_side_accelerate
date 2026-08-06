import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import NA
from .metrics import fmt, to_float_or_none
from .parsers import strip_terminal_control_tokens


_SUPPORTED_SCHEMA_TYPES = {
    "string",
    "integer",
    "number",
    "boolean",
    "object",
    "array",
}
ACTION_VALIDATOR_VERSION = "strict_json_v2"


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


@dataclass(frozen=True)
class ParsedAction:
    value: Any
    action: Any
    arguments: Any
    method: str
    error_type: str
    message: str


@dataclass(frozen=True)
class ActionValidation:
    valid: bool
    envelope_valid: bool
    action_known: bool
    arguments_schema_valid: bool
    error_type: str
    message: str


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    raise ValueError(f"Unsupported schema type: {expected}")

def _normalize_argument(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.strip().split())
    if isinstance(value, dict):
        return {k: _normalize_argument(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_argument(v) for v in value]
    return value


def validate_action_setup(
    action_schema: dict[str, Any],
    samples: list[dict[str, Any]],
) -> None:
    # action_schema 必须是 object；配置本身错误时应在实验开始前失败，不能记成模型的 schema_error。
    if not isinstance(action_schema, dict):
        raise ValueError("'action_schema' must be an object")

    # 每个 action 的参数定义必须是 object，且每个参数类型必须属于 validator 支持的类型集合。
    for action, expected_args in action_schema.items():
        if not isinstance(expected_args, dict):
            raise ValueError(f"Schema for action '{action}' must be an object")
        for name, expected_type in expected_args.items():
            if str(expected_type) not in _SUPPORTED_SCHEMA_TYPES:
                raise ValueError(
                    f"Unsupported schema type for action '{action}', argument '{name}': "
                    f"{expected_type}"
                )

    # 每条 gold action 必须是候选 action；错误标注应在运行前暴露，不能归因到模型输出。
    for sample in samples:
        sample_id = sample.get("sample_id", "unknown")
        gold_action = sample.get("gold_action")
        if not isinstance(gold_action, str) or gold_action not in action_schema:
            raise ValueError(f"Unknown gold_action in sample '{sample_id}': {gold_action}")

        # gold arguments 的参数名必须与 gold action 定义完全一致，否则后续正确性比较没有可信基准。
        gold_arguments = sample.get("gold_arguments")
        if not isinstance(gold_arguments, dict):
            raise ValueError(f"gold_arguments must be an object in sample '{sample_id}'")
        expected_args = action_schema[gold_action]
        if set(gold_arguments) != set(expected_args):
            missing = sorted(set(expected_args) - set(gold_arguments))
            extra = sorted(set(gold_arguments) - set(expected_args))
            raise ValueError(
                f"gold argument keys mismatch in sample '{sample_id}'; "
                f"missing={missing}, extra={extra}"
            )

        # gold arguments 的 JSON 类型也必须符合 schema，避免把数据集标注错误统计成模型 wrong_argument。
        for name, expected_type in expected_args.items():
            if not _type_matches(gold_arguments[name], str(expected_type)):
                raise ValueError(
                    f"gold argument '{name}' in sample '{sample_id}' must have type "
                    f"{expected_type}"
                )


def parse_action_text(text: str) -> ParsedAction:
    # 空输出无法形成 JSON 文档，直接归为 parse_error。
    if not text or not text.strip():
        return ParsedAction(NA, NA, NA, "empty", "parse_error", "empty model output")

    # 对整个模型输出只解析一次；附加解释、代码围栏、多个 JSON、尾随文字和截断 JSON 都必须解析失败。
    # json.loads 允许 JSON 前后的空白，但要求除空白外的全部字符都属于同一个完整 JSON 文档。
    # Python 默认接受 NaN/Infinity 等扩展常量，这里显式拒绝，保证按标准 JSON 衡量模型输出。
    try:
        value = json.loads(text, parse_constant=_reject_non_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        return ParsedAction(
            NA,
            NA,
            NA,
            "strict_json_document",
            "parse_error",
            f"invalid JSON document: {exc}",
        )

    # parse 阶段只负责 JSON 语法和唯一文档，不负责要求顶层一定是 object。
    # 顶层数组、字符串等都是合法 JSON，后续由 validator 归入 schema_error。
    action = value.get("action", NA) if isinstance(value, dict) else NA
    arguments = value.get("arguments", NA) if isinstance(value, dict) else NA
    return ParsedAction(
        value=value,
        action=action,
        arguments=arguments,
        method="strict_json_document",
        error_type="none",
        message="",
    )


def validate_action_schema(
    parsed: ParsedAction, action_schema: dict[str, Any]
) -> ActionValidation:
    # JSON 解析错误直接向上传递；此时还没有合法 JSON，不能继续做 schema 检查。
    if parsed.error_type == "parse_error":
        return ActionValidation(
            valid=False,
            envelope_valid=False,
            action_known=False,
            arguments_schema_valid=False,
            error_type="parse_error",
            message=parsed.message,
        )

    # 合法 JSON 的顶层必须是 object；数组、字符串、数字等不符合 function-call 外壳。
    if not isinstance(parsed.value, dict):
        return ActionValidation(
            valid=False,
            envelope_valid=False,
            action_known=False,
            arguments_schema_valid=False,
            error_type="schema_error",
            message="top-level JSON must be an object",
        )

    # function-call 顶层必须恰好包含 action 和 arguments；缺字段或附加字段都属于外壳 schema 错误。
    required_top = {"action", "arguments"}
    actual_top = set(parsed.value)
    if actual_top != required_top:
        missing = sorted(required_top - actual_top)
        extra = sorted(actual_top - required_top)
        return ActionValidation(
            valid=False,
            envelope_valid=False,
            action_known=False,
            arguments_schema_valid=False,
            error_type="schema_error",
            message=f"top-level keys mismatch; missing={missing}, extra={extra}",
        )

    # action 字段必须是字符串，非字符串无法作为动作标识符使用，属于外壳 schema 错误。
    action = parsed.action
    if not isinstance(action, str):
        return ActionValidation(
            valid=False,
            envelope_valid=False,
            action_known=False,
            arguments_schema_valid=False,
            error_type="schema_error",
            message="'action' must be a string",
        )

    # arguments 字段必须是 object；其他 JSON 类型不能表达命名参数，属于外壳 schema 错误。
    arguments = parsed.arguments
    if not isinstance(arguments, dict):
        return ActionValidation(
            valid=False,
            envelope_valid=False,
            action_known=False,
            arguments_schema_valid=False,
            error_type="schema_error",
            message="'arguments' must be an object",
        )

    # action 不在候选 schema 中时，外壳仍然合法，但该调用不可执行；主错误稍后归为 wrong_action_type。
    if action not in action_schema:
        return ActionValidation(
            valid=False,
            envelope_valid=True,
            action_known=False,
            arguments_schema_valid=False,
            error_type="none",
            message=f"unknown action: {action}",
        )

    # action_schema 自身格式错误属于实验配置错误，不应归因到模型输出，因此直接抛出异常终止实验。
    expected_args = action_schema[action]
    if not isinstance(expected_args, dict):
        raise ValueError(f"Schema for action '{action}' must be an object")

    # 参数名必须与预测 action 的定义完全一致；缺失或多余参数稍后统一归为 wrong_argument。
    expected_keys = set(expected_args)
    actual_keys = set(arguments)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        return ActionValidation(
            valid=False,
            envelope_valid=True,
            action_known=True,
            arguments_schema_valid=False,
            error_type="none",
            message=f"argument keys mismatch for {action}; missing={missing}, extra={extra}",
        )

    # 每个参数的 JSON 类型必须与预测 action 的定义一致；类型错误稍后统一归为 wrong_argument。
    for name, expected_type in expected_args.items():
        if not _type_matches(arguments[name], str(expected_type)):
            return ActionValidation(
                valid=False,
                envelope_valid=True,
                action_known=True,
                arguments_schema_valid=False,
                error_type="none",
                message=f"argument '{name}' must have type {expected_type}",
            )

    # action 已知且参数名称、数量、类型都符合定义，因此该调用是 schema-valid action。
    return ActionValidation(
        valid=True,
        envelope_valid=True,
        action_known=True,
        arguments_schema_valid=True,
        error_type="none",
        message="",
    )


def classify_action(
    parsed: ParsedAction,
    validation: ActionValidation,
    gold_action: str,
    gold_arguments: dict[str, Any],
) -> tuple[bool, bool, str, str]:
    # JSON 解析失败或通用 function-call 外壳不合法时，保留 parser/validator 给出的主错误。
    if not validation.envelope_valid:
        return False, False, validation.error_type, validation.message

    # 只要 action 未知或与 gold action 不同，主错误就是 wrong_action_type。
    # 即使错误 action 的参数也不合法，也不再让参数 schema 错误掩盖动作选择错误。
    if parsed.action != gold_action:
        message = f"expected action={gold_action}, got={parsed.action}"
        if not validation.action_known:
            message += "; " + validation.message
        return validation.valid, False, "wrong_action_type", message

    # action 正确后再检查参数名称、数量和类型；这些问题统一归为 wrong_argument。
    if not validation.arguments_schema_valid:
        return False, False, "wrong_argument", validation.message

    # 参数 schema 合法后再比较参数值；字符串只做递归空白归一化，其他值保持 JSON 语义。
    if _normalize_argument(parsed.arguments) != _normalize_argument(gold_arguments):
        return (
            validation.valid,
            False,
            "wrong_argument",
            "expected arguments="
            + json.dumps(gold_arguments, ensure_ascii=False, sort_keys=True)
            + ", got="
            + json.dumps(parsed.arguments, ensure_ascii=False, sort_keys=True),
        )
    return validation.valid, True, "none", ""


def find_time_to_valid_action_ms(
    *,
    token_timestamps_path: Path,
    action_schema: dict[str, Any],
    final_action_valid: bool,
    fallback_e2e_ms: Any,
) -> tuple[Any, str]:
    # 最终输出不是 valid action 时，中途短暂出现过的合法 JSON 不能计为 time_to_valid_action。
    if not final_action_valid:
        return NA, "unavailable"

    if token_timestamps_path.exists():
        prefix = ""
        first_timestamp_ms: float | None = None
        with token_timestamps_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                timestamp_ms = to_float_or_none(row.get("timestamp_ms"))
                if first_timestamp_ms is None and timestamp_ms is not None:
                    first_timestamp_ms = timestamp_ms

                if bool(row.get("prompt_echo", False)):
                    continue
                prefix += str(row.get("piece", ""))

                parsed = parse_action_text(strip_terminal_control_tokens(prefix))
                validation = validate_action_schema(parsed, action_schema)
                if not validation.valid:
                    continue

                delta_ms = to_float_or_none(row.get("delta_ms"))
                if delta_ms is not None:
                    return fmt(delta_ms), "token_timestamps_delta_ms"
                if timestamp_ms is not None and first_timestamp_ms is not None:
                    return fmt(timestamp_ms - first_timestamp_ms), "token_timestamps_timestamp_ms"

    e2e = to_float_or_none(fallback_e2e_ms)
    if e2e is not None:
        return fmt(e2e), "e2e_latency_fallback"

    return NA, "unavailable"
