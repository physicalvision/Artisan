_PREFERRED_TOOL_ARG_KEYS = (
    "path",
    "content",
    "query",
    "offset",
    "limit",
    "instruction",
    "old_string",
    "new_string",
    "allow_multiple",
    "replace_all",
)


def format_cost(cost: float) -> str:
    if cost <= 0:
        return "$0.0000"
    if cost < 0.0001:
        return "<$0.0001"
    elif cost < 0.01:
        return f"${cost:.6f}"
    elif cost < 1.0:
        return f"${cost:.4f}"
    else:
        return f"${cost:.2f}"


def format_tokens(tokens: int) -> str:
    if tokens < 1000:
        return str(tokens)
    elif tokens < 1_000_000:
        return f"{tokens / 1000:.1f}K"
    else:
        return f"{tokens / 1_000_000:.1f}M"


def _format_percent(ratio: float) -> str:
    percent = ratio * 100.0
    if percent < 10:
        return f"{percent:.1f}%"
    return f"{percent:.0f}%"


def _int_or_none(value) -> int | None:
    return value if isinstance(value, int) else None


def format_context_pressure(pressure: dict | None) -> str:
    if not isinstance(pressure, dict):
        return ""

    prompt_tokens = _int_or_none(pressure.get("prompt_tokens"))
    max_context_tokens = _int_or_none(pressure.get("max_context_tokens"))
    remaining_context_tokens = _int_or_none(pressure.get("remaining_context_tokens"))
    output_tokens = _int_or_none(pressure.get("output_tokens"))
    cached_tokens = _int_or_none(pressure.get("cached_tokens"))

    parts: list[str] = []
    if prompt_tokens is not None and max_context_tokens is not None:
        ratio = pressure.get("pressure_ratio")
        if not isinstance(ratio, (int, float)):
            ratio = prompt_tokens / max_context_tokens if max_context_tokens > 0 else None
        if isinstance(ratio, (int, float)):
            parts.append(
                f"ctx={format_tokens(prompt_tokens)}/{format_tokens(max_context_tokens)} "
                f"({_format_percent(float(ratio))})"
            )
        else:
            parts.append(f"ctx={format_tokens(prompt_tokens)}/{format_tokens(max_context_tokens)}")
        if remaining_context_tokens is not None:
            parts.append(f"left={format_tokens(max(0, remaining_context_tokens))}")
    elif prompt_tokens is not None:
        parts.append(f"ctx={format_tokens(prompt_tokens)}/?")

    if output_tokens is not None:
        parts.append(f"out={format_tokens(output_tokens)}")

    if cached_tokens is not None:
        if isinstance(prompt_tokens, int) and prompt_tokens > 0 and cached_tokens > 0:
            parts.append(
                f"cache={format_tokens(cached_tokens)} "
                f"({_format_percent(cached_tokens / prompt_tokens)})"
            )
        elif cached_tokens > 0:
            parts.append(f"cache={format_tokens(cached_tokens)}")

    return "  ".join(parts)


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"

    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"

    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins}m"


def format_timestamp(seconds: float) -> str:
    if seconds < 3600:
        minutes, secs = divmod(int(seconds), 60)
        return f"{minutes:02d}:{secs:02d}"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}:{minutes:02d}:{secs:02d}"


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def format_tool_args(args, max_items: int | None = None) -> list[str]:
    if isinstance(args, str):
        try:
            import json

            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return [truncate_text(args, 60)] if args else []

    if not isinstance(args, dict):
        return [str(args)] if args else []

    preferred_order = {key: index for index, key in enumerate(_PREFERRED_TOOL_ARG_KEYS)}
    ordered_items = sorted(
        args.items(),
        key=lambda item: (preferred_order.get(item[0], len(preferred_order)), item[0]),
    )

    formatted = []
    for i, (key, value) in enumerate(ordered_items):
        if max_items is not None and i >= max_items:
            formatted.append(f"... ({len(args) - max_items} more)")
            break

        if isinstance(value, str):
            if len(value) > 50:
                value = truncate_text(value, 50)
            formatted.append(f"{key}: {value}")
        elif isinstance(value, (int, float, bool)):
            formatted.append(f"{key}: {value}")
        elif isinstance(value, (list, dict)):
            formatted.append(f"{key}: {type(value).__name__}({len(value)})")
        else:
            formatted.append(f"{key}: {type(value).__name__}")

    return formatted
