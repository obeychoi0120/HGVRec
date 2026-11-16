from __future__ import annotations

import math

from typing import Any


QWEN_DEFAULTS = {
    "gpu_memory_utilization": 0.90,
    "max_num_seqs": 32,
    "max_num_batched_tokens": 16384,
    "max_model_len": "auto",
    "renderer_num_workers": 4,
    "enable_chunked_prefill": True,
    "enable_prefix_caching": True,
    "enforce_eager": False,
    "async_scheduling": None,
}


def qwen_settings(value: dict[str, Any] | None = None) -> dict[str, Any]:
    """vLLM engine settings: config values over defaults; vLLM validates them."""
    return {**QWEN_DEFAULTS, **(value or {})}


def penalty_schedule(value):
    values = value if isinstance(value, list) else [value]
    if not values or any(
        type(x) not in (int, float) or not math.isfinite(x) or not 1 <= x <= 2 for x in values
    ):
        raise ValueError("repetition penalty must be a number or non-empty list in [1, 2]")
    return [float(x) for x in values]


# Local Qwen3-VL Instruct checkpoints, one key per parameter size.
QWEN_MODEL_SIZES = {"qwen3vl_2b": "2B", "qwen3vl_4b": "4B", "qwen3vl_8b": "8B"}
QWEN_MODELS = tuple(QWEN_MODEL_SIZES)


def validate_presence_penalty(value: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not -2 <= value <= 2:
        raise ValueError("presence penalty must be a finite number in [-2, 2]")
    return float(value)


def generation_settings(
    model: str, *, representation: str,
    repetition_penalty: float | list[float] | None = None,
    presence_penalty: float | None = None,
    stop_repeated_lines: bool = False,
) -> dict[str, Any]:
    """Greedy generation with optional presence and generated-token repetition penalties.

    ``stop_repeated_lines`` ends a Graph output instead of repeating an earlier line. It is
    recorded only when enabled, so existing generation identities are unchanged.
    """
    if model not in QWEN_MODELS:
        raise ValueError(f"unsupported Qwen model: {model}")
    if representation not in ("graph", "description"):
        raise ValueError(f"unsupported representation: {representation}")
    if stop_repeated_lines and representation != "graph":
        raise ValueError("stopping before a repeated line is supported only for Graph scenes")
    penalty = 1.05 if representation == "graph" else 1.0
    if repetition_penalty is not None:
        # A schedule starts at its first value; later passes are set per request.
        penalty = penalty_schedule(repetition_penalty)[0]
    return {
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "presence_penalty": (
            validate_presence_penalty(presence_penalty)
            # Graph roles copy entity names and repeat delimiters on every line,
            # so reuse of generated tokens is not penalised by default.
            if presence_penalty is not None else 0.0 if representation == "graph" else 1.5
        ),
        "repetition_penalty": penalty,
        "model": model,
        **({"stop_repeated_lines": True} if stop_repeated_lines else {}),
    }
