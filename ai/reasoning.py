from __future__ import annotations

from typing import Any

from .errors import UnsupportedFeatureError
from .options import normalize_reasoning


def build_reasoning_config(model: Any, reasoning: Any) -> dict[str, Any]:
    """把统一 reasoning 等级映射为目标 provider 的专用参数。"""

    level = normalize_reasoning(reasoning)
    if level is None:
        return {}

    provider = getattr(model, "provider", None)
    if provider == "openai":
        return {"reasoning": {"effort": level}}
    if provider == "anthropic":
        if level in {"off", "minimal"}:
            return {}
        budget_map = {"low": 1024, "medium": 4096, "high": 8192, "xhigh": 16384}
        return {"thinking": {"type": "enabled", "budget_tokens": budget_map[level]}}
    if provider == "zhipu":
        runtime_model = getattr(model, "id", "")
        runtime_name = runtime_model.split(":", 1)[1] if ":" in runtime_model else runtime_model
        if level in {"off", "minimal", "low"}:
            return {"thinking": {"type": "disabled"}}
        if level == "medium":
            return {"thinking": {"type": "enabled"}}
        if runtime_name == "glm-4.6":
            return {"thinking": {"type": "enabled"}}
        return {"thinking": {"type": "enabled"}, "clear_thinking": False}
    raise UnsupportedFeatureError(
        f"Provider '{provider or 'unknown'}' does not support reasoning mapping"
    )



def resolve_reasoning_config(model: Any, reasoning: Any) -> dict[str, Any]:
    """兼容别名，返回目标 provider 的 reasoning 配置。"""

    return build_reasoning_config(model, reasoning)



def merge_reasoning_metadata(metadata: dict[str, Any] | None, model: Any, reasoning: Any) -> dict[str, Any]:
    """把 provider 专用 reasoning 映射写入 metadata，供后续运行时读取。"""

    merged = dict(metadata or {})
    try:
        mapped = build_reasoning_config(model, reasoning)
    except UnsupportedFeatureError:
        return merged
    if mapped:
        merged["_providerReasoning"] = mapped
    return merged
