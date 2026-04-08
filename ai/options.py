from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .types import ReasoningLevel

ContextOverflowStrategy = Literal["reject", "truncate_oldest"]


@dataclass(slots=True)
class ReasoningConfig:
    """定义统一 reasoning 配置。"""

    effort: ReasoningLevel


@dataclass(slots=True)
class Options:
    """定义完整调用接口使用的通用配置。"""

    reasoning: ReasoningLevel | ReasoningConfig | None = None
    temperature: float | None = None
    maxTokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timeout: float | None = None
    includeRawProviderEvents: bool = False
    streamQueueMaxSize: int = 64
    streamPutTimeout: float | None = None
    contextOverflowStrategy: ContextOverflowStrategy = "reject"
    debug: dict[str, Any] = field(default_factory=dict)
    provider: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SimpleOptions:
    """定义简化调用接口使用的通用配置。"""

    reasoning: ReasoningLevel | ReasoningConfig | None = None
    temperature: float | None = None
    maxTokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timeout: float | None = None
    streamQueueMaxSize: int = 64
    streamPutTimeout: float | None = None



def normalize_reasoning(
    reasoning: ReasoningLevel | ReasoningConfig | None,
) -> ReasoningLevel | None:
    """将 reasoning 配置归一化为统一的等级字符串。"""

    if reasoning is None:
        return None
    if isinstance(reasoning, ReasoningConfig):
        return reasoning.effort
    return reasoning



def ensure_options(options: Options | None) -> Options:
    """确保上层总能获得一个可用的 `Options` 实例。"""

    return options if options is not None else Options()
