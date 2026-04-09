from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .types import ReasoningLevel

ContextOverflowStrategy = Literal["reject", "truncate_oldest"]


@dataclass(slots=True)
class ReasoningConfig:
    """定义统一 reasoning 配置。"""

    effort: ReasoningLevel  # 推理强度等级。


@dataclass(slots=True)
class Options:
    """定义完整调用接口使用的通用配置。"""

    reasoning: ReasoningLevel | ReasoningConfig | None = None  # 本次请求的推理等级。
    temperature: float | None = None  # 采样温度。
    maxTokens: int | None = None  # 期望输出 token 上限。
    metadata: dict[str, Any] = field(default_factory=dict)  # 透传的调用元信息。
    timeout: float | None = None  # 调用超时时间，单位秒。
    includeRawProviderEvents: bool = False  # 是否在流式结果中保留原始 provider 事件。
    streamQueueMaxSize: int = 64  # 流式事件队列最大容量。
    streamPutTimeout: float | None = None  # 向事件队列写入时的超时设置。
    contextOverflowStrategy: ContextOverflowStrategy = "reject"  # 上下文超窗时的处理策略。
    debug: dict[str, Any] = field(default_factory=dict)  # 调试开关与调试数据。
    provider: dict[str, Any] = field(default_factory=dict)  # provider 专属附加参数。


@dataclass(slots=True)
class SimpleOptions:
    """定义简化调用接口使用的通用配置。"""

    reasoning: ReasoningLevel | ReasoningConfig | None = None  # 简化接口的推理等级。
    temperature: float | None = None  # 简化接口的采样温度。
    maxTokens: int | None = None  # 简化接口的输出 token 上限。
    metadata: dict[str, Any] = field(default_factory=dict)  # 简化接口的透传元信息。
    timeout: float | None = None  # 简化接口的超时时间。
    streamQueueMaxSize: int = 64  # 简化接口流式队列容量。
    streamPutTimeout: float | None = None  # 简化接口流式写队列超时。



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
