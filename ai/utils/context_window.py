from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai.options import Options
from ai.types import (
    AssistantMessage,
    Context,
    ImageContent,
    Model,
    TextContent,
    ThinkingContent,
    ToolCallContent,
    ToolResultContent,
    ToolResultMessage,
    UserMessage,
)


class ContextOverflowError(ValueError):
    """表示上下文估算后已超出模型窗口限制。"""


@dataclass(slots=True)
class ContextOverflowReport:
    """描述一次上下文窗口检测的结果。"""

    estimated_tokens: int
    requested_output_tokens: int
    total_tokens: int
    limit: int

    @property
    def is_overflow(self) -> bool:
        """返回当前预算是否已经超出窗口上限。"""

        return self.total_tokens > self.limit



def estimate_context_tokens(context: Context) -> int:
    """粗略估算统一 `Context` 占用的 token 数。"""

    total = 0
    if context.systemPrompt:
        total += _estimate_text_tokens(context.systemPrompt)

    for message in context.messages:
        total += _estimate_message_tokens(message)

    for tool in context.tools:
        total += _estimate_text_tokens(_tool_value(tool, "name", ""))
        total += _estimate_text_tokens(_tool_value(tool, "description", "") or "")
        total += _estimate_text_tokens(str(_tool_value(tool, "inputSchema", {})))

    return total



def detect_context_overflow(model: Model, context: Context, options: Options | None = None) -> ContextOverflowReport:
    """检测上下文是否会超过模型窗口，并返回检测结果。"""

    estimated_tokens = estimate_context_tokens(context)
    requested_output_tokens = 0
    if options is not None and options.maxTokens is not None:
        requested_output_tokens = options.maxTokens
    elif model.maxOutputTokens:
        requested_output_tokens = model.maxOutputTokens

    total_budget = estimated_tokens + requested_output_tokens
    return ContextOverflowReport(
        estimated_tokens=estimated_tokens,
        requested_output_tokens=requested_output_tokens,
        total_tokens=total_budget,
        limit=model.contextWindow,
    )



def ensure_context_fits_window(model: Model, context: Context, options: Options | None = None) -> ContextOverflowReport:
    """在超出模型窗口时抛错，否则返回检测结果。"""

    report = detect_context_overflow(model, context, options)
    if report.is_overflow:
        raise ContextOverflowError(
            f"context exceeds model window: estimated={report.estimated_tokens}, "
            f"requested_output={report.requested_output_tokens}, "
            f"limit={report.limit}"
        )
    return report


def truncate_context_to_window(model: Model, context: Context, options: Options | None = None) -> Context:
    """按最旧优先裁剪消息，直到预算回到窗口以内。"""

    messages = list(context.messages)
    while messages:
        candidate = Context(systemPrompt=context.systemPrompt, messages=messages, tools=context.tools)
        if not detect_context_overflow(model, candidate, options).is_overflow:
            return candidate
        messages.pop(0)
    candidate = Context(systemPrompt=context.systemPrompt, messages=[], tools=context.tools)
    if detect_context_overflow(model, candidate, options).is_overflow:
        raise ContextOverflowError("context exceeds model window even after truncation")
    return candidate



def _estimate_message_tokens(message: UserMessage | AssistantMessage | ToolResultMessage) -> int:
    """粗略估算一条消息占用的 token 数。"""

    total = 0
    for block in message.content:
        total += _estimate_content_block_tokens(block)
    if isinstance(message, ToolResultMessage):
        total += _estimate_text_tokens(message.toolCallId)
        total += _estimate_text_tokens(message.toolName)
    return total



def _estimate_text_tokens(text: str) -> int:
    """用保守近似方法估算文本 token 数。"""

    if not text:
        return 0
    return max(1, (len(text) + 2) // 3)


def _estimate_content_block_tokens(block: object) -> int:
    """估算单个内容块占用的 token 数。"""

    if isinstance(block, TextContent):
        return _estimate_text_tokens(block.text)
    if isinstance(block, ThinkingContent):
        return _estimate_text_tokens(block.thinking)
    if isinstance(block, ToolCallContent):
        return (
            _estimate_text_tokens(block.id)
            + _estimate_text_tokens(block.name)
            + _estimate_text_tokens(block.arguments_text)
        )
    if isinstance(block, ToolResultContent):
        return _estimate_text_tokens(block.text) + _estimate_text_tokens(str(block.metadata))
    if isinstance(block, ImageContent):
        return _estimate_text_tokens(block.mimeType) + _estimate_text_tokens(str(block.metadata))
    return _estimate_text_tokens(str(block))



def _tool_value(tool: Any, key: str, default: Any) -> Any:
    """兼容读取工具对象或工具字典中的字段。"""

    if isinstance(tool, dict):
        return tool.get(key, default)
    return getattr(tool, key, default)
