from __future__ import annotations

from copy import deepcopy

from ..types import (
    AssistantMessage,
    Context,
    ImageContent,
    TextContent,
    ThinkingContent,
    ToolCallContent,
    ToolResultContent,
    ToolResultMessage,
    UserMessage,
)


def apply_model_capabilities(model, context: Context) -> Context:
    """根据模型能力矩阵裁剪不支持的字段。"""

    supports_images = getattr(model, "supports_images", False)
    messages = [_apply_message_capabilities(message, supports_images=supports_images) for message in context.messages]
    return Context(systemPrompt=context.systemPrompt, messages=messages, tools=deepcopy(context.tools))


def _apply_message_capabilities(message, *, supports_images: bool):
    """按模型能力裁剪单条消息。"""

    if isinstance(message, UserMessage):
        return UserMessage(content=_filter_blocks(message.content, supports_images=supports_images), metadata=dict(message.metadata), timestamp=message.timestamp)
    if isinstance(message, AssistantMessage):
        return AssistantMessage(
            content=_filter_blocks(message.content, supports_images=supports_images),
            metadata=dict(message.metadata),
            usage=deepcopy(message.usage),
            stopReason=message.stopReason,
            responseId=message.responseId,
            errorMessage=message.errorMessage,
            timestamp=message.timestamp,
        )
    if isinstance(message, ToolResultMessage):
        return ToolResultMessage(
            toolCallId=message.toolCallId,
            toolName=message.toolName,
            content=_filter_blocks(message.content, supports_images=supports_images),
            metadata=dict(message.metadata),
            isError=message.isError,
            details=deepcopy(message.details),
            timestamp=message.timestamp,
        )
    return message


def _filter_blocks(blocks, *, supports_images: bool):
    """裁剪内容块列表中的不支持项。"""

    filtered = []
    for block in blocks:
        if isinstance(block, ImageContent) and not supports_images:
            filtered.append(TextContent(text="[image omitted by capability clamp]"))
            continue
        if isinstance(block, (TextContent, ThinkingContent, ToolCallContent, ToolResultContent, ImageContent)):
            filtered.append(block)
            continue
        filtered.append(block)
    return filtered
