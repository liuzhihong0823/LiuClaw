from __future__ import annotations

from copy import deepcopy

from ..types import AssistantMessage, Context, ToolResultMessage, UserMessage


def convert_thinking_for_provider(model, context: Context) -> Context:
    """处理 thinking 相关的 provider 兼容转换。"""

    provider = getattr(model, "provider", None)
    if provider is None:
        return context

    messages = []
    for message in context.messages:
        if isinstance(message, AssistantMessage):
            metadata = dict(message.metadata)
            if message.thinking:
                metadata.setdefault("historicalThinking", message.thinking)
            messages.append(
                AssistantMessage(
                    content=deepcopy(message.content),
                    metadata=metadata,
                    usage=deepcopy(message.usage),
                    stopReason=message.stopReason,
                    responseId=message.responseId,
                    errorMessage=message.errorMessage,
                    timestamp=message.timestamp,
                )
            )
        elif isinstance(message, UserMessage):
            messages.append(UserMessage(content=deepcopy(message.content), metadata=dict(message.metadata), timestamp=message.timestamp))
        elif isinstance(message, ToolResultMessage):
            messages.append(
                ToolResultMessage(
                    toolCallId=message.toolCallId,
                    toolName=message.toolName,
                    content=deepcopy(message.content),
                    metadata=dict(message.metadata),
                    isError=message.isError,
                    details=deepcopy(message.details),
                    timestamp=message.timestamp,
                )
            )
        else:
            messages.append(message)
    return Context(systemPrompt=context.systemPrompt, messages=messages, tools=deepcopy(context.tools))
