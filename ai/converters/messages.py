from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..types import AssistantMessage, Context, ToolResultMessage, UserMessage, ensure_context
from .capabilities import apply_model_capabilities
from .thinking import convert_thinking_for_provider
from .tools import convert_tools_for_provider

_KNOWN_PROVIDERS = {"openai", "anthropic", "zhipu"}


def convert_context_for_provider(model: Any, context: Context | dict[str, Any]) -> Context:
    """将统一 `Context` 转换为目标 provider 可消费的兼容上下文。"""

    normalized = ensure_context(context)
    normalized = apply_model_capabilities(model, normalized)
    normalized = convert_thinking_for_provider(model, normalized)
    provider = getattr(model, "provider", None)
    if provider not in _KNOWN_PROVIDERS:
        return normalized
    return Context(
        systemPrompt=normalized.systemPrompt,
        messages=convert_messages_for_provider(normalized.messages, target_provider=provider),
        tools=convert_tools_for_provider(normalized.tools, target_provider=provider),
    )



def convert_messages_for_provider(messages: list[Any], target_provider: str | None) -> list[Any]:
    """将历史消息转换为目标 provider 兼容的统一消息对象。"""

    return [_convert_message(target_provider, message) for message in messages]



def _convert_message(provider: str | None, message: Any) -> Any:
    """把单条统一消息转换为目标 provider 兼容的表示。"""

    if isinstance(message, UserMessage):
        return UserMessage(content=deepcopy(message.content), metadata=dict(message.metadata), timestamp=message.timestamp)
    if isinstance(message, ToolResultMessage):
        metadata = dict(message.metadata)
        metadata["targetProvider"] = provider
        return ToolResultMessage(
            toolCallId=message.toolCallId,
            toolName=message.toolName,
            content=deepcopy(message.content),
            metadata=metadata,
            isError=message.isError,
            details=deepcopy(message.details),
            timestamp=message.timestamp,
        )
    if isinstance(message, AssistantMessage):
        return _convert_assistant_message(provider, message)
    return message



def _convert_assistant_message(provider: str | None, message: AssistantMessage) -> AssistantMessage:
    """把 assistant 历史消息转换为目标 provider 兼容的统一表示。"""

    metadata = dict(message.metadata)
    metadata["targetProvider"] = provider
    if provider in _KNOWN_PROVIDERS and message.thinking:
        metadata.setdefault("historicalThinking", message.thinking)

    return AssistantMessage(
        content=deepcopy(message.content),
        metadata=metadata,
        usage=deepcopy(message.usage),
        stopReason=message.stopReason,
        responseId=message.responseId,
        errorMessage=message.errorMessage,
        timestamp=message.timestamp,
    )
