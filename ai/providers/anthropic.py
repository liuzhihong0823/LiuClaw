from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from ai.errors import AuthenticationError, ProviderResponseError
from ai.options import Options
from ai.providers.base import Provider
from ai.types import AssistantMessage, StreamEvent, ToolCall
from ai.utils.streaming import EventBuilder, create_done_event


class AnthropicProvider(Provider):
    """Anthropic provider 的统一适配实现。"""

    name = "anthropic"

    def supports(self, model: Any) -> bool:
        """判断模型是否应由 Anthropic provider 处理。"""

        model_id = self._model_id(model)
        provider = getattr(model, "provider", None)
        if provider:
            return provider == self.name
        return model_id.startswith("anthropic:") or model_id.startswith("claude")

    def _model_id(self, model: Any) -> str:
        """读取统一模型 ID。"""

        return getattr(model, "id", model)

    def _runtime_model_name(self, model: Any) -> str:
        """把统一模型标识转换为 Anthropic SDK 使用的模型名。"""

        model_id = self._model_id(model)
        return model_id.split(":", 1)[1] if model_id.startswith("anthropic:") else model_id

    def _require_api_key(self) -> str:
        """确保 Anthropic API Key 已配置。"""

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise AuthenticationError("ANTHROPIC_API_KEY is not set")
        return api_key

    def _client_kwargs(self, options: Options) -> dict[str, Any]:
        """构造 Anthropic SDK 客户端初始化参数。"""

        return {
            "api_key": self._require_api_key(),
            "timeout": options.timeout,
            "base_url": os.getenv("ANTHROPIC_BASE_URL"),
        }

    def _context_tools(self, context: Any) -> list[Any]:
        """读取 `Context.tools`。"""

        return list(getattr(context, "tools", []))

    def _context_messages(self, context: Any) -> list[Any]:
        """读取 `Context.messages`。"""

        return list(getattr(context, "messages", []))

    def _context_system_prompt(self, context: Any) -> str | None:
        """读取 `Context.systemPrompt`。"""

        return getattr(context, "systemPrompt", None)

    def _tool_value(self, tool: Any, key: str, default: Any = None) -> Any:
        """兼容读取工具对象或工具字典中的字段。"""

        if isinstance(tool, dict):
            return tool.get(key, default)
        return getattr(tool, key, default)

    def _build_tools(self, context: Any) -> list[dict[str, Any]] | None:
        """把统一工具定义映射为 Anthropic 请求格式。"""

        tools = self._context_tools(context)
        if not tools:
            return None
        return [
            {
                "name": self._tool_value(tool, "name"),
                "description": self._tool_value(tool, "description", "") or "",
                "input_schema": self._tool_value(tool, "inputSchema", {}),
            }
            for tool in tools
        ]

    def _message_to_input(self, message: Any) -> dict[str, Any]:
        """把统一消息类型映射为 Anthropic `messages` 项。"""

        if hasattr(message, "toolCallId"):
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": getattr(message, "toolCallId"),
                        "content": getattr(message, "content", ""),
                    }
                ],
            }

        tool_calls = getattr(message, "toolCalls", [])
        if tool_calls:
            content: list[dict[str, Any]] = []
            text = getattr(message, "content", "")
            if text:
                content.append({"type": "text", "text": text})
            for tool_call in tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "input": tool_call.arguments,
                    }
                )
            return {"role": "assistant", "content": content}

        return {"role": getattr(message, "role", "user"), "content": getattr(message, "content", "")}

    def _provider_reasoning(self, options: Options) -> dict[str, Any]:
        """读取预先映射好的 provider reasoning 配置。"""

        mapped = options.metadata.get("_providerReasoning")
        return mapped if isinstance(mapped, dict) else {}

    def _build_request(self, model: Any, context: Any, options: Options) -> dict[str, Any]:
        """构造 Anthropic Messages API 的请求参数。"""

        request: dict[str, Any] = {
            "model": self._runtime_model_name(model),
            "messages": [self._message_to_input(message) for message in self._context_messages(context)],
            "max_tokens": options.maxTokens or getattr(model, "maxOutputTokens", 1024) or 1024,
        }
        system_prompt = self._context_system_prompt(context)
        if system_prompt:
            request["system"] = system_prompt
        if options.temperature is not None:
            request["temperature"] = options.temperature
        tools = self._build_tools(context)
        if tools is not None:
            request["tools"] = tools
        request.update(self._provider_reasoning(options))
        return request

    async def stream(
        self,
        model: Any,
        context: Any,
        options: Options,
    ) -> AsyncIterator[StreamEvent]:
        """把 Anthropic 流式响应转换为统一事件流。"""

        request = self._build_request(model, context, options)
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            builder = EventBuilder(model=model, provider=self.name)
            yield builder.build_error(
                "anthropic package is not installed",
                metadata={"source": "provider", "provider": self.name, "exception_type": type(exc).__name__},
            )
            return

        client = AsyncAnthropic(**self._client_kwargs(options))
        builder = EventBuilder(model=model, provider=self.name)
        final_message = AssistantMessage(content="", thinking="", toolCalls=[], metadata={})
        tool_buffers: dict[str, str] = {}
        tool_names: dict[str, str] = {}
        tool_index_to_id: dict[str, str] = {}
        text_started = False
        thinking_started = False

        try:
            async with client.messages.stream(**request) as stream:
                yield builder.build("start", lifecycle="start", itemType="message")
                async for event in stream:
                    event_type = getattr(event, "type", None)
                    raw_event = event if options.includeRawProviderEvents else None
                    if event_type == "content_block_start":
                        block = getattr(event, "content_block", None)
                        block_type = getattr(block, "type", None)
                        if block_type == "tool_use":
                            tool_call_id = getattr(block, "id", str(getattr(event, "index", "0")))
                            tool_name = getattr(block, "name", "")
                            tool_names[tool_call_id] = tool_name
                            tool_index_to_id[str(getattr(event, "index", "0"))] = tool_call_id
                            tool_buffers.setdefault(tool_call_id, "")
                            yield builder.build("toolcall_start", toolCallId=tool_call_id, toolName=tool_name, rawEvent=raw_event)
                        elif block_type == "thinking":
                            thinking_started = True
                            yield builder.build("thinking_start", rawEvent=raw_event)
                        elif block_type == "text":
                            text_started = True
                            yield builder.build("text_start", rawEvent=raw_event)
                    elif event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        delta_type = getattr(delta, "type", None)
                        if delta_type == "text_delta":
                            text = getattr(delta, "text", "")
                            if not text_started:
                                text_started = True
                                yield builder.build("text_start", rawEvent=raw_event)
                            final_message.content += text
                            yield builder.build("text_delta", text=text, rawEvent=raw_event)
                        elif delta_type == "thinking_delta":
                            thinking = getattr(delta, "thinking", getattr(delta, "text", ""))
                            if not thinking_started:
                                thinking_started = True
                                yield builder.build("thinking_start", rawEvent=raw_event)
                            final_message.thinking += thinking
                            yield builder.build("thinking_delta", thinking=thinking, rawEvent=raw_event)
                        elif delta_type == "input_json_delta":
                            tool_call_id = tool_index_to_id.get(str(getattr(event, "index", "0")), str(getattr(event, "index", "0")))
                            partial = getattr(delta, "partial_json", "")
                            tool_buffers.setdefault(tool_call_id, "")
                            tool_buffers[tool_call_id] += partial
                            yield builder.build("toolcall_delta", toolCallId=tool_call_id, toolName=tool_names.get(tool_call_id), argumentsDelta=partial, rawEvent=raw_event)
                    elif event_type == "content_block_stop":
                        block_index = str(getattr(event, "index", "0"))
                        tool_call_id = tool_index_to_id.get(block_index)
                        if tool_call_id is not None:
                            arguments = tool_buffers.get(tool_call_id, "")
                            final_message.toolCalls.append(ToolCall(id=tool_call_id, name=tool_names.get(tool_call_id, ""), arguments=arguments))
                            yield builder.build("toolcall_end", toolCallId=tool_call_id, toolName=tool_names.get(tool_call_id), arguments=arguments, rawEvent=raw_event)
                        elif thinking_started:
                            yield builder.build("thinking_end", rawEvent=raw_event)
                            thinking_started = False
                        elif text_started:
                            yield builder.build("text_end", rawEvent=raw_event)
                            text_started = False

                if text_started:
                    yield builder.build("text_end")
                if thinking_started:
                    yield builder.build("thinking_end")
                final_response = await stream.get_final_message()
                final_message.metadata["message"] = final_response
                yield create_done_event(
                    final_message,
                    model=model,
                    provider=self.name,
                    usage=getattr(final_response, "usage", None),
                    stop_reason=getattr(final_response, "stop_reason", None),
                    response_id=getattr(final_response, "id", None),
                    provider_metadata={"stop_type": getattr(final_response, "stop_type", None)},
                )
        except AuthenticationError as exc:
            yield builder.build_error(str(exc), metadata={"source": "provider", "provider": self.name})
        except Exception as exc:  # pragma: no cover
            error = exc if isinstance(exc, ProviderResponseError) else ProviderResponseError(f"Anthropic streaming failed: {exc}")
            yield builder.build_error(str(error), metadata={"source": "provider", "provider": self.name, "exception_type": type(exc).__name__})
