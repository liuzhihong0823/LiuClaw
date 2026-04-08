from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from ai.config import ProviderConfig
from ai.errors import AuthenticationError, ProviderResponseError
from ai.options import Options
from ai.providers.base import Provider
from ai.types import AssistantMessage, StreamEvent, TextContent, ThinkingContent, ToolCallContent, parse_tool_arguments
from ai.utils.streaming import EventBuilder, create_done_event


class OpenAIProvider(Provider):
    """OpenAI provider 的统一适配实现。"""

    name = "openai"

    def supports(self, model: Any) -> bool:
        """判断模型是否应由 OpenAI provider 处理。"""

        model_id = self._model_id(model)
        provider = getattr(model, "provider", None)
        if provider:
            return provider == self.name
        return model_id.startswith("openai:") or model_id.startswith("gpt-") or model_id.startswith("o")

    def _model_id(self, model: Any) -> str:
        """读取统一模型 ID。"""

        return getattr(model, "id", model)

    def _runtime_model_name(self, model: Any) -> str:
        """把统一模型标识转换为 OpenAI SDK 使用的模型名。"""

        model_id = self._model_id(model)
        return model_id.split(":", 1)[1] if model_id.startswith("openai:") else model_id

    def _require_api_key(self, model: Any | None = None) -> str:
        """确保 OpenAI API Key 已配置。"""

        config = self._runtime_config(model)
        api_key = config.resolve_api_key() or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise AuthenticationError("OPENAI_API_KEY is not set")
        return api_key

    def _runtime_config(self, model: Any | None = None) -> ProviderConfig:
        """返回当前 provider 的运行时配置视图。"""

        provider_config = getattr(model, "providerConfig", {}) if model is not None else {}
        merged = ProviderConfig(name=self.name)
        if self.config is not None:
            merged = ProviderConfig(**{**merged.__dict__, **self.config.__dict__})
        if provider_config:
            merged.baseUrl = provider_config.get("baseUrl", merged.baseUrl)
            merged.apiKey = provider_config.get("apiKey", merged.apiKey)
            merged.apiKeyEnv = provider_config.get("apiKeyEnv", merged.apiKeyEnv)
            merged.headers = {**merged.headers, **provider_config.get("headers", {})}
            merged.providerOverrides = {**merged.providerOverrides, **provider_config}
        return merged

    def _client_kwargs(self, options: Options, model: Any | None = None) -> dict[str, Any]:
        """构造 OpenAI SDK 客户端初始化参数。"""

        config = self._runtime_config(model)
        return {
            "api_key": self._require_api_key(model),
            "timeout": options.timeout,
            "base_url": config.baseUrl or os.getenv("OPENAI_BASE_URL"),
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
        """把统一工具定义映射为 OpenAI 请求格式。"""

        tools = self._context_tools(context)
        if not tools:
            return None
        return [
            {
                "type": "function",
                "name": self._tool_value(tool, "name"),
                "description": self._tool_value(tool, "description", "") or "",
                "parameters": self._tool_value(tool, "inputSchema", {}),
            }
            for tool in tools
        ]

    def _message_to_input(self, message: Any) -> dict[str, Any]:
        """把统一消息类型映射为 OpenAI `input` 项。"""

        if hasattr(message, "toolCallId"):
            return {
                "type": "function_call_output",
                "call_id": getattr(message, "toolCallId"),
                "output": getattr(message, "text", getattr(message, "content", "")),
            }

        payload: dict[str, Any] = {
            "role": getattr(message, "role", "user"),
            "content": getattr(message, "text", getattr(message, "content", "")),
        }
        tool_calls = getattr(message, "toolCalls", [])
        if tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments_text,
                    },
                }
                for tool_call in tool_calls
            ]
        return payload

    def _provider_reasoning(self, options: Options) -> dict[str, Any]:
        """读取预先映射好的 provider reasoning 配置。"""

        mapped = options.metadata.get("_providerReasoning")
        return mapped if isinstance(mapped, dict) else {}

    def _build_request(self, model: Any, context: Any, options: Options) -> dict[str, Any]:
        """构造 OpenAI Responses API 的请求参数。"""

        input_items = []
        system_prompt = self._context_system_prompt(context)
        if system_prompt:
            input_items.append({"role": "system", "content": system_prompt})
        input_items.extend(self._message_to_input(message) for message in self._context_messages(context))

        request: dict[str, Any] = {
            "model": self._runtime_model_name(model),
            "input": input_items,
        }
        request.update(self._provider_reasoning(options))
        if options.temperature is not None:
            request["temperature"] = options.temperature
        if options.maxTokens is not None:
            request["max_output_tokens"] = options.maxTokens
        if options.metadata:
            request["metadata"] = options.metadata
        tools = self._build_tools(context)
        if tools is not None:
            request["tools"] = tools
        return request

    async def stream(
        self,
        model: Any,
        context: Any,
        options: Options,
    ) -> AsyncIterator[StreamEvent]:
        """把 OpenAI 流式响应转换为统一事件流。"""

        request = self._build_request(model, context, options)
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            builder = EventBuilder(model=model, provider=self.name)
            yield builder.build_error(
                "openai package is not installed",
                metadata={"source": "provider", "provider": self.name, "exception_type": type(exc).__name__},
            )
            return

        client = AsyncOpenAI(**self._client_kwargs(options, model))
        builder = EventBuilder(model=model, provider=self.name)
        final_message = AssistantMessage(content=[], metadata={})
        tool_buffers: dict[str, str] = {}
        text_started = False
        thinking_started = False

        try:
            async with client.responses.stream(**request) as response_stream:
                yield builder.build("start", lifecycle="start", itemType="message")
                async for event in response_stream:
                    event_type = getattr(event, "type", None)
                    raw_event = event if options.includeRawProviderEvents else None
                    if event_type == "response.output_text.delta":
                        delta = getattr(event, "delta", "")
                        if not text_started:
                            text_started = True
                            yield builder.build("text_start", rawEvent=raw_event)
                        final_message.content.append(TextContent(text=delta))
                        yield builder.build("text_delta", text=delta, rawEvent=raw_event)
                    elif event_type in {"response.reasoning_text.delta", "response.reasoning_summary_text.delta"}:
                        delta = getattr(event, "delta", "")
                        if not thinking_started:
                            thinking_started = True
                            yield builder.build("thinking_start", rawEvent=raw_event)
                        final_message.content.append(ThinkingContent(thinking=delta))
                        yield builder.build("thinking_delta", thinking=delta, rawEvent=raw_event)
                    elif event_type == "response.function_call_arguments.delta":
                        tool_call_id = getattr(event, "item_id", "")
                        delta = getattr(event, "delta", "")
                        tool_name = getattr(event, "name", None)
                        if tool_call_id not in tool_buffers:
                            tool_buffers[tool_call_id] = ""
                            yield builder.build("toolcall_start", toolCallId=tool_call_id, toolName=tool_name, rawEvent=raw_event)
                        tool_buffers[tool_call_id] += delta
                        yield builder.build("toolcall_delta", toolCallId=tool_call_id, toolName=tool_name, argumentsDelta=delta, rawEvent=raw_event)
                    elif event_type == "response.function_call_arguments.done":
                        tool_call_id = getattr(event, "item_id", "")
                        tool_name = getattr(event, "name", None)
                        arguments = getattr(event, "arguments", tool_buffers.get(tool_call_id, ""))
                        final_message.content.append(
                            ToolCallContent(
                                id=tool_call_id,
                                name=tool_name or "",
                                arguments=parse_tool_arguments(arguments),
                            )
                        )
                        yield builder.build("toolcall_end", toolCallId=tool_call_id, toolName=tool_name, arguments=arguments, rawEvent=raw_event)

                if text_started:
                    yield builder.build("text_end")
                if thinking_started:
                    yield builder.build("thinking_end")
                final_response = await response_stream.get_final_response()
                final_message.metadata["response"] = final_response
                yield create_done_event(
                    final_message,
                    model=model,
                    provider=self.name,
                    usage=getattr(final_response, "usage", None),
                    stop_reason=getattr(final_response, "status", None),
                    response_id=getattr(final_response, "id", None),
                    provider_metadata={"response_status": getattr(final_response, "status", None)},
                )
        except AuthenticationError as exc:
            yield builder.build_error(str(exc), metadata={"source": "provider", "provider": self.name})
        except Exception as exc:  # pragma: no cover
            error = exc if isinstance(exc, ProviderResponseError) else ProviderResponseError(f"OpenAI streaming failed: {exc}")
            yield builder.build_error(str(error), metadata={"source": "provider", "provider": self.name, "exception_type": type(exc).__name__})


class OpenAICompatibleProvider(OpenAIProvider):
    """用于快速接入 OpenAI 兼容协议服务的 provider。"""

    name = "openai_compatible"

    def supports(self, model: Any) -> bool:
        model_id = self._model_id(model)
        provider = getattr(model, "provider", None)
        if provider:
            return provider == self.name
        return model_id.startswith("openai_compatible:") or model_id.startswith("openai-compatible:")
