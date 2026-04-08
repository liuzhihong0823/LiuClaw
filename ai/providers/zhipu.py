from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

from ai.config import ProviderConfig
from ai.errors import AuthenticationError, ProviderResponseError
from ai.options import Options
from ai.providers.base import Provider
from ai.types import AssistantMessage, StreamEvent, TextContent, ThinkingContent, ToolCallContent, parse_tool_arguments
from ai.utils.streaming import EventBuilder, create_done_event


class ZhipuProvider(Provider):
    """智谱 GLM provider 的统一适配实现。"""

    name = "zhipu"
    _TOOL_STREAM_MODELS = {"glm-4.6", "glm-4.7"}

    def supports(self, model: Any) -> bool:
        """判断模型是否应由智谱 provider 处理。"""

        model_id = self._model_id(model)
        provider = getattr(model, "provider", None)
        if provider:
            return provider == self.name
        return model_id.startswith("zhipu:") or model_id.startswith("glm-")

    def _model_id(self, model: Any) -> str:
        """读取统一模型 ID。"""

        return getattr(model, "id", model)

    def _runtime_model_name(self, model: Any) -> str:
        """把统一模型标识转换为智谱 API 使用的模型名。"""

        model_id = self._model_id(model)
        return model_id.split(":", 1)[1] if model_id.startswith("zhipu:") else model_id

    def _require_api_key(self, model: Any | None = None) -> str:
        """确保智谱 API Key 已配置。"""

        config = self._runtime_config(model)
        api_key = config.resolve_api_key() or os.getenv("ZHIPU_API_KEY") or os.getenv("ZHIPUAI_API_KEY")
        if not api_key:
            raise AuthenticationError("ZAI_API_KEY or ZHIPUAI_API_KEY is not set")
        return api_key

    def _base_url(self, model: Any | None = None) -> str:
        """返回智谱 API 根地址。"""

        config = self._runtime_config(model)
        return (config.baseUrl or os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")).rstrip("/")

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

    def _headers(self, model: Any | None = None) -> dict[str, str]:
        """构造智谱请求头。"""

        return {
            "Authorization": f"Bearer {self._require_api_key(model)}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
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
        """把统一工具定义映射为智谱请求格式。"""

        tools = self._context_tools(context)
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": self._tool_value(tool, "name"),
                    "description": self._tool_value(tool, "description", "") or "",
                    "parameters": self._tool_value(tool, "inputSchema", {}),
                },
            }
            for tool in tools
        ]

    def _message_to_input(self, message: Any) -> dict[str, Any]:
        """把统一消息类型映射为智谱 `messages` 项。"""

        if hasattr(message, "toolCallId"):
            return {
                "role": "tool",
                "tool_call_id": getattr(message, "toolCallId"),
                "content": getattr(message, "text", getattr(message, "content", "")),
            }

        payload: dict[str, Any] = {
            "role": getattr(message, "role", "user"),
            "content": getattr(message, "text", getattr(message, "content", "")),
        }

        thinking = getattr(message, "thinking", "")
        if thinking:
            payload["reasoning_content"] = thinking

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
        """构造智谱 chat/completions 请求参数。"""

        messages = []
        system_prompt = self._context_system_prompt(context)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self._message_to_input(message) for message in self._context_messages(context))

        request: dict[str, Any] = {
            "model": self._runtime_model_name(model),
            "messages": messages,
            "stream": True,
        }
        request.update(self._provider_reasoning(options))

        if options.temperature is not None:
            request["temperature"] = options.temperature
        if options.maxTokens is not None:
            request["max_tokens"] = options.maxTokens

        tools = self._build_tools(context)
        if tools is not None:
            request["tools"] = tools
            if request["model"] in self._TOOL_STREAM_MODELS:
                request["tool_stream"] = True
        return request

    async def _iter_sse_chunks(
        self,
        model: Any,
        request: dict[str, Any],
        *,
        timeout: float | None,
    ) -> AsyncIterator[dict[str, Any]]:
        """调用智谱 SSE 接口，并逐个产出 JSON chunk。"""

        try:
            import httpx
        except ImportError as exc:
            raise ProviderResponseError("httpx package is not installed") from exc

        url = f"{self._base_url(model)}/chat/completions"
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, headers=self._headers(model), json=request) as response:
                if response.status_code >= 400:
                    text = await response.aread()
                    raise ProviderResponseError(
                        f"Zhipu request failed with status {response.status_code}: {text.decode('utf-8', errors='ignore')}"
                    )

                data_lines: list[str] = []
                async for line in response.aiter_lines():
                    if not line:
                        payload = "\n".join(data_lines).strip()
                        data_lines.clear()
                        if not payload:
                            continue
                        if payload == "[DONE]":
                            return
                        try:
                            yield json.loads(payload)
                        except json.JSONDecodeError as exc:
                            raise ProviderResponseError(f"Zhipu returned malformed SSE payload: {payload}") from exc
                        continue
                    if line.startswith("data:"):
                        data_lines.append(line[5:].strip())

                if data_lines:
                    payload = "\n".join(data_lines).strip()
                    if payload and payload != "[DONE]":
                        try:
                            yield json.loads(payload)
                        except json.JSONDecodeError as exc:
                            raise ProviderResponseError(f"Zhipu returned malformed SSE payload: {payload}") from exc

    def _close_text_if_needed(
        self,
        builder: EventBuilder,
        events: list[StreamEvent],
        *,
        text_started: bool,
    ) -> bool:
        """在需要时补一个 `text_end` 事件。"""

        if text_started:
            events.append(builder.build("text_end"))
        return False

    def _close_thinking_if_needed(
        self,
        builder: EventBuilder,
        events: list[StreamEvent],
        *,
        thinking_started: bool,
    ) -> bool:
        """在需要时补一个 `thinking_end` 事件。"""

        if thinking_started:
            events.append(builder.build("thinking_end"))
        return False

    async def stream(
        self,
        model: Any,
        context: Any,
        options: Options,
    ) -> AsyncIterator[StreamEvent]:
        """把智谱流式响应转换为统一事件流。"""

        request = self._build_request(model, context, options)
        builder = EventBuilder(model=model, provider=self.name)
        final_message = AssistantMessage(content=[], metadata={})
        tool_buffers: dict[str, str] = {}
        tool_names: dict[str, str] = {}
        tool_finished: set[str] = set()
        text_started = False
        thinking_started = False
        stop_reason: str | None = None
        usage: dict[str, Any] | None = None
        response_id: str | None = None

        try:
            yield builder.build("start", lifecycle="start", itemType="message")
            async for chunk in self._iter_sse_chunks(model, request, timeout=options.timeout):
                raw_event = chunk if options.includeRawProviderEvents else None
                response_id = chunk.get("id", response_id)
                choices = chunk.get("choices") or []
                usage = chunk.get("usage", usage)
                if not choices:
                    continue

                choice = choices[0]
                delta = choice.get("delta") or {}
                stop_reason = choice.get("finish_reason") or stop_reason
                emitted_events: list[StreamEvent] = []

                thinking_delta = delta.get("reasoning_content")
                if thinking_delta:
                    if text_started:
                        text_started = self._close_text_if_needed(builder, emitted_events, text_started=text_started)
                    if not thinking_started:
                        emitted_events.append(builder.build("thinking_start", rawEvent=raw_event))
                        thinking_started = True
                    final_message.content.append(ThinkingContent(thinking=thinking_delta))
                    emitted_events.append(builder.build("thinking_delta", thinking=thinking_delta, rawEvent=raw_event))

                text_delta = delta.get("content")
                if text_delta:
                    if thinking_started:
                        thinking_started = self._close_thinking_if_needed(builder, emitted_events, thinking_started=thinking_started)
                    if not text_started:
                        emitted_events.append(builder.build("text_start", rawEvent=raw_event))
                        text_started = True
                    final_message.content.append(TextContent(text=text_delta))
                    emitted_events.append(builder.build("text_delta", text=text_delta, rawEvent=raw_event))

                for tool_delta in delta.get("tool_calls") or []:
                    key = tool_delta.get("id") or f"index_{tool_delta.get('index', 0)}"
                    function = tool_delta.get("function") or {}
                    tool_name = function.get("name") or tool_names.get(key)
                    arguments_delta = function.get("arguments", "")
                    if key not in tool_buffers:
                        tool_buffers[key] = ""
                        emitted_events.append(
                            builder.build(
                                "toolcall_start",
                                toolCallId=key,
                                toolName=tool_name,
                                rawEvent=raw_event,
                            )
                        )
                    if tool_name:
                        tool_names[key] = tool_name
                    if arguments_delta:
                        tool_buffers[key] += arguments_delta
                        emitted_events.append(
                            builder.build(
                                "toolcall_delta",
                                toolCallId=key,
                                toolName=tool_names.get(key),
                                argumentsDelta=arguments_delta,
                                rawEvent=raw_event,
                            )
                        )

                if stop_reason in {"tool_calls", "stop", "length"}:
                    if text_started:
                        text_started = self._close_text_if_needed(builder, emitted_events, text_started=text_started)
                    if thinking_started:
                        thinking_started = self._close_thinking_if_needed(builder, emitted_events, thinking_started=thinking_started)
                    for key, arguments in list(tool_buffers.items()):
                        if key in tool_finished:
                            continue
                        tool_finished.add(key)
                        final_message.content.append(
                            ToolCallContent(
                                id=key,
                                name=tool_names.get(key, ""),
                                arguments=parse_tool_arguments(arguments),
                            )
                        )
                        emitted_events.append(
                            builder.build(
                                "toolcall_end",
                                toolCallId=key,
                                toolName=tool_names.get(key),
                                arguments=arguments,
                                rawEvent=raw_event,
                            )
                        )

                for event in emitted_events:
                    yield event

            if text_started:
                yield builder.build("text_end")
            if thinking_started:
                yield builder.build("thinking_end")
            for key, arguments in list(tool_buffers.items()):
                if key in tool_finished:
                    continue
                final_message.content.append(
                    ToolCallContent(
                        id=key,
                        name=tool_names.get(key, ""),
                        arguments=parse_tool_arguments(arguments),
                    )
                )
                yield builder.build("toolcall_end", toolCallId=key, toolName=tool_names.get(key), arguments=arguments)
            yield create_done_event(
                final_message,
                model=model,
                provider=self.name,
                usage=usage,
                stop_reason=stop_reason,
                response_id=response_id,
                provider_metadata={"request_model": request.get("model")},
            )
        except AuthenticationError as exc:
            yield builder.build_error(str(exc), metadata={"source": "provider", "provider": self.name})
        except Exception as exc:
            error = exc if isinstance(exc, ProviderResponseError) else ProviderResponseError(f"Zhipu streaming failed: {exc}")
            yield builder.build_error(str(error), metadata={"source": "provider", "provider": self.name, "exception_type": type(exc).__name__})
