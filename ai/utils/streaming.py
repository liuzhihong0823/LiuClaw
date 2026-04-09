from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

from ai.types import AssistantMessage, Model, StreamEvent, TextContent, ThinkingContent, ToolCallContent, parse_tool_arguments

DEFAULT_STREAM_QUEUE_MAXSIZE = 128


class EventBuilder:
    """负责按统一类型快速构造流式事件对象。"""

    def __init__(self, model: Model | None = None, provider: str | None = None) -> None:
        """初始化事件构造器并保存默认的模型与 provider 信息。"""

        self._model = model  # 默认挂载到事件上的模型对象。
        self._provider = provider or (model.provider if model is not None else None)  # 默认 provider 名。

    @property
    def model(self) -> Model | None:
        """返回当前默认绑定的模型对象。"""

        return self._model

    @property
    def provider(self) -> str | None:
        """返回当前默认绑定的 provider 名称。"""

        return self._provider

    def build(self, event_type: str, **kwargs: Any) -> StreamEvent:
        """创建一个统一 `StreamEvent`，并自动补齐默认上下文。"""

        return StreamEvent(
            type=event_type,
            lifecycle=kwargs.pop("lifecycle", None),
            itemType=kwargs.pop("itemType", None),
            messageId=kwargs.pop("messageId", None),
            model=kwargs.pop("model", self._model),
            provider=kwargs.pop("provider", self._provider),
            text=kwargs.pop("text", None),
            thinking=kwargs.pop("thinking", None),
            delta=kwargs.pop("delta", None),
            toolCallId=kwargs.pop("toolCallId", None),
            toolName=kwargs.pop("toolName", None),
            argumentsDelta=kwargs.pop("argumentsDelta", None),
            arguments=kwargs.pop("arguments", None),
            assistantMessage=kwargs.pop("assistantMessage", None),
            toolResultMessage=kwargs.pop("toolResultMessage", None),
            usage=kwargs.pop("usage", None),
            stopReason=kwargs.pop("stopReason", None),
            responseId=kwargs.pop("responseId", None),
            error=kwargs.pop("error", None),
            details=kwargs.pop("details", None),
            metadata=kwargs.pop("metadata", {}),
            providerMetadata=kwargs.pop("providerMetadata", {}),
            rawEvent=kwargs.pop("rawEvent", None),
        )

    def build_error(
        self,
        error: str,
        *,
        metadata: dict[str, Any] | None = None,
        raw_event: Any | None = None,
    ) -> StreamEvent:
        """基于默认上下文创建一个统一 `error` 事件。"""

        return self.build(
            "error",
            lifecycle="error",
            itemType="message",
            error=error,
            details=metadata,
            metadata=dict(metadata or {}),
            rawEvent=raw_event,
        )

    def build_done(
        self,
        assistant_message: AssistantMessage,
        *,
        usage: dict[str, Any] | None = None,
        stop_reason: str | None = None,
        response_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        provider_metadata: dict[str, Any] | None = None,
        raw_event: Any | None = None,
    ) -> StreamEvent:
        """基于默认上下文创建一个统一 `done` 事件。"""

        return create_done_event(
            assistant_message,
            model=self._model,
            provider=self._provider,
            usage=usage,
            stop_reason=stop_reason,
            response_id=response_id,
            metadata=metadata,
            provider_metadata=provider_metadata,
            raw_event=raw_event,
        )


class StreamAccumulator:
    """聚合统一事件流，生成最终的 `AssistantMessage`。"""

    def __init__(self) -> None:
        """初始化聚合状态。"""

        self._assistant_message = AssistantMessage()  # 正在累积的 assistant 消息。
        self._tool_call_index: dict[str, int] = {}  # tool_call_id 到内容块索引的映射。
        self._usage: dict[str, Any] | None = None  # 最近一次 done 事件的 usage 信息。
        self._stop_reason: str | None = None  # 最近一次 done 事件的停止原因。
        self._done_event: StreamEvent | None = None  # 最近一次收到的完成事件。
        self._error_event: StreamEvent | None = None  # 最近一次收到的错误事件。

    @property
    def assistant_message(self) -> AssistantMessage:
        """返回当前已聚合的 assistant 消息。"""

        return self._assistant_message

    @property
    def usage(self) -> dict[str, Any] | None:
        """返回最近一次流式聚合得到的 usage 信息。"""

        return self._usage

    @property
    def stop_reason(self) -> str | None:
        """返回最近一次流式聚合得到的停止原因。"""

        return self._stop_reason

    @property
    def done_event(self) -> StreamEvent | None:
        """返回最近一次接收到的 `done` 事件。"""

        return self._done_event

    @property
    def error_event(self) -> StreamEvent | None:
        """返回最近一次接收到的 `error` 事件。"""

        return self._error_event

    @property
    def is_finished(self) -> bool:
        """返回当前流是否已经收到终止事件。"""

        return self._done_event is not None or self._error_event is not None

    def apply(self, event: StreamEvent) -> AssistantMessage | None:
        """消费一个统一事件，并在 `done` 时返回最终消息。"""

        if event.itemType == "text" and event.lifecycle == "update" and event.text:
            self._assistant_message.content.append(TextContent(text=event.text))
        elif event.itemType == "thinking" and event.lifecycle == "update" and event.thinking:
            self._assistant_message.content.append(ThinkingContent(thinking=event.thinking))
        elif event.itemType == "tool_call" and event.lifecycle == "start" and event.toolCallId:
            self._ensure_tool_call(event.toolCallId, event.toolName)
        elif event.itemType == "tool_call" and event.lifecycle == "update" and event.toolCallId:
            tool_call = self._ensure_tool_call(event.toolCallId, event.toolName)
            tool_call.arguments = f"{tool_call.arguments}{event.argumentsDelta or ''}"
        elif event.itemType == "tool_call" and event.lifecycle == "done" and event.toolCallId:
            tool_call = self._ensure_tool_call(event.toolCallId, event.toolName)
            if event.arguments is not None:
                tool_call.arguments = parse_tool_arguments(event.arguments)
        elif event.type == "done" and event.lifecycle == "done":
            self._done_event = event
            self._usage = event.usage
            self._stop_reason = event.stopReason
            if event.assistantMessage is not None:
                self._assistant_message = replace(event.assistantMessage)
            return self._assistant_message
        elif event.type == "error" and event.lifecycle == "error":
            self._error_event = event
        return None

    def _ensure_tool_call(self, tool_call_id: str, tool_name: str | None) -> ToolCallContent:
        """确保指定 id 的工具调用在聚合状态中存在。"""

        if tool_call_id in self._tool_call_index:
            tool_call = self._assistant_message.content[self._tool_call_index[tool_call_id]]
            if tool_name and not tool_call.name:
                tool_call.name = tool_name
            return tool_call

        tool_call = ToolCallContent(id=tool_call_id, name=tool_name or "", arguments="")
        self._assistant_message.content.append(tool_call)
        self._tool_call_index[tool_call_id] = len(self._assistant_message.content) - 1
        return tool_call


async def create_event_queue(maxsize: int = DEFAULT_STREAM_QUEUE_MAXSIZE) -> asyncio.Queue[StreamEvent]:
    """创建一个默认有界的事件队列。"""

    return asyncio.Queue(maxsize=maxsize)


async def enqueue_event(
    queue: asyncio.Queue[StreamEvent],
    event: StreamEvent,
    *,
    put_timeout: float | None = None,
) -> None:
    """向事件队列写入一个事件，并在需要时等待背压释放。"""

    if put_timeout is None:
        await queue.put(event)
        return
    await asyncio.wait_for(queue.put(event), timeout=put_timeout)


async def consume_queue(
    queue: asyncio.Queue[StreamEvent],
) -> AsyncIterator[StreamEvent]:
    """持续从事件队列消费，并在 `done/error` 后结束。"""

    while True:
        event = await queue.get()
        yield event
        queue.task_done()
        if event.is_terminal:
            return


async def drain_queue_to_accumulator(
    queue: asyncio.Queue[StreamEvent],
    *,
    accumulator: StreamAccumulator | None = None,
) -> AssistantMessage:
    """从队列消费事件并聚合，直到收到 `done` 或 `error`。"""

    stream_accumulator = accumulator or StreamAccumulator()
    async for event in consume_queue(queue):
        final_message = stream_accumulator.apply(event)
        if final_message is not None:
            return final_message
        if event.type == "error":
            return stream_accumulator.assistant_message
    return stream_accumulator.assistant_message


async def forward_stream_to_queue(
    event_stream: AsyncIterator[StreamEvent],
    queue: asyncio.Queue[StreamEvent],
    *,
    builder: EventBuilder | None = None,
    put_timeout: float | None = None,
) -> None:
    """把 provider 产出的事件流桥接到队列中。"""

    try:
        async for event in event_stream:
            await enqueue_event(queue, event, put_timeout=put_timeout)
            if event.is_terminal:
                return
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        error_builder = builder or EventBuilder()
        await enqueue_event(
            queue,
            error_builder.build_error(str(exc), metadata={"source": "forward_stream_to_queue"}),
            put_timeout=put_timeout,
        )


async def finalize_producer_error(
    queue: asyncio.Queue[StreamEvent],
    error: str,
    *,
    builder: EventBuilder | None = None,
    put_timeout: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """在 producer 失败时向队列补发统一 `error` 事件。"""

    error_builder = builder or EventBuilder()
    await enqueue_event(
        queue,
        error_builder.build_error(error, metadata=metadata),
        put_timeout=put_timeout,
    )


async def cancel_producer_task(
    task: asyncio.Task[Any],
    queue: asyncio.Queue[StreamEvent],
    *,
    builder: EventBuilder | None = None,
    put_timeout: float | None = None,
) -> None:
    """取消一个 producer 任务，并在必要时向队列补发 `error` 事件。"""

    if task.done():
        try:
            await task
        except Exception as exc:
            await finalize_producer_error(
                queue,
                str(exc),
                builder=builder,
                put_timeout=put_timeout,
                metadata={"source": "cancel_producer_task"},
            )
        return

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        await finalize_producer_error(
            queue,
            "stream producer cancelled",
            builder=builder,
            put_timeout=put_timeout,
            metadata={"source": "cancel_producer_task"},
        )
    except Exception as exc:
        await finalize_producer_error(
            queue,
            str(exc),
            builder=builder,
            put_timeout=put_timeout,
            metadata={"source": "cancel_producer_task"},
        )


def create_done_event(
    assistant_message: AssistantMessage,
    *,
    model: Model | None = None,
    provider: str | None = None,
    usage: dict[str, Any] | None = None,
    stop_reason: str | None = None,
    response_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    provider_metadata: dict[str, Any] | None = None,
    raw_event: Any | None = None,
) -> StreamEvent:
    """根据最终消息构造统一 `done` 事件。"""

    return StreamEvent(
        type="done",
        lifecycle="done",
        itemType="message",
        model=model,
        provider=provider or (model.provider if model is not None else None),
        assistantMessage=assistant_message,
        usage=usage,
        stopReason=stop_reason,
        responseId=response_id,
        metadata=dict(metadata or {}),
        providerMetadata=dict(provider_metadata or {}),
        rawEvent=raw_event,
    )
