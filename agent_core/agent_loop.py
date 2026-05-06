from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from typing import Any

from ai import streamSimple
from ai.session import StreamSession
from ai.types import (
    AssistantMessage,
    ContentBlocks,
    Context,
    ConversationMessage,
    Model,
    StreamEvent,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolCallContent,
    ToolResultContent,
    ToolResultMessage,
    UserMessage,
    ensure_message,
    ensure_model,
    parse_tool_arguments,
)
from ai.utils.schema_validation import SchemaValidationError, validate_tool_arguments

from .types import (
    AbortSignal,
    AfterToolCallContext,
    AfterToolCallPass,
    AfterToolCallReplace,
    AgentContext,
    AgentError,
    AgentEvent,
    AgentLoopConfig,
    AgentRuntimeFlags,
    AgentState,
    AgentTool,
    BeforeToolCallAllow,
    BeforeToolCallContext,
    BeforeToolCallError,
    BeforeToolCallSkip,
    RetryContext,
    default_convert_to_llm,
    default_retry_policy,
    default_transform_context,
)


@dataclass(slots=True)
class PreparedToolCall:
    """表示已经通过可用性与参数校验的工具调用。"""

    toolCall: ToolCall  # 原始工具调用对象。
    tool: AgentTool  # 匹配到的可执行工具。
    params: Any  # 已解析和校验过的参数。
    assistantMessage: AssistantMessage | None  # 发起该调用的 assistant 消息。
    agentContext: AgentContext  # 调用发生时的 Agent 上下文快照。


@dataclass(slots=True)
class PreparedToolCallError:
    """表示工具调用在预处理阶段失败或被短路。"""

    toolCall: ToolCall  # 原始工具调用对象。
    error: AgentError | None = None  # 预处理阶段的错误信息。
    shortcutResult: ToolResultMessage | None = None  # 被短路时直接返回的工具结果。


async def _maybe_await(value: Any) -> Any:
    """在值为 awaitable 时等待完成，否则原样返回。"""

    if inspect.isawaitable(value):
        return await value
    return value


def _copy_tools(tools: list[AgentTool]) -> list[AgentTool]:
    """复制工具定义，避免状态与配置共享可变对象。"""

    return [replace(tool) for tool in tools]


def _copy_messages(messages: list[Any]) -> list[Any]:
    """复制消息历史，避免事件快照共享引用。"""

    copied: list[Any] = []
    for message in messages:
        try:
            copied.append(replace(message))
        except TypeError:
            copied.append(message)
    return copied


def _snapshot_state(state: AgentState) -> AgentState:
    """生成用于事件流输出的状态快照。"""

    return AgentState(
        systemPrompt=state.systemPrompt,
        model=state.model,
        thinking=state.thinking,
        tools=_copy_tools(state.tools),
        messages=_copy_messages(state.messages),
        stream_message=replace(state.stream_message) if state.stream_message is not None else None,
        pending_tool_calls=[replace(tool_call) for tool_call in state.pending_tool_calls],
        error=replace(state.error) if state.error is not None else None,
        runtime_flags=replace(state.runtime_flags),
    )


def _normalize_messages(messages: list[Any] | None) -> list[Any]:
    """把输入统一归一化为消息列表。"""

    normalized: list[Any] = []
    for message in messages or []:
        if isinstance(message, dict) and "role" in message:
            normalized.append(ensure_message(message))
        else:
            normalized.append(message)
    return normalized


def _find_tool(tools: list[AgentTool], name: str) -> AgentTool | None:
    """根据工具名查找可执行工具。"""

    for tool in tools:
        if tool.name == name:
            return tool
    return None


async def _to_agent_context(state: AgentState, loop: AgentLoopConfig) -> AgentContext:
    """从当前状态构造一次 AgentContext。"""

    convert_fn = loop.convert_to_llm or default_convert_to_llm
    llm_messages = list(await _maybe_await(convert_fn(state.messages, state)))
    context = AgentContext(
        systemPrompt=state.systemPrompt,
        messages=llm_messages,
        tools=[
            Tool(
                name=tool.name,
                description=tool.description,
                inputSchema=dict(tool.inputSchema),
                metadata=dict(tool.metadata),
                renderMetadata=dict(tool.renderMetadata),
            )
            for tool in state.tools
        ],
    )
    transform_fn = loop.transform_context or default_transform_context
    transformed = await _maybe_await(transform_fn(context, state))
    return transformed


def _context_to_llm_context(context: AgentContext) -> Context:
    """把 AgentContext 转成 ai.Context。"""

    return Context(
        systemPrompt=context.systemPrompt,
        messages=[ensure_message(message) for message in context.messages],
        tools=context.tools,
    )


def _normalize_tool_result(result: Any, tool_call: ToolCall, *, is_error: bool = False) -> ToolResultMessage:
    """把任意工具执行产出统一转成 ToolResultMessage。"""

    if isinstance(result, ToolResultMessage):
        normalized = replace(result)
        normalized.toolCallId = normalized.toolCallId or tool_call.id
        normalized.toolName = normalized.toolName or tool_call.name
        normalized.isError = normalized.isError or is_error
        if normalized.isError:
            normalized.metadata["error"] = True
        return normalized
    if isinstance(result, dict):
        metadata = dict(result.get("metadata", {}))
        content = result.get("content")
        if content is None:
            content = json.dumps(result, ensure_ascii=False, sort_keys=True)
        tool_message = ToolResultMessage(
            toolCallId=str(result.get("toolCallId", tool_call.id)),
            toolName=str(result.get("toolName", tool_call.name)),
            content=content,
            metadata=metadata,
            isError=bool(result.get("isError", metadata.get("error", False) or is_error)),
            details=result.get("details"),
        )
        if tool_message.isError:
            tool_message.metadata["error"] = True
        return tool_message
    tool_message = ToolResultMessage(
        toolCallId=tool_call.id,
        toolName=tool_call.name,
        content=str(result),
        metadata={},
        isError=is_error,
    )
    if tool_message.isError:
        tool_message.metadata["error"] = True
    return tool_message


def _tool_error_result(tool_call: ToolCall, error: AgentError | str) -> ToolResultMessage:
    """根据错误对象生成标准工具错误结果。"""

    message = error.message if isinstance(error, AgentError) else str(error)
    details = error.details if isinstance(error, AgentError) else None
    return ToolResultMessage(
        toolCallId=tool_call.id,
        toolName=tool_call.name,
        content=ContentBlocks([ToolResultContent(text=message)]),
        metadata={"error": True},
        isError=True,
        details=details,
    )


def _copy_payload_value(value: Any) -> Any:
    try:
        return replace(value)
    except TypeError:
        return value


async def _emit_event(
    queue: asyncio.Queue[AgentEvent],
    *,
    event_type: str,
    state: AgentState,
    payload: dict[str, Any] | None = None,
) -> AgentEvent:
    """构造事件并写入队列。"""

    event = AgentEvent(
        type=event_type,
        state=_snapshot_state(state),
        payload={key: _copy_payload_value(value) for key, value in (payload or {}).items()},
    )
    await queue.put(event)
    return event


async def _resolve_control_messages(fn, state: AgentState, signal: AbortSignal) -> list[Any]:
    """获取 steering 或 follow-up 消息并统一归一化。"""

    if fn is None:
        return []
    signal.throw_if_aborted()
    try:
        result = fn(state, signal)
    except TypeError:
        result = fn(state)
    return _normalize_messages(await _maybe_await(result))


async def _default_stream(
    model: Model | str,
    context: AgentContext,
    thinking,
    registry,
    *,
    signal: AbortSignal | None = None,
) -> StreamSession[StreamEvent]:
    """使用默认的 `ai.streamSimple` 打开一次流式会话。"""

    if signal is not None:
        signal.throw_if_aborted()
    return await streamSimple(
        model,
        _context_to_llm_context(context),
        reasoning=thinking,
        registry=registry,
    )


async def _open_stream(
    state: AgentState,
    loop: AgentLoopConfig,
    signal: AbortSignal,
) -> StreamSession[StreamEvent]:
    """根据循环配置获取底层 AI 流式会话。"""

    model = ensure_model(loop.model or state.model)
    context = await _to_agent_context(state, loop)
    if getattr(loop, "traceRecorder", None) is not None:
        loop.traceRecorder.record_context_snapshot(state.runtime_flags.turnIndex, context)
    stream_fn = loop.stream or _default_stream
    try:
        return await stream_fn(model, context, loop.thinking or state.thinking, loop.registry, signal=signal)
    except TypeError:
        return await stream_fn(model, context, loop.thinking or state.thinking, loop.registry)


def _append_text_delta(message: AssistantMessage, text: str) -> None:
    """把文本增量追加到当前流式消息。"""

    if message.content and isinstance(message.content[-1], TextContent):
        message.content[-1].text = f"{message.content[-1].text}{text}"
        return
    message.content.append(TextContent(text=text))


def _append_thinking_delta(message: AssistantMessage, thinking: str) -> None:
    """把 thinking 增量追加到当前流式消息。"""

    if message.content and isinstance(message.content[-1], ThinkingContent):
        message.content[-1].thinking = f"{message.content[-1].thinking}{thinking}"
        return
    message.content.append(ThinkingContent(thinking=thinking))


async def _handle_provider_error(
    state: AgentState,
    loop: AgentLoopConfig,
    signal: AbortSignal,
    error_message: str,
) -> bool:
    """根据 provider 错误和重试策略决定是否继续重试。"""

    state.error = AgentError(kind="provider_error", message=error_message, retriable=True)
    retry_policy = loop.retryPolicy or default_retry_policy
    decision = await _maybe_await(
        retry_policy(
            RetryContext(
                error=state.error,
                state=state,
                attempt=state.runtime_flags.retryCount + 1,
                signal=signal,
            )
        )
    )
    if not decision.shouldRetry:
        if getattr(loop, "traceRecorder", None) is not None:
            loop.traceRecorder.record_retry_decision(
                state.runtime_flags.turnIndex,
                error_message=error_message,
                retry_count=state.runtime_flags.retryCount + 1,
                should_retry=False,
                delay_seconds=decision.delaySeconds,
            )
        return False
    state.runtime_flags.retryCount += 1
    if getattr(loop, "traceRecorder", None) is not None:
        loop.traceRecorder.record_retry_decision(
            state.runtime_flags.turnIndex,
            error_message=error_message,
            retry_count=state.runtime_flags.retryCount,
            should_retry=True,
            delay_seconds=decision.delaySeconds,
        )
    if decision.delaySeconds > 0:
        await asyncio.sleep(decision.delaySeconds)
    return True


async def streamAssistantResponse(
    state: AgentState,
    loop: AgentLoopConfig,
    queue: asyncio.Queue[AgentEvent],
    signal: AbortSignal,
) -> AssistantMessage | None:
    """获取 AI 的单轮回复，并把流式过程映射到事件流中。"""

    while True:
        try:
            session = await _open_stream(state, loop, signal)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            should_retry = await _handle_provider_error(state, loop, signal, str(exc))
            if should_retry:
                continue
            return None

        final_message: AssistantMessage | None = None
        message_started = False
        state.runtime_flags.isStreaming = True
        state.stream_message = AssistantMessage(content=[])

        try:
            async for raw_event in session.consume():
                signal.throw_if_aborted()
                if raw_event.lifecycle == "start" and raw_event.type == "start":
                    if not message_started:
                        message_started = True
                        await _emit_event(
                            queue,
                            event_type="message_start",
                            state=state,
                            payload={"message": state.stream_message},
                        )
                    continue

                if raw_event.lifecycle == "done" and raw_event.type == "done":
                    final_message = replace(raw_event.assistantMessage or state.stream_message or AssistantMessage(content=[]))
                    state.stream_message = replace(final_message)
                    if not message_started:
                        message_started = True
                        await _emit_event(
                            queue,
                            event_type="message_start",
                            state=state,
                            payload={"message": state.stream_message},
                        )
                    state.messages.append(replace(final_message))
                    await _emit_event(
                        queue,
                        event_type="message_end",
                        state=state,
                        payload={"message": final_message},
                    )
                    return final_message

                if raw_event.lifecycle == "error" and raw_event.type == "error":
                    should_retry = await _handle_provider_error(
                        state,
                        loop,
                        signal,
                        raw_event.error or "provider emitted error event",
                    )
                    if should_retry:
                        break
                    return None

                if state.stream_message is None:
                    state.stream_message = AssistantMessage(content=[])
                if raw_event.itemType == "text" and raw_event.lifecycle == "update":
                    _append_text_delta(state.stream_message, raw_event.text or raw_event.delta or "")
                elif raw_event.itemType == "thinking" and raw_event.lifecycle == "update":
                    _append_thinking_delta(state.stream_message, raw_event.thinking or raw_event.delta or "")
                elif raw_event.itemType == "tool_call" and raw_event.lifecycle == "start" and raw_event.toolCallId:
                    state.stream_message.content.append(
                        ToolCallContent(id=raw_event.toolCallId, name=raw_event.toolName or "", arguments={})
                    )
                elif raw_event.itemType == "tool_call" and raw_event.lifecycle == "update" and raw_event.toolCallId:
                    for block in state.stream_message.content:
                        if isinstance(block, ToolCallContent) and block.id == raw_event.toolCallId:
                            current = block.arguments_text
                            block.arguments = parse_tool_arguments(f"{current}{raw_event.argumentsDelta or ''}")
                            break
                elif raw_event.itemType == "tool_call" and raw_event.lifecycle == "done" and raw_event.toolCallId:
                    for block in state.stream_message.content:
                        if isinstance(block, ToolCallContent) and block.id == raw_event.toolCallId:
                            block.arguments = raw_event.arguments
                            break

                if not message_started:
                    message_started = True
                    await _emit_event(
                        queue,
                        event_type="message_start",
                        state=state,
                        payload={"message": state.stream_message},
                    )
                await _emit_event(
                    queue,
                    event_type="message_update",
                    state=state,
                    payload={
                        "message": state.stream_message,
                        "messageDelta": raw_event.text or raw_event.thinking or raw_event.argumentsDelta,
                        "rawEvent": raw_event,
                    },
                )
            else:
                return None
        finally:
            state.runtime_flags.isStreaming = False
            await session.close()

        if state.error is not None and state.error.kind == "provider_error" and state.error.retriable:
            continue
        if final_message is None and state.stream_message is not None and message_started:
            final_message = replace(state.stream_message)
            state.messages.append(replace(final_message))
            await _emit_event(
                queue,
                event_type="message_end",
                state=state,
                payload={"message": final_message},
            )
            return final_message
        return final_message


async def prepareToolCall(
    tool_call: ToolCall,
    assistant_message: AssistantMessage | None,
    state: AgentState,
    loop: AgentLoopConfig,
    signal: AbortSignal,
) -> PreparedToolCall | PreparedToolCallError:
    """准备工具调用，检查工具、参数与 beforeToolCall。"""

    tool = _find_tool(state.tools, tool_call.name)
    if tool is None or tool.executor is None:
        return PreparedToolCallError(
            toolCall=tool_call,
            error=AgentError(kind="tool_error", message=f"Tool '{tool_call.name}' is not available"),
        )

    try:
        params = parse_tool_arguments(tool_call.arguments)
    except Exception as exc:
        return PreparedToolCallError(
            toolCall=tool_call,
            error=AgentError(kind="tool_error", message=f"Invalid tool arguments: {exc}"),
        )

    try:
        validate_tool_arguments(tool, params)
    except SchemaValidationError as exc:
        return PreparedToolCallError(
            toolCall=tool_call,
            error=AgentError(kind="tool_error", message=str(exc), details={"tool": tool_call.name}),
        )

    context = BeforeToolCallContext(
        state=state,
        tool=tool,
        toolCall=tool_call,
        params=params,
        assistantMessage=assistant_message,
        agentContext=await _to_agent_context(state, loop),
        signal=signal,
    )
    before_result = await _maybe_await(loop.beforeToolCall(context)) if loop.beforeToolCall else None
    if before_result is None or isinstance(before_result, BeforeToolCallAllow):
        return PreparedToolCall(
            toolCall=tool_call,
            tool=tool,
            params=params,
            assistantMessage=assistant_message,
            agentContext=context.agentContext,
        )
    if isinstance(before_result, BeforeToolCallSkip):
        if getattr(loop, "traceRecorder", None) is not None:
            loop.traceRecorder.record_before_tool_result(
                state.runtime_flags.turnIndex,
                tool_call,
                outcome="skip",
            )
        return PreparedToolCallError(
            toolCall=tool_call,
            shortcutResult=_normalize_tool_result(before_result.result, tool_call),
        )
    if isinstance(before_result, BeforeToolCallError):
        if getattr(loop, "traceRecorder", None) is not None:
            loop.traceRecorder.record_before_tool_result(
                state.runtime_flags.turnIndex,
                tool_call,
                outcome="error",
                error_message=before_result.error,
            )
        return PreparedToolCallError(
            toolCall=tool_call,
            error=AgentError(kind="tool_error", message=before_result.error, details=before_result.details),
        )
    return PreparedToolCallError(
        toolCall=tool_call,
        error=AgentError(kind="runtime_error", message="Unsupported beforeToolCall result"),
    )


async def executePreparedToolCall(
    prepared: PreparedToolCall,
    state: AgentState,
    queue: asyncio.Queue[AgentEvent],
    signal: AbortSignal,
) -> Any:
    """执行已经通过准备阶段的工具调用。"""

    state.pending_tool_calls = [replace(prepared.toolCall)]

    async def _on_update(update: Any) -> None:
        await _emit_event(
            queue,
            event_type="tool_execution_update",
            state=state,
            payload={"toolCall": prepared.toolCall, "update": update},
        )

    await _emit_event(
        queue,
        event_type="tool_execution_update",
        state=state,
        payload={"toolCall": prepared.toolCall, "update": {"status": "running"}},
    )
    try:
        try:
            return await _maybe_await(prepared.tool.executor(prepared.toolCall.id, prepared.params, signal, _on_update))
        except TypeError:
            return await _maybe_await(prepared.tool.executor(prepared.toolCall.arguments_text, prepared.agentContext))
    finally:
        state.pending_tool_calls = []


async def emitToolCallOutcome(
    tool_call: ToolCall,
    result: ToolResultMessage,
    state: AgentState,
    queue: asyncio.Queue[AgentEvent],
    *,
    error: AgentError | None = None,
) -> ToolResultMessage:
    """发出工具执行结果，并把其写入上下文历史。"""

    tool_message = replace(result)
    if error is not None:
        tool_message = _tool_error_result(tool_call, error)
        state.error = error

    await _emit_event(
        queue,
        event_type="tool_execution_end",
        state=state,
        payload={"toolCall": tool_call, "toolResult": tool_message, "error": error},
    )
    state.messages.append(replace(tool_message))
    await _emit_event(
        queue,
        event_type="message_start",
        state=state,
        payload={"message": tool_message},
    )
    await _emit_event(
        queue,
        event_type="message_end",
        state=state,
        payload={"message": tool_message},
    )
    return tool_message


async def finalizeExecutedToolCall(
    prepared: PreparedToolCall,
    executed_result: Any,
    state: AgentState,
    loop: AgentLoopConfig,
    queue: asyncio.Queue[AgentEvent],
    signal: AbortSignal,
) -> ToolResultMessage:
    """执行 afterToolCall，并最终落盘工具结果。"""

    result = _normalize_tool_result(executed_result, prepared.toolCall)
    if loop.afterToolCall:
        after_context = AfterToolCallContext(
            state=state,
            tool=prepared.tool,
            toolCall=prepared.toolCall,
            params=prepared.params,
            result=result,
            assistantMessage=prepared.assistantMessage,
            agentContext=prepared.agentContext,
            signal=signal,
        )
        after_result = await _maybe_await(loop.afterToolCall(after_context))
        if isinstance(after_result, AfterToolCallReplace):
            result = _normalize_tool_result(after_result.result, prepared.toolCall)
        elif after_result is None or isinstance(after_result, AfterToolCallPass):
            pass
        else:
            return await emitToolCallOutcome(
                prepared.toolCall,
                result,
                state,
                queue,
                error=AgentError(kind="runtime_error", message="Unsupported afterToolCall result"),
            )
    return await emitToolCallOutcome(prepared.toolCall, result, state, queue)


async def executeToolCallsSequential(
    tool_calls: list[ToolCall],
    assistant_message: AssistantMessage | None,
    state: AgentState,
    loop: AgentLoopConfig,
    queue: asyncio.Queue[AgentEvent],
    signal: AbortSignal,
) -> list[ToolResultMessage]:
    """按顺序执行一组工具调用。"""

    results: list[ToolResultMessage] = []
    for tool_call in tool_calls:
        await _emit_event(
            queue,
            event_type="tool_execution_start",
            state=state,
            payload={"toolCall": tool_call},
        )
        prepared = await prepareToolCall(tool_call, assistant_message, state, loop, signal)
        if isinstance(prepared, PreparedToolCallError):
            if prepared.shortcutResult is not None:
                results.append(await emitToolCallOutcome(tool_call, prepared.shortcutResult, state, queue))
            else:
                results.append(
                    await emitToolCallOutcome(
                        tool_call,
                        _tool_error_result(tool_call, prepared.error or "Unknown tool error"),
                        state,
                        queue,
                        error=prepared.error or AgentError(kind="tool_error", message="Unknown tool error"),
                    )
                )
            continue

        try:
            executed = await executePreparedToolCall(prepared, state, queue, signal)
            results.append(await finalizeExecutedToolCall(prepared, executed, state, loop, queue, signal))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            results.append(
                await emitToolCallOutcome(
                    tool_call,
                    _tool_error_result(tool_call, AgentError(kind="tool_error", message=str(exc))),
                    state,
                    queue,
                    error=AgentError(kind="tool_error", message=str(exc)),
                )
            )
    return results


async def executeToolCallsParallel(
    tool_calls: list[ToolCall],
    assistant_message: AssistantMessage | None,
    state: AgentState,
    loop: AgentLoopConfig,
    queue: asyncio.Queue[AgentEvent],
    signal: AbortSignal,
) -> list[ToolResultMessage]:
    """并发执行一组工具调用，并按原始顺序收集结果。"""

    immediate_results: dict[str, ToolResultMessage] = {}
    prepared_calls: list[PreparedToolCall] = []
    for tool_call in tool_calls:
        await _emit_event(queue, event_type="tool_execution_start", state=state, payload={"toolCall": tool_call})
        prepared = await prepareToolCall(tool_call, assistant_message, state, loop, signal)
        if isinstance(prepared, PreparedToolCallError):
            if prepared.shortcutResult is not None:
                immediate_results[tool_call.id] = await emitToolCallOutcome(tool_call, prepared.shortcutResult, state, queue)
            else:
                immediate_results[tool_call.id] = await emitToolCallOutcome(
                    tool_call,
                    _tool_error_result(tool_call, prepared.error or "Unknown tool error"),
                    state,
                    queue,
                    error=prepared.error or AgentError(kind="tool_error", message="Unknown tool error"),
                )
            continue
        prepared_calls.append(prepared)

    async def _run_prepared(prepared: PreparedToolCall) -> ToolResultMessage:
        try:
            executed = await executePreparedToolCall(prepared, state, queue, signal)
            return await finalizeExecutedToolCall(prepared, executed, state, loop, queue, signal)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return await emitToolCallOutcome(
                prepared.toolCall,
                _tool_error_result(prepared.toolCall, AgentError(kind="tool_error", message=str(exc))),
                state,
                queue,
                error=AgentError(kind="tool_error", message=str(exc)),
            )

    executed_results = await asyncio.gather(*(_run_prepared(prepared) for prepared in prepared_calls))
    finalized_results = {prepared.toolCall.id: result for prepared, result in zip(prepared_calls, executed_results, strict=False)}
    ordered_results: list[ToolResultMessage] = []
    for tool_call in tool_calls:
        ordered_results.append(immediate_results.get(tool_call.id) or finalized_results[tool_call.id])
    ordered_ids = {tool_call.id for tool_call in tool_calls}
    state.messages = [
        message
        for message in state.messages
        if not (isinstance(message, ToolResultMessage) and message.toolCallId in ordered_ids)
    ]
    for result in ordered_results:
        state.messages.append(replace(result))
    return ordered_results


async def executeToolCalls(
    assistant_message: AssistantMessage,
    state: AgentState,
    loop: AgentLoopConfig,
    queue: asyncio.Queue[AgentEvent],
    signal: AbortSignal,
) -> list[ToolResultMessage]:
    """提取 assistant 的全部 tool calls，并按配置执行。"""

    tool_calls = list(assistant_message.toolCalls)
    if not tool_calls:
        return []
    if loop.toolExecutionMode == "parallel":
        return await executeToolCallsParallel(tool_calls, assistant_message, state, loop, queue, signal)
    return await executeToolCallsSequential(tool_calls, assistant_message, state, loop, queue, signal)


async def runLoop(
    state: AgentState,
    loop: AgentLoopConfig,
    queue: asyncio.Queue[AgentEvent],
    signal: AbortSignal,
    *,
    first_turn_started: bool,
) -> None:
    """执行 Agent 主循环。"""

    pending_messages = await _resolve_control_messages(loop.get_steering_messages, state, signal)
    first_inner_turn = True

    while True:
        has_more_tool_calls = True

        while has_more_tool_calls or first_inner_turn or pending_messages:
            signal.throw_if_aborted()
            if not first_turn_started or not first_inner_turn:
                state.runtime_flags.turnIndex += 1
                await _emit_event(queue, event_type="turn_start", state=state, payload={"turnIndex": state.runtime_flags.turnIndex})
            first_turn_started = False
            first_inner_turn = False

            if pending_messages:
                for pending_message in pending_messages:
                    state.messages.append(_copy_payload_value(pending_message))
                    await _emit_event(queue, event_type="message_start", state=state, payload={"message": pending_message})
                    await _emit_event(queue, event_type="message_end", state=state, payload={"message": pending_message})
                pending_messages = []

            assistant_message = await streamAssistantResponse(state, loop, queue, signal)
            if assistant_message is None:
                await _emit_event(queue, event_type="agent_end", state=state, payload={"error": state.error})
                return

            has_more_tool_calls = bool(assistant_message.toolCalls)
            if has_more_tool_calls:
                await executeToolCalls(assistant_message, state, loop, queue, signal)

            await _emit_event(queue, event_type="turn_end", state=state, payload={"turnIndex": state.runtime_flags.turnIndex})
            state.stream_message = None
            pending_messages = [ _copy_payload_value(message) for message in await _resolve_control_messages(loop.get_steering_messages, state, signal)]
            if pending_messages:
                continue
            if not has_more_tool_calls:
                break

        follow_up_messages = await _resolve_control_messages(loop.get_follow_up_messages, state, signal)
        if follow_up_messages:
            pending_messages = [_copy_payload_value(message) for message in follow_up_messages]
            first_inner_turn = False
            continue
        break

    await _emit_event(queue, event_type="agent_end", state=state, payload={"error": state.error})


async def runAgentLoop(
    state: AgentState,
    loop: AgentLoopConfig,
    queue: asyncio.Queue[AgentEvent],
    signal: AbortSignal,
    *,
    new_messages: list[Any],
) -> None:
    """后台执行 Agent 循环，并把事件写入给定队列。"""

    try:
        recorded_messages = [_copy_payload_value(message) for message in new_messages]
        state.runtime_flags.isRunning = True
        state.runtime_flags.turnIndex = 1
        await _emit_event(queue, event_type="agent_start", state=state, payload={"signal": signal})
        await _emit_event(queue, event_type="turn_start", state=state, payload={"turnIndex": state.runtime_flags.turnIndex})
        for message in recorded_messages:
            state.messages.append(_copy_payload_value(message))
            await _emit_event(queue, event_type="message_start", state=state, payload={"message": message})
            await _emit_event(queue, event_type="message_end", state=state, payload={"message": message})
        await runLoop(state, loop, queue, signal, first_turn_started=True)
    except asyncio.CancelledError:
        if getattr(loop, "traceRecorder", None) is not None:
            loop.traceRecorder.record_abort(state.runtime_flags.turnIndex, signal.reason or "agent loop cancelled")
        signal.abort("agent loop cancelled")
        state.runtime_flags.isCancelled = True
        state.error = AgentError(kind="aborted", message="agent loop cancelled")
        await _emit_event(queue, event_type="agent_end", state=state, payload={"error": state.error})
    except Exception as exc:
        if getattr(loop, "traceRecorder", None) is not None:
            loop.traceRecorder.record_abort(state.runtime_flags.turnIndex, str(exc))
        state.error = AgentError(kind="runtime_error", message=str(exc))
        await _emit_event(queue, event_type="agent_end", state=state, payload={"error": state.error})
    finally:
        state.runtime_flags.isRunning = False
        state.runtime_flags.isStreaming = False
        state.stream_message = None
        state.pending_tool_calls = []


def _should_stop_agent_event(event: AgentEvent) -> bool:
    """判断一条 agent 事件是否应终止会话消费。"""

    return event.type == "agent_end"


def _createAgentLoopSession(
    state: AgentState,
    loop: AgentLoopConfig,
    *,
    new_messages: list[Any],
    signal: AbortSignal | None = None,
) -> StreamSession[AgentEvent]:
    """创建 Agent 事件流会话，并在后台启动真正的循环任务。"""

    resolved_signal = signal or AbortSignal()
    queue: asyncio.Queue[AgentEvent] = asyncio.Queue(maxsize=64)
    producer_task = asyncio.create_task(runAgentLoop(state, loop, queue, resolved_signal, new_messages=new_messages))
    model = ensure_model(loop.model or state.model)
    return StreamSession(
        model=model,
        queue=queue,
        producer_task=producer_task,
        should_stop=_should_stop_agent_event,
    )


async def agentLoop(
    loop: AgentLoopConfig,
    *,
    initialMessages: list[Any] | None = None,
    signal: AbortSignal | None = None,
) -> StreamSession[AgentEvent]:
    """启动新对话，并立即返回可订阅的事件流会话。"""

    if loop.model is None:
        raise ValueError("AgentLoopConfig.model is required")
    state = AgentState(
        systemPrompt=loop.systemPrompt,
        model=ensure_model(loop.model),
        thinking=loop.thinking,
        tools=_copy_tools(loop.tools),
        messages=[],
        stream_message=None,
        pending_tool_calls=[],
        error=None,
        runtime_flags=AgentRuntimeFlags(),
    )
    return _createAgentLoopSession(state, loop, new_messages=_normalize_messages(initialMessages), signal=signal)


async def agentLoopContinue(
    state: AgentState,
    *,
    loop: AgentLoopConfig | None = None,
    signal: AbortSignal | None = None,
) -> StreamSession[AgentEvent]:
    """从当前上下文继续运行，不添加新消息。"""

    if not state.messages:
        raise ValueError("agentLoopContinue requires existing history")
    if isinstance(state.messages[-1], AssistantMessage):
        raise ValueError("agentLoopContinue cannot continue from an assistant message")

    resolved_loop = loop or AgentLoopConfig(
        systemPrompt=state.systemPrompt,
        model=state.model,
        thinking=state.thinking,
        tools=_copy_tools(state.tools),
    )
    if resolved_loop.model is None:
        raise ValueError("AgentLoopConfig.model is required")
    state.systemPrompt = resolved_loop.systemPrompt
    state.model = ensure_model(resolved_loop.model)
    state.thinking = resolved_loop.thinking
    state.tools = _copy_tools(resolved_loop.tools)
    state.error = None
    state.runtime_flags.isCancelled = False
    return _createAgentLoopSession(state, resolved_loop, new_messages=[], signal=signal)
