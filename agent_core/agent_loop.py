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
    ConversationMessage,
    Model,
    StreamEvent,
    Tool,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    ensure_message,
    ensure_model,
)
from ai.utils.schema_validation import SchemaValidationError, validate_tool_arguments

from .types import (
    AfterToolCallContext,
    AfterToolCallPass,
    AfterToolCallReplace,
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentState,
    AgentTool,
    BeforeToolCallAllow,
    BeforeToolCallContext,
    BeforeToolCallError,
    BeforeToolCallSkip,
)


@dataclass(slots=True)
class PreparedToolCall:
    """表示已经通过可用性与参数校验的工具调用。"""

    toolCall: ToolCall
    tool: AgentTool
    arguments: str
    parsedArguments: Any
    agentContext: AgentContext


@dataclass(slots=True)
class PreparedToolCallError:
    """表示工具调用在预处理阶段失败或被短路。"""

    toolCall: ToolCall
    error: str | None = None
    shortcutResult: Any | None = None


async def _maybe_await(value: Any) -> Any:
    """在值为 awaitable 时等待完成，否则原样返回。"""

    if inspect.isawaitable(value):
        return await value
    return value


def _copy_tools(tools: list[AgentTool]) -> list[AgentTool]:
    """复制工具定义，避免状态与配置共享可变对象。"""

    return [replace(tool) for tool in tools]


def _copy_history(history: list[ConversationMessage]) -> list[ConversationMessage]:
    """复制消息历史，避免事件快照共享引用。"""

    return [replace(message) for message in history]


def _snapshot_state(state: AgentState) -> AgentState:
    """生成用于事件流输出的状态快照。"""

    return AgentState(
        systemPrompt=state.systemPrompt,
        model=state.model,
        thinking=state.thinking,
        tools=_copy_tools(state.tools),
        history=_copy_history(state.history),
        isStreaming=state.isStreaming,
        currentMessage=replace(state.currentMessage) if state.currentMessage is not None else None,
        runningToolCall=replace(state.runningToolCall) if state.runningToolCall is not None else None,
        error=state.error,
    )


def _normalize_messages(
    messages: list[ConversationMessage | dict[str, Any]] | None,
) -> list[ConversationMessage]:
    """把输入统一归一化为消息列表。"""

    return [ensure_message(message) for message in (messages or [])]


def _find_tool(tools: list[AgentTool], name: str) -> AgentTool | None:
    """根据工具名查找可执行工具。"""

    for tool in tools:
        if tool.name == name:
            return tool
    return None


def _to_agent_context(state: AgentState) -> AgentContext:
    """从当前状态构造一次 AgentContext。"""

    return AgentContext(
        systemPrompt=state.systemPrompt,
        history=_copy_history(state.history),
        tools=[
            Tool(
                name=tool.name,
                description=tool.description,
                inputSchema=dict(tool.inputSchema),
                metadata=dict(tool.metadata),
            )
            for tool in state.tools
        ],
    )


def _normalize_tool_result(result: Any, tool_call: ToolCall) -> ToolResultMessage:
    """把任意工具执行产出统一转成 ToolResultMessage。"""

    if isinstance(result, ToolResultMessage):
        return ToolResultMessage(
            toolCallId=result.toolCallId or tool_call.id,
            toolName=result.toolName or tool_call.name,
            content=result.content,
            metadata=dict(result.metadata),
        )
    if isinstance(result, dict):
        metadata = dict(result.get("metadata", {}))
        content = result.get("content")
        if content is None:
            content = json.dumps(result, ensure_ascii=False)
        return ToolResultMessage(
            toolCallId=str(result.get("toolCallId", tool_call.id)),
            toolName=str(result.get("toolName", tool_call.name)),
            content=str(content),
            metadata=metadata,
        )
    return ToolResultMessage(
        toolCallId=tool_call.id,
        toolName=tool_call.name,
        content=str(result),
        metadata={},
    )


def _tool_error_result(tool_call: ToolCall, error: str) -> ToolResultMessage:
    """根据错误文本生成标准工具错误结果。"""

    return ToolResultMessage(
        toolCallId=tool_call.id,
        toolName=tool_call.name,
        content=error,
        metadata={"error": True},
    )


async def _emit_event(
    queue: asyncio.Queue[AgentEvent],
    *,
    event_type: str,
    state: AgentState,
    message: ConversationMessage | AssistantMessage | ToolResultMessage | None = None,
    message_delta: str | None = None,
    tool_call: ToolCall | None = None,
    tool_result: ToolResultMessage | None = None,
    error: str | None = None,
) -> AgentEvent:
    """构造事件并写入队列。"""

    event = AgentEvent(
        type=event_type,
        state=_snapshot_state(state),
        message=replace(message) if isinstance(message, (AssistantMessage, ToolResultMessage)) else message,
        messageDelta=message_delta,
        toolCall=replace(tool_call) if tool_call is not None else None,
        toolResult=replace(tool_result) if tool_result is not None else None,
        error=error,
    )
    await queue.put(event)
    return event


async def _resolve_control_messages(
    fn,
    state: AgentState,
) -> list[ConversationMessage]:
    """获取 steering 或 follow-up 消息并统一归一化。"""

    if fn is None:
        return []
    return _normalize_messages(await _maybe_await(fn(state)))


async def _default_stream(
    model: Model | str,
    context: AgentContext,
    thinking,
    registry,
) -> StreamSession[StreamEvent]:
    """使用默认的 `ai.streamSimple` 打开一次流式会话。"""

    return await streamSimple(
        model,
        {
            "systemPrompt": context.systemPrompt,
            "messages": context.history,
            "tools": context.tools,
        },
        reasoning=thinking,
        registry=registry,
    )


async def _open_stream(state: AgentState, loop: AgentLoopConfig) -> StreamSession[StreamEvent]:
    """根据循环配置获取底层 AI 流式会话。"""

    model = ensure_model(loop.model or state.model)
    context = _to_agent_context(state)
    stream_fn = loop.stream or _default_stream
    return await stream_fn(model, context, loop.thinking or state.thinking, loop.registry)


async def streamAssistantResponse(
    state: AgentState,
    loop: AgentLoopConfig,
    queue: asyncio.Queue[AgentEvent],
) -> AssistantMessage | None:
    """获取 AI 的单轮回复，并把流式过程映射到事件流中。"""

    try:
        session = await _open_stream(state, loop)
    except Exception as exc:
        state.error = str(exc)
        return None

    final_message: AssistantMessage | None = None
    message_started = False
    state.isStreaming = True
    state.currentMessage = AssistantMessage(content="", thinking="", toolCalls=[])

    try:
        async for raw_event in session.consume():
            if raw_event.type == "start" and raw_event.lifecycle == "start":
                if not message_started:
                    message_started = True
                    await _emit_event(queue, event_type="message_start", state=state)
                continue

            if raw_event.type == "done" and raw_event.lifecycle == "done":
                final_message = replace(raw_event.assistantMessage or state.currentMessage or AssistantMessage())
                state.currentMessage = replace(final_message)
                if not message_started:
                    await _emit_event(queue, event_type="message_start", state=state)
                state.history.append(replace(final_message))
                await _emit_event(
                    queue,
                    event_type="message_end",
                    state=state,
                    message=final_message,
                )
                return final_message

            if raw_event.type == "error" and raw_event.lifecycle == "error":
                state.error = raw_event.error or "provider emitted error event"
                return None

            if state.currentMessage is None:
                state.currentMessage = AssistantMessage(content="", thinking="", toolCalls=[])
            if raw_event.itemType == "text" and raw_event.lifecycle == "update":
                state.currentMessage.content += raw_event.text or ""
            elif raw_event.itemType == "thinking" and raw_event.lifecycle == "update":
                state.currentMessage.thinking += raw_event.thinking or ""
            elif raw_event.itemType == "tool_call" and raw_event.lifecycle == "start" and raw_event.toolCallId:
                state.currentMessage.toolCalls.append(
                    ToolCall(id=raw_event.toolCallId, name=raw_event.toolName or "", arguments="")
                )
            elif raw_event.itemType == "tool_call" and raw_event.lifecycle == "update" and raw_event.toolCallId:
                for tool_call in state.currentMessage.toolCalls:
                    if tool_call.id == raw_event.toolCallId:
                        tool_call.arguments += raw_event.argumentsDelta or ""
                        break
            elif raw_event.itemType == "tool_call" and raw_event.lifecycle == "done" and raw_event.toolCallId:
                for tool_call in state.currentMessage.toolCalls:
                    if tool_call.id == raw_event.toolCallId and raw_event.arguments is not None:
                        tool_call.arguments = raw_event.arguments
                        break

            if not message_started:
                message_started = True
                await _emit_event(queue, event_type="message_start", state=state)
            await _emit_event(
                queue,
                event_type="message_update",
                state=state,
                message_delta=raw_event.text or raw_event.thinking or raw_event.argumentsDelta,
            )
    finally:
        state.isStreaming = False
        await session.close()

    if final_message is None and state.currentMessage is not None and message_started:
        await _emit_event(
            queue,
            event_type="message_end",
            state=state,
            message=state.currentMessage,
        )
        final_message = replace(state.currentMessage)
        state.history.append(replace(final_message))
        state.currentMessage = None
    return final_message


def _parse_tool_arguments(tool_call: ToolCall) -> Any:
    """解析工具参数字符串。"""

    if not tool_call.arguments:
        return {}
    return json.loads(tool_call.arguments)


async def prepareToolCall(
    tool_call: ToolCall,
    state: AgentState,
    loop: AgentLoopConfig,
) -> PreparedToolCall | PreparedToolCallError:
    """准备工具调用，检查工具、参数与 beforeToolCall。"""

    tool = _find_tool(state.tools, tool_call.name)
    if tool is None or tool.execute is None:
        return PreparedToolCallError(toolCall=tool_call, error=f"Tool '{tool_call.name}' is not available")

    try:
        parsed_arguments = _parse_tool_arguments(tool_call)
    except json.JSONDecodeError as exc:
        return PreparedToolCallError(toolCall=tool_call, error=f"Invalid tool arguments: {exc.msg}")

    try:
        validate_tool_arguments(tool, parsed_arguments)
    except SchemaValidationError as exc:
        return PreparedToolCallError(toolCall=tool_call, error=str(exc))

    context = BeforeToolCallContext(
        state=state,
        tool=tool,
        toolCall=tool_call,
        arguments=tool_call.arguments,
        agentContext=_to_agent_context(state),
    )
    before_result = await _maybe_await(loop.beforeToolCall(context)) if loop.beforeToolCall else None
    if before_result is None or isinstance(before_result, BeforeToolCallAllow):
        return PreparedToolCall(
            toolCall=tool_call,
            tool=tool,
            arguments=tool_call.arguments,
            parsedArguments=parsed_arguments,
            agentContext=context.agentContext,
        )
    if isinstance(before_result, BeforeToolCallSkip):
        return PreparedToolCallError(toolCall=tool_call, shortcutResult=before_result.result)
    if isinstance(before_result, BeforeToolCallError):
        return PreparedToolCallError(toolCall=tool_call, error=before_result.error)
    return PreparedToolCallError(toolCall=tool_call, error="Unsupported beforeToolCall result")


async def executePreparedToolCall(
    prepared: PreparedToolCall,
    state: AgentState,
    loop: AgentLoopConfig,
    queue: asyncio.Queue[AgentEvent],
) -> Any:
    """执行已经通过准备阶段的工具调用。"""

    state.runningToolCall = replace(prepared.toolCall)
    await _emit_event(
        queue,
        event_type="tool_execution_update",
        state=state,
        tool_call=prepared.toolCall,
    )
    try:
        return await _maybe_await(prepared.tool.execute(prepared.arguments, prepared.agentContext))
    finally:
        state.runningToolCall = None


async def emitToolCallOutcome(
    tool_call: ToolCall,
    result: Any,
    state: AgentState,
    queue: asyncio.Queue[AgentEvent],
    *,
    error: str | None = None,
) -> ToolResultMessage:
    """发出工具执行结果，并把其写入上下文历史。"""

    tool_message = _normalize_tool_result(result, tool_call)
    if error is not None:
        tool_message.metadata["error"] = True
        tool_message.content = error
        state.error = error

    await _emit_event(
        queue,
        event_type="tool_execution_end",
        state=state,
        tool_call=tool_call,
        tool_result=tool_message,
        error=error,
    )
    state.history.append(replace(tool_message))
    await _emit_event(queue, event_type="message_start", state=state, message=tool_message)
    await _emit_event(queue, event_type="message_end", state=state, message=tool_message)
    return tool_message


async def finalizeExecutedToolCall(
    prepared: PreparedToolCall,
    executed_result: Any,
    state: AgentState,
    loop: AgentLoopConfig,
    queue: asyncio.Queue[AgentEvent],
) -> ToolResultMessage:
    """执行 afterToolCall，并最终落盘工具结果。"""

    result = _normalize_tool_result(executed_result, prepared.toolCall)
    if loop.afterToolCall:
        after_context = AfterToolCallContext(
            state=state,
            tool=prepared.tool,
            toolCall=prepared.toolCall,
            arguments=prepared.arguments,
            result=result,
            agentContext=prepared.agentContext,
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
                error="Unsupported afterToolCall result",
            )
    return await emitToolCallOutcome(prepared.toolCall, result, state, queue)


async def executeToolCallsSequential(
    tool_calls: list[ToolCall],
    state: AgentState,
    loop: AgentLoopConfig,
    queue: asyncio.Queue[AgentEvent],
) -> list[ToolResultMessage]:
    """按顺序执行一组工具调用。"""

    results: list[ToolResultMessage] = []
    for tool_call in tool_calls:
        await _emit_event(queue, event_type="tool_execution_start", state=state, tool_call=tool_call)
        prepared = await prepareToolCall(tool_call, state, loop)
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
                        error=prepared.error or "Unknown tool error",
                    )
                )
            continue

        executed = await executePreparedToolCall(prepared, state, loop, queue)
        results.append(await finalizeExecutedToolCall(prepared, executed, state, loop, queue))
    return results


async def executeToolCallsParallel(
    tool_calls: list[ToolCall],
    state: AgentState,
    loop: AgentLoopConfig,
    queue: asyncio.Queue[AgentEvent],
) -> list[ToolResultMessage]:
    """并发执行一组工具调用，并按原始顺序收集结果。"""

    immediate_results: dict[str, ToolResultMessage] = {}
    prepared_calls: list[PreparedToolCall] = []
    for tool_call in tool_calls:
        await _emit_event(queue, event_type="tool_execution_start", state=state, tool_call=tool_call)
        prepared = await prepareToolCall(tool_call, state, loop)
        if isinstance(prepared, PreparedToolCallError):
            if prepared.shortcutResult is not None:
                immediate_results[tool_call.id] = await emitToolCallOutcome(
                    tool_call,
                    prepared.shortcutResult,
                    state,
                    queue,
                )
            else:
                immediate_results[tool_call.id] = await emitToolCallOutcome(
                    tool_call,
                    _tool_error_result(tool_call, prepared.error or "Unknown tool error"),
                    state,
                    queue,
                    error=prepared.error or "Unknown tool error",
                )
            continue
        prepared_calls.append(prepared)

    executed_results = await asyncio.gather(
        *(executePreparedToolCall(prepared, state, loop, queue) for prepared in prepared_calls)
    )
    finalized_results: dict[str, ToolResultMessage] = {}
    for prepared, executed in zip(prepared_calls, executed_results, strict=False):
        finalized_results[prepared.toolCall.id] = await finalizeExecutedToolCall(
            prepared,
            executed,
            state,
            loop,
            queue,
        )

    ordered_results: list[ToolResultMessage] = []
    for tool_call in tool_calls:
        if tool_call.id in immediate_results:
            ordered_results.append(immediate_results[tool_call.id])
        else:
            ordered_results.append(finalized_results[tool_call.id])
    return ordered_results


async def executeToolCalls(
    assistant_message: AssistantMessage,
    state: AgentState,
    loop: AgentLoopConfig,
    queue: asyncio.Queue[AgentEvent],
) -> list[ToolResultMessage]:
    """提取 assistant 的全部 tool calls，并按配置执行。"""

    tool_calls = list(assistant_message.toolCalls)
    if not tool_calls:
        return []
    if loop.toolExecutionMode == "parallel":
        return await executeToolCallsParallel(tool_calls, state, loop, queue)
    return await executeToolCallsSequential(tool_calls, state, loop, queue)


async def runLoop(
    state: AgentState,
    loop: AgentLoopConfig,
    queue: asyncio.Queue[AgentEvent],
    *,
    first_turn_started: bool,
) -> None:
    """执行 Agent 主循环。"""

    pending_messages = await _resolve_control_messages(loop.steer, state)
    first_inner_turn = True

    while True:
        while first_inner_turn or pending_messages:
            if not first_turn_started or not first_inner_turn:
                await _emit_event(queue, event_type="turn_start", state=state)
            first_turn_started = False
            first_inner_turn = False

            if pending_messages:
                for pending_message in pending_messages:
                    state.history.append(replace(pending_message))
                    await _emit_event(queue, event_type="message_start", state=state, message=pending_message)
                    await _emit_event(queue, event_type="message_end", state=state, message=pending_message)
                pending_messages = []

            assistant_message = await streamAssistantResponse(state, loop, queue)
            if assistant_message is None:
                await _emit_event(queue, event_type="agent_end", state=state, error=state.error)
                return

            if assistant_message.toolCalls:
                await executeToolCalls(assistant_message, state, loop, queue)

            await _emit_event(queue, event_type="turn_end", state=state)
            state.currentMessage = None
            pending_messages = [replace(message) for message in await _resolve_control_messages(loop.steer, state)]
            if pending_messages:
                continue
            break

        follow_up_messages = await _resolve_control_messages(loop.followUp, state)
        if follow_up_messages:
            pending_messages = [replace(message) for message in follow_up_messages]
            first_inner_turn = False
            continue
        break

    await _emit_event(queue, event_type="agent_end", state=state, error=state.error)


async def runAgentLoop(
    state: AgentState,
    loop: AgentLoopConfig,
    queue: asyncio.Queue[AgentEvent],
    *,
    new_messages: list[ConversationMessage],
) -> None:
    """后台执行 Agent 循环，并把事件写入给定队列。"""

    try:
        recorded_messages = [replace(message) for message in new_messages]
        agent_context = _to_agent_context(state)
        _ = agent_context
        await _emit_event(queue, event_type="agent_start", state=state)
        await _emit_event(queue, event_type="turn_start", state=state)
        for message in recorded_messages:
            state.history.append(replace(message))
            if isinstance(message, UserMessage):
                await _emit_event(queue, event_type="message_start", state=state, message=message)
                await _emit_event(queue, event_type="message_end", state=state, message=message)
        await runLoop(state, loop, queue, first_turn_started=True)
    except asyncio.CancelledError:
        state.error = "agent loop cancelled"
        await _emit_event(queue, event_type="agent_end", state=state, error=state.error)
    except Exception as exc:
        state.error = str(exc)
        await _emit_event(queue, event_type="agent_end", state=state, error=state.error)


def _should_stop_agent_event(event: AgentEvent) -> bool:
    """判断一条 agent 事件是否应终止会话消费。"""

    return event.type == "agent_end"


def _createAgentLoopSession(
    state: AgentState,
    loop: AgentLoopConfig,
    *,
    new_messages: list[ConversationMessage],
) -> StreamSession[AgentEvent]:
    """创建 Agent 事件流会话，并在后台启动真正的循环任务。"""

    queue: asyncio.Queue[AgentEvent] = asyncio.Queue(maxsize=64)
    producer_task = asyncio.create_task(runAgentLoop(state, loop, queue, new_messages=new_messages))
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
    initialMessages: list[ConversationMessage | dict[str, Any]] | None = None,
) -> StreamSession[AgentEvent]:
    """启动新对话，并立即返回可订阅的事件流会话。"""

    if loop.model is None:
        raise ValueError("AgentLoopConfig.model is required")
    state = AgentState(
        systemPrompt=loop.systemPrompt,
        model=ensure_model(loop.model),
        thinking=loop.thinking,
        tools=_copy_tools(loop.tools),
        history=[],
        isStreaming=False,
        currentMessage=None,
        runningToolCall=None,
        error=None,
    )
    return _createAgentLoopSession(state, loop, new_messages=_normalize_messages(initialMessages))


async def agentLoopContinue(
    state: AgentState,
    *,
    loop: AgentLoopConfig | None = None,
) -> StreamSession[AgentEvent]:
    """从当前上下文继续运行，不添加新消息。"""

    if not state.history:
        raise ValueError("agentLoopContinue requires existing history")
    if isinstance(state.history[-1], AssistantMessage):
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
    return _createAgentLoopSession(state, resolved_loop, new_messages=[])
