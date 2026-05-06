from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from ai.session import StreamSession
from ai.types import AssistantMessage, ConversationMessage, ensure_message, ensure_model

from .agent_loop import _createAgentLoopSession, agentLoop, agentLoopContinue
from .types import AbortSignal, AgentError, AgentEvent, AgentLoopConfig, AgentRuntimeFlags, AgentState


class AgentEventListener(Protocol):
    """定义高层 Agent 事件监听器的统一签名。"""

    def __call__(self, event: AgentEvent) -> None | Awaitable[None]:
        """消费一条已经过 Agent 状态同步的事件。"""


@dataclass(slots=True)
class AgentOptions:
    """定义创建高层 Agent 时使用的配置选项。"""

    loop: AgentLoopConfig  # 底层 Agent 循环配置。
    initialState: AgentState | None = None  # 可选的初始状态。
    listeners: list[AgentEventListener] = field(default_factory=list)  # 初始事件监听器。
    pendingMessages: list[Any] = field(default_factory=list)  # 初始待发送普通消息。
    steeringMessages: list[Any] = field(default_factory=list)  # 初始 steering 消息。
    followUpMessages: list[Any] = field(default_factory=list)  # 初始 follow-up 消息。
    autoCopyState: bool = True  # 是否在注入初始状态时复制一份。


def _copy_message(message: Any) -> Any:
    try:
        return replace(message)
    except TypeError:
        return message


def _copy_state(state: AgentState) -> AgentState:
    """复制一个 AgentState，避免高层与外部共享可变引用。"""

    return AgentState(
        systemPrompt=state.systemPrompt,
        model=state.model,
        thinking=state.thinking,
        tools=[replace(tool) for tool in state.tools],
        messages=[_copy_message(message) for message in state.messages],
        stream_message=replace(state.stream_message) if state.stream_message is not None else None,
        pending_tool_calls=[replace(tool_call) for tool_call in state.pending_tool_calls],
        error=replace(state.error) if state.error is not None else None,
        runtime_flags=replace(state.runtime_flags),
    )


def _build_initial_state(loop: AgentLoopConfig) -> AgentState:
    """根据 loop 配置构造一个新的初始状态。"""

    if loop.model is None:
        raise ValueError("AgentLoopConfig.model is required")
    return AgentState(
        systemPrompt=loop.systemPrompt,
        model=ensure_model(loop.model),
        thinking=loop.thinking,
        tools=[replace(tool) for tool in loop.tools],
        messages=[],
        stream_message=None,
        pending_tool_calls=[],
        error=None,
        runtime_flags=AgentRuntimeFlags(),
    )


class Agent:
    """面向高层业务的 Agent 运行时封装。"""

    def __init__(self, options: AgentOptions | AgentLoopConfig) -> None:
        """根据 AgentOptions 或 AgentLoopConfig 初始化高层 Agent。"""

        resolved_options = options if isinstance(options, AgentOptions) else AgentOptions(loop=options)
        self._options = resolved_options  # 完整的 Agent 初始化选项。
        self._loop = resolved_options.loop  # 当前绑定的底层循环配置。
        self._listeners: list[AgentEventListener] = list(resolved_options.listeners)  # 事件监听器列表。
        self._pendingMessages: list[Any] = [ensure_message(message) if isinstance(message, dict) and "role" in message else message for message in resolved_options.pendingMessages]  # 普通待处理消息队列。
        self._steeringMessages: list[Any] = [ensure_message(message) if isinstance(message, dict) and "role" in message else message for message in resolved_options.steeringMessages]  # steering 消息队列。
        self._followUpMessages: list[Any] = [ensure_message(message) if isinstance(message, dict) and "role" in message else message for message in resolved_options.followUpMessages]  # follow-up 消息队列。
        self._currentSession: StreamSession[AgentEvent] | None = None  # 当前对外暴露的事件流会话。
        self._currentTask: asyncio.Task[None] | None = None  # 负责桥接底层循环的后台任务。
        self._abortSignal: AbortSignal | None = None  # 当前运行使用的中断信号。
        self._cancelRequested = False  # 是否已经请求取消。
        self._isRunning = False  # 当前 Agent 是否正在运行。
        if resolved_options.initialState is not None:
            self.state = _copy_state(resolved_options.initialState) if resolved_options.autoCopyState else resolved_options.initialState
        else:
            self.state = _build_initial_state(self._loop)

    @property
    def isRunning(self) -> bool:
        """返回当前 Agent 是否处于运行中。"""

        return self._isRunning

    @property
    def lastMessage(self) -> AssistantMessage | None:
        """返回历史中最近一条 assistant 消息。"""

        for message in reversed(self.state.messages):
            if isinstance(message, AssistantMessage):
                return message
        return None

    @property
    def pendingMessages(self) -> list[Any]:
        """返回当前普通待处理消息队列的副本。"""

        return [_copy_message(message) for message in self._pendingMessages]

    @property
    def steeringMessages(self) -> list[Any]:
        """返回当前 steering 消息队列的副本。"""

        return [_copy_message(message) for message in self._steeringMessages]

    @property
    def followUpMessages(self) -> list[Any]:
        """返回当前 follow-up 消息队列的副本。"""

        return [_copy_message(message) for message in self._followUpMessages]

    @property
    def listeners(self) -> list[AgentEventListener]:
        """返回当前监听器列表的副本。"""

        return list(self._listeners)

    @property
    def currentSession(self) -> StreamSession[AgentEvent] | None:
        """返回当前对外暴露的事件流会话。"""

        return self._currentSession

    @property
    def currentTask(self) -> asyncio.Task[None] | None:
        """返回当前桥接底层循环的后台任务。"""

        return self._currentTask

    @property
    def abortSignal(self) -> AbortSignal | None:
        """返回当前运行使用的中断信号。"""

        return self._abortSignal

    def getState(self) -> AgentState:
        """返回当前 AgentState 的快照。"""

        return _copy_state(self.state)

    def setState(self, state: AgentState) -> None:
        """直接替换整个 AgentState。"""

        if self._isRunning:
            raise RuntimeError("Cannot set state while Agent is running")
        self.state = _copy_state(state)

    def updateState(self, **kwargs: Any) -> None:
        """局部更新 AgentState 字段。"""

        restricted = {"stream_message", "pending_tool_calls", "runtime_flags"}
        if self._isRunning and restricted.intersection(kwargs):
            raise RuntimeError("Cannot update runtime-only state fields while Agent is running")
        for key, value in kwargs.items():
            if key == "history":
                self.state.messages = list(value)
            elif key == "currentMessage":
                self.state.stream_message = value
            elif key == "runningToolCall":
                self.state.pending_tool_calls = [value] if value is not None else []
            elif key == "isStreaming":
                self.state.runtime_flags.isStreaming = bool(value)
            else:
                setattr(self.state, key, value)

    def setThinking(self, thinking) -> None:
        """设置高层 Agent 的思考级别。"""

        self.state.thinking = thinking
        self._loop.thinking = thinking

    def setSystemPrompt(self, prompt: str | None) -> None:
        """设置系统提示词，并同步到 loop 配置。"""

        self.state.systemPrompt = prompt
        self._loop.systemPrompt = prompt

    def setModel(self, model) -> None:
        """设置当前模型，并同步到 loop 配置。"""

        self.state.model = ensure_model(model)
        self._loop.model = self.state.model

    def setTools(self, tools) -> None:
        """设置可用工具列表，并同步到 loop 配置。"""

        copied_tools = [replace(tool) for tool in tools]
        self.state.tools = copied_tools
        self._loop.tools = [replace(tool) for tool in copied_tools]

    def setTraceRecorder(self, recorder) -> None:
        """为后续运行挂接或清空 trace 记录器。"""

        self._loop.traceRecorder = recorder

    def subscribe(self, listener: AgentEventListener) -> None:
        """注册一个新的事件监听器。"""

        self._listeners.append(listener)

    def unsubscribe(self, listener: AgentEventListener) -> None:
        """移除一个已注册的事件监听器。"""

        self._listeners = [item for item in self._listeners if item is not listener]

    def clearListeners(self) -> None:
        """清空全部事件监听器。"""

        self._listeners.clear()

    def enqueue(self, message: Any | list[Any]) -> None:
        """向普通待处理消息队列追加一条或多条消息。"""

        items = message if isinstance(message, list) else [message]
        self._pendingMessages.extend(ensure_message(item) if isinstance(item, dict) and "role" in item else item for item in items)

    def enqueueSteering(self, message: Any | list[Any]) -> None:
        """向 steering 队列追加一条或多条中途插话消息。"""

        items = message if isinstance(message, list) else [message]
        self._steeringMessages.extend(ensure_message(item) if isinstance(item, dict) and "role" in item else item for item in items)

    def enqueueFollowUp(self, message: Any | list[Any]) -> None:
        """向 follow-up 队列追加一条或多条任务后处理消息。"""

        items = message if isinstance(message, list) else [message]
        self._followUpMessages.extend(ensure_message(item) if isinstance(item, dict) and "role" in item else item for item in items)

    async def send(self, message: ConversationMessage | dict[str, Any] | list[Any]) -> None:
        """兼容旧接口，等价于 `enqueue()`。"""

        self.enqueue(message)

    def dequeueAll(self) -> list[Any]:
        """取出并清空全部普通待处理消息。"""

        messages = [_copy_message(message) for message in self._pendingMessages]
        self._pendingMessages.clear()
        return messages

    def dequeueSteeringAll(self) -> list[Any]:
        """取出并清空全部 steering 消息。"""

        messages = [_copy_message(message) for message in self._steeringMessages]
        self._steeringMessages.clear()
        return messages

    def dequeueFollowUpAll(self) -> list[Any]:
        """取出并清空全部 follow-up 消息。"""

        messages = [_copy_message(message) for message in self._followUpMessages]
        self._followUpMessages.clear()
        return messages

    def clearQueue(self) -> None:
        """清空普通待处理消息队列。"""

        self._pendingMessages.clear()

    def clearSteeringQueue(self) -> None:
        """清空 steering 消息队列。"""

        self._steeringMessages.clear()

    def clearFollowUpQueue(self) -> None:
        """清空 follow-up 消息队列。"""

        self._followUpMessages.clear()

    def queueSize(self) -> int:
        """返回当前普通待处理消息数量。"""

        return len(self._pendingMessages)

    def steeringQueueSize(self) -> int:
        """返回当前 steering 消息数量。"""

        return len(self._steeringMessages)

    def followUpQueueSize(self) -> int:
        """返回当前 follow-up 消息数量。"""

        return len(self._followUpMessages)

    def cancel(self, reason: str | None = None) -> None:
        """取消当前运行中的循环，但保留历史消息。"""

        self._cancelRequested = True
        if self._abortSignal is not None:
            self._abortSignal.abort(reason or "cancelled by agent")
        if self._currentSession is not None:
            self._currentSession.producer_task.cancel()
        if self._currentTask is not None:
            self._currentTask.cancel()

    async def wait(self) -> None:
        """等待当前运行中的会话结束。"""

        if self._currentSession is not None:
            await self._currentSession.wait_closed()
        if self._currentTask is None or self._currentTask.done():
            self._cleanup_after_run()

    def reset(self) -> None:
        """重置状态、队列和运行控制。"""

        if self._isRunning:
            raise RuntimeError("Cannot reset an Agent while it is running")
        self._pendingMessages.clear()
        self._steeringMessages.clear()
        self._followUpMessages.clear()
        self._currentSession = None
        self._currentTask = None
        self._abortSignal = None
        self._cancelRequested = False
        self.state = _build_initial_state(self._loop)

    async def prompt(self, message: Any | list[Any]) -> StreamSession[AgentEvent]:
        """追加消息并立即启动一轮循环。"""

        self.enqueue(message)
        return await self.run()

    async def continueConversation(self) -> StreamSession[AgentEvent]:
        """从当前上下文继续对话，不添加新消息。"""

        if self.state.messages and isinstance(self.state.messages[-1], AssistantMessage):
            queued_steering = self.dequeueSteeringAll()
            if queued_steering:
                return await self._runLoopSession(newMessages=queued_steering)
            queued_follow_up = self.dequeueFollowUpAll()
            if queued_follow_up:
                return await self._runLoopSession(newMessages=queued_follow_up)
            raise ValueError("Agent.continueConversation cannot continue from an assistant message without queued messages")
        return await self._runLoopSession(continueOnly=True)

    async def resume(self) -> StreamSession[AgentEvent]:
        """作为 `continueConversation()` 的兼容别名。"""

        return await self.continueConversation()

    async def run(self) -> StreamSession[AgentEvent]:
        """根据当前队列或历史状态启动一次高层运行。"""

        has_pending_messages = bool(self._pendingMessages)
        return await self._runLoopSession(
            newMessages=self.dequeueAll() if has_pending_messages else None,
            continueOnly=not has_pending_messages,
        )

    async def _emit_to_listeners(self, event: AgentEvent) -> None:
        """按注册顺序把事件分发给全部监听器。"""

        for listener in list(self._listeners):
            try:
                result = listener(event)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                self.state.error = AgentError(kind="runtime_error", message=str(exc))

    async def emit(self, event: AgentEvent, queue: asyncio.Queue[AgentEvent]) -> None:
        """把已处理事件分发给监听器并写入高层事件流。"""

        await self._emit_to_listeners(event)
        await queue.put(event)

    def _handleLoopEvent(self, event: AgentEvent) -> AgentEvent:
        """处理底层循环事件，更新内部状态并返回事件本身。"""

        if event.state is not None:
            self.state = _copy_state(event.state)

        if event.type == "agent_start":
            self.state.error = None
        elif event.type == "message_start":
            if isinstance(event.message, AssistantMessage):
                self.state.stream_message = _copy_message(event.message)
        elif event.type == "message_end":
            if isinstance(event.message, AssistantMessage):
                self.state.stream_message = None
        elif event.type == "tool_execution_start":
            if event.toolCall is not None:
                self.state.pending_tool_calls = [_copy_message(event.toolCall)]
        elif event.type == "tool_execution_end":
            self.state.pending_tool_calls = []
            if event.payload.get("error") is not None:
                error = event.payload["error"]
                self.state.error = error if isinstance(error, AgentError) else AgentError(kind="tool_error", message=str(error))
        elif event.type == "agent_end":
            self.state.runtime_flags.isStreaming = False
            self.state.runtime_flags.isRunning = False

        return event

    def _buildLoopConfig(self) -> AgentLoopConfig:
        """基于当前高层状态构建一次底层循环配置。"""

        async def _get_steering_messages(state: AgentState, signal: AbortSignal):
            signal.throw_if_aborted()
            queued_messages = self.dequeueSteeringAll()
            hook = self._loop.get_steering_messages
            if hook is None:
                return queued_messages
            try:
                result = hook(state, signal)
            except TypeError:
                result = hook(state)
            if inspect.isawaitable(result):
                result = await result
            return queued_messages + list(result or [])

        async def _get_follow_up_messages(state: AgentState, signal: AbortSignal):
            signal.throw_if_aborted()
            queued_messages = self.dequeueFollowUpAll()
            hook = self._loop.get_follow_up_messages
            if hook is None:
                return queued_messages
            try:
                result = hook(state, signal)
            except TypeError:
                result = hook(state)
            if inspect.isawaitable(result):
                result = await result
            return queued_messages + list(result or [])

        return AgentLoopConfig(
            systemPrompt=self.state.systemPrompt,
            model=self.state.model,
            thinking=self.state.thinking,
            tools=[replace(tool) for tool in self.state.tools],
            stream=self._loop.stream,
            convert_to_llm=self._loop.convert_to_llm,
            transform_context=self._loop.transform_context,
            get_steering_messages=_get_steering_messages,
            get_follow_up_messages=_get_follow_up_messages,
            toolExecutionMode=self._loop.toolExecutionMode,
            beforeToolCall=self._loop.beforeToolCall,
            afterToolCall=self._loop.afterToolCall,
            retryPolicy=self._loop.retryPolicy,
            registry=self._loop.registry,
            traceRecorder=self._loop.traceRecorder,
        )

    def _cleanup_after_run(self) -> None:
        """清理一次运行结束后的高层控制状态。"""

        self._isRunning = False
        self._cancelRequested = False
        self._currentSession = None
        self._currentTask = None
        self._abortSignal = None
        self.state.runtime_flags.isStreaming = False
        self.state.runtime_flags.isRunning = False
        self.state.stream_message = None
        self.state.pending_tool_calls = []

    async def _runLoopSession(
        self,
        *,
        newMessages: list[Any] | None = None,
        continueOnly: bool = False,
    ) -> StreamSession[AgentEvent]:
        """创建高层桥接会话，并在后台同步底层循环事件。"""

        if self._isRunning:
            raise RuntimeError("Agent is already running")

        if continueOnly and not self.state.messages:
            raise ValueError("Agent.continueConversation requires existing history")

        new_messages = [_copy_message(message) for message in (newMessages or [])]
        if not self.state.messages and not new_messages:
            raise ValueError("Agent.run requires pending messages when history is empty")

        self._cancelRequested = False
        self._isRunning = True
        self.state.error = None
        self.state.runtime_flags.isRunning = True
        if self._loop.thinking is not None:
            self.state.thinking = self._loop.thinking

        loop_config = self._buildLoopConfig()
        signal = AbortSignal()
        self._abortSignal = signal
        if continueOnly and not new_messages:
            lower_session = await agentLoopContinue(self.state, loop=loop_config, signal=signal)
        elif self.state.messages and new_messages:
            lower_session = _createAgentLoopSession(self.state, loop_config, new_messages=new_messages, signal=signal)
        else:
            lower_session = await agentLoop(loop_config, initialMessages=new_messages, signal=signal)

        queue: asyncio.Queue[AgentEvent] = asyncio.Queue(maxsize=64)
        model = ensure_model(loop_config.model)

        async def _bridge_events() -> None:
            try:
                async for event in lower_session.consume():
                    processed_event = self._handleLoopEvent(event)
                    await self.emit(processed_event, queue)
            except asyncio.CancelledError:
                self._cancelRequested = True
                signal.abort("bridge cancelled")
                await lower_session.close()
                raise
            except Exception as exc:
                self.state.error = AgentError(kind="runtime_error", message=str(exc))
                await self.emit(
                    AgentEvent(type="agent_end", state=_copy_state(self.state), payload={"error": self.state.error}),
                    queue,
                )

        bridge_task = asyncio.create_task(_bridge_events())

        async def _tracked_bridge() -> None:
            try:
                await bridge_task
                while not queue.empty() and not self._cancelRequested:
                    await asyncio.sleep(0)
            except asyncio.CancelledError:
                bridge_task.cancel()
                try:
                    await bridge_task
                except asyncio.CancelledError:
                    pass
                raise
            finally:
                self._cleanup_after_run()

        tracked_task = asyncio.create_task(_tracked_bridge())
        session = StreamSession(
            model=model,
            queue=queue,
            producer_task=tracked_task,
            should_stop=lambda event: event.type == "agent_end",
        )
        self._currentSession = session
        self._currentTask = tracked_task
        return session
