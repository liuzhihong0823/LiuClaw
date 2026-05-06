from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol, TypeAlias

from ai.registry import ProviderRegistry
from ai.session import StreamSession
from ai.types import (
    AssistantMessage,
    Context,
    ConversationMessage,
    Model,
    ReasoningLevel,
    Tool,
    ToolCall,
    ToolResultMessage,
)

AgentEventType: TypeAlias = Literal[
    "agent_start",
    "agent_end",
    "turn_start",
    "turn_end",
    "message_start",
    "message_update",
    "message_end",
    "tool_execution_start",
    "tool_execution_update",
    "tool_execution_end",
]
ToolExecutionMode: TypeAlias = Literal["serial", "parallel"]
AgentErrorKind: TypeAlias = Literal["provider_error", "tool_error", "runtime_error", "aborted"]
ToolExecutionResult: TypeAlias = str | dict[str, Any] | ToolResultMessage
AgentPayload: TypeAlias = dict[str, Any]


@dataclass(slots=True)
class AbortSignal:
    """定义 Agent 内统一使用的中断信号。"""

    reason: str | None = None  # 中断原因。
    _event: asyncio.Event = field(default_factory=asyncio.Event)  # 内部等待/唤醒事件。

    @property
    def aborted(self) -> bool:
        """返回当前信号是否已被中断。"""

        return self._event.is_set()

    def abort(self, reason: str | None = None) -> None:
        """触发中断并记录可选原因。"""

        if reason is not None:
            self.reason = reason
        self._event.set()

    async def wait(self) -> None:
        """等待直到信号进入中断状态。"""

        await self._event.wait()

    def throw_if_aborted(self) -> None:
        """在已中断时抛出取消异常。"""

        if self.aborted:
            raise asyncio.CancelledError(self.reason or "aborted")


@dataclass(slots=True)
class AgentError:
    """定义 Agent 运行期统一错误对象。"""

    kind: AgentErrorKind  # 错误分类。
    message: str  # 面向上层的错误消息。
    details: Any | None = None  # 原始错误细节。
    retriable: bool = False  # 是否适合重试。

    def __str__(self) -> str:
        return self.message

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.message == other
        if not isinstance(other, AgentError):
            return False
        return (
            self.kind == other.kind
            and self.message == other.message
            and self.details == other.details
            and self.retriable == other.retriable
        )


@dataclass(slots=True)
class AgentRuntimeFlags:
    """定义 Agent 运行中的临时标记位。"""

    isStreaming: bool = False  # 当前是否处于流式输出中。
    isRunning: bool = False  # Agent 循环是否正在运行。
    isCancelled: bool = False  # 本轮是否已被取消。
    turnIndex: int = 0  # 当前处于第几轮 turn。
    retryCount: int = 0  # 当前运行已重试次数。


@dataclass(slots=True)
class AgentContext:
    """定义一次 Agent 调用 AI 或工具时使用的上下文快照。"""

    systemPrompt: str | None = None  # 本次调用使用的系统提示词。
    messages: list[Any] = field(default_factory=list)  # 发送给模型的消息上下文。
    tools: list[Tool] = field(default_factory=list)  # 本次调用可见的工具列表。

    @property
    def history(self) -> list[Any]:
        """兼容旧实现读取 history。"""

        return self.messages


class AgentStreamFn(Protocol):
    """定义“调用 AI 的函数”应遵循的统一签名。"""

    async def __call__(
        self,
        model: Model | str,
        context: AgentContext,
        thinking: ReasoningLevel | None,
        registry: ProviderRegistry | None = None,
        *,
        signal: AbortSignal | None = None,
    ) -> StreamSession:
        """根据模型、上下文和思考级别返回一个流式会话。"""


class ConvertToLlmFn(Protocol):
    """定义将 Agent 内消息转换为 LLM 消息的钩子签名。"""

    def __call__(
        self,
        messages: list[Any],
        state: AgentState,
    ) -> Awaitable[list[ConversationMessage]] | list[ConversationMessage]:
        """把内部消息列表转换成可送给模型的消息列表。"""


class TransformContextFn(Protocol):
    """定义对模型调用上下文做裁剪或注入的钩子签名。"""

    def __call__(
        self,
        context: AgentContext,
        state: AgentState,
    ) -> Awaitable[AgentContext] | AgentContext:
        """在真正调用模型前变换上下文。"""


ToolUpdateFn: TypeAlias = Callable[[Any], Awaitable[None] | None]


class AgentToolExecutor(Protocol):
    """定义 Agent 工具执行器的统一签名。"""

    def __call__(
        self,
        tool_call_id: str,
        params: Any,
        signal: AbortSignal,
        on_update: ToolUpdateFn,
    ) -> Awaitable[ToolExecutionResult] | ToolExecutionResult:
        """执行工具，并返回标准化前的工具结果。"""


@dataclass(init=False, slots=True)
class AgentTool(Tool):
    """定义可供 Agent 调用的工具。"""

    executor: AgentToolExecutor | None = None  # 工具实际执行函数。

    def __init__(
        self,
        name: str,
        description: str | None = None,
        inputSchema: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        renderMetadata: dict[str, Any] | None = None,
        *,
        executor: AgentToolExecutor | None = None,
        execute: AgentToolExecutor | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.inputSchema = dict(inputSchema or {})
        self.metadata = dict(metadata or {})
        self.renderMetadata = dict(renderMetadata or {})
        self.executor = executor or execute

    @property
    def execute(self) -> AgentToolExecutor | None:
        """兼容旧实现读取 execute。"""

        return self.executor


@dataclass(init=False, slots=True)
class AgentState:
    """定义 Agent 在循环运行中的实时状态。"""

    systemPrompt: str | None = None  # 当前生效的系统提示词。
    model: Model | str | None = None  # 当前使用的模型或模型 ID。
    thinking: ReasoningLevel | None = None  # 当前推理等级。
    tools: list[AgentTool] = field(default_factory=list)  # 当前可调用工具。
    messages: list[Any] = field(default_factory=list)  # 累积的内部消息历史。
    stream_message: AssistantMessage | None = None  # 正在流式生成的 assistant 消息。
    pending_tool_calls: list[ToolCall] = field(default_factory=list)  # 尚未执行完成的工具调用。
    error: AgentError | None = None  # 当前运行中的最后错误。
    runtime_flags: AgentRuntimeFlags = field(default_factory=AgentRuntimeFlags)  # 运行期标记位。

    def __init__(
        self,
        systemPrompt: str | None = None,
        model: Model | str | None = None,
        thinking: ReasoningLevel | None = None,
        tools: list[AgentTool] | None = None,
        messages: list[Any] | None = None,
        stream_message: AssistantMessage | None = None,
        pending_tool_calls: list[ToolCall] | None = None,
        error: AgentError | str | None = None,
        runtime_flags: AgentRuntimeFlags | None = None,
        *,
        history: list[Any] | None = None,
        currentMessage: AssistantMessage | None = None,
        runningToolCall: ToolCall | None = None,
        isStreaming: bool | None = None,
    ) -> None:
        self.systemPrompt = systemPrompt  # 当前生效的系统提示词。
        self.model = model  # 当前模型或模型 ID。
        self.thinking = thinking  # 当前推理等级。
        self.tools = list(tools or [])  # 当前可用工具。
        self.messages = list(messages if messages is not None else (history or []))  # 运行中的内部消息历史。
        self.stream_message = stream_message if stream_message is not None else currentMessage  # 正在流式生成的 assistant 消息。
        self.pending_tool_calls = list(pending_tool_calls or ([runningToolCall] if runningToolCall is not None else []))  # 未处理完成的工具调用。
        if isinstance(error, str):
            self.error = AgentError(kind="runtime_error", message=error)
        else:
            self.error = error  # 最近错误对象。
        self.runtime_flags = replace(runtime_flags) if runtime_flags is not None else AgentRuntimeFlags()  # 运行时标记位。
        if isStreaming is not None:
            self.runtime_flags.isStreaming = isStreaming

    @property
    def history(self) -> list[Any]:
        """兼容旧实现读取 history。"""

        return self.messages

    @property
    def currentMessage(self) -> AssistantMessage | None:
        """兼容旧实现读取 currentMessage。"""

        return self.stream_message

    @property
    def runningToolCall(self) -> ToolCall | None:
        """兼容旧实现读取 runningToolCall。"""

        return self.pending_tool_calls[0] if self.pending_tool_calls else None

    @property
    def isStreaming(self) -> bool:
        """兼容旧实现读取 isStreaming。"""

        return self.runtime_flags.isStreaming


@dataclass(slots=True)
class BeforeToolCallContext:
    """定义 beforeToolCall 收到的上下文信息。"""

    state: AgentState  # 当前 Agent 状态。
    tool: AgentTool  # 即将执行的工具。
    toolCall: ToolCall  # 原始工具调用。
    params: Any  # 已解析参数。
    assistantMessage: AssistantMessage | None  # 发起调用的 assistant 消息。
    agentContext: AgentContext  # 调用发生时的上下文快照。
    signal: AbortSignal  # 当前运行的中断信号。

    @property
    def arguments(self) -> str:
        """兼容旧实现读取字符串参数。"""

        return self.toolCall.arguments_text


@dataclass(slots=True)
class AfterToolCallContext:
    """定义 afterToolCall 收到的上下文信息。"""

    state: AgentState  # 当前 Agent 状态。
    tool: AgentTool  # 已执行完成的工具。
    toolCall: ToolCall  # 原始工具调用。
    params: Any  # 已解析参数。
    result: ToolResultMessage  # 当前工具结果。
    assistantMessage: AssistantMessage | None  # 发起调用的 assistant 消息。
    agentContext: AgentContext  # 调用发生时的上下文快照。
    signal: AbortSignal  # 当前运行的中断信号。

    @property
    def arguments(self) -> str:
        """兼容旧实现读取字符串参数。"""

        return self.toolCall.arguments_text


@dataclass(slots=True)
class BeforeToolCallAllow:
    """表示 beforeToolCall 允许工具继续执行。"""

    action: Literal["allow"] = "allow"  # 动作类型：允许执行。


@dataclass(slots=True)
class BeforeToolCallSkip:
    """表示 beforeToolCall 跳过真实执行并直接返回替代结果。"""

    result: ToolExecutionResult  # 直接返回的替代结果。
    action: Literal["skip"] = "skip"  # 动作类型：跳过真实执行。


@dataclass(slots=True)
class BeforeToolCallError:
    """表示 beforeToolCall 阻止工具执行并返回错误。"""

    error: str  # 错误文本。
    details: Any | None = None  # 额外错误细节。
    action: Literal["error"] = "error"  # 动作类型：阻止执行。


BeforeToolCallResult: TypeAlias = BeforeToolCallAllow | BeforeToolCallSkip | BeforeToolCallError | None


@dataclass(slots=True)
class AfterToolCallPass:
    """表示 afterToolCall 保留原始工具结果。"""

    action: Literal["pass"] = "pass"  # 动作类型：保留原结果。


@dataclass(slots=True)
class AfterToolCallReplace:
    """表示 afterToolCall 用新结果替换原始工具结果。"""

    result: ToolExecutionResult  # 替换后的结果。
    action: Literal["replace"] = "replace"  # 动作类型：替换结果。


AfterToolCallResult: TypeAlias = AfterToolCallPass | AfterToolCallReplace | None


class BeforeToolCallFn(Protocol):
    """定义工具执行前安检钩子的签名。"""

    def __call__(
        self,
        context: BeforeToolCallContext,
    ) -> Awaitable[BeforeToolCallResult] | BeforeToolCallResult:
        """在工具执行前决定允许、跳过或阻止调用。"""


class AfterToolCallFn(Protocol):
    """定义工具执行后质检钩子的签名。"""

    def __call__(
        self,
        context: AfterToolCallContext,
    ) -> Awaitable[AfterToolCallResult] | AfterToolCallResult:
        """在工具执行后决定保留或替换结果。"""


class GetSteeringMessagesFn(Protocol):
    """定义中途插话消息获取函数的签名。"""

    def __call__(
        self,
        state: AgentState,
        signal: AbortSignal,
    ) -> Awaitable[list[Any]] | list[Any] | None:
        """在 turn 结束后获取 steering 消息。"""


class GetFollowUpMessagesFn(Protocol):
    """定义任务结束后消息获取函数的签名。"""

    def __call__(
        self,
        state: AgentState,
        signal: AbortSignal,
    ) -> Awaitable[list[Any]] | list[Any] | None:
        """在内层循环退出后获取 follow-up 消息。"""


@dataclass(slots=True)
class RetryDecision:
    """定义一次错误后的重试决策。"""

    shouldRetry: bool = False  # 是否应该重试。
    delaySeconds: float = 0.0  # 重试前等待秒数。


@dataclass(slots=True)
class RetryContext:
    """定义重试策略收到的上下文。"""

    error: AgentError  # 当前错误对象。
    state: AgentState  # 出错时的状态快照。
    attempt: int  # 当前是第几次尝试。
    signal: AbortSignal  # 当前运行的中断信号。


class RetryPolicyFn(Protocol):
    """定义 provider 错误自动重试策略的签名。"""

    def __call__(
        self,
        context: RetryContext,
    ) -> Awaitable[RetryDecision] | RetryDecision:
        """根据错误上下文决定是否重试。"""


@dataclass(init=False, slots=True)
class AgentLoopConfig:
    """定义 Agent 主循环运行所需的一切配置。"""

    systemPrompt: str | None = None  # 系统提示词。
    model: Model | str | None = None  # 当前模型。
    thinking: ReasoningLevel | None = None  # 推理等级。
    tools: list[AgentTool] = field(default_factory=list)  # 可用工具列表。
    stream: AgentStreamFn | None = None  # 自定义流式调用函数。
    convert_to_llm: ConvertToLlmFn | None = None  # 内部消息到 LLM 消息的转换函数。
    transform_context: TransformContextFn | None = None  # 发送前上下文变换函数。
    get_steering_messages: GetSteeringMessagesFn | None = None  # 中途插话消息提供函数。
    get_follow_up_messages: GetFollowUpMessagesFn | None = None  # 结束后 follow-up 消息提供函数。
    toolExecutionMode: ToolExecutionMode = "serial"  # 工具执行模式。
    beforeToolCall: BeforeToolCallFn | None = None  # 工具执行前钩子。
    afterToolCall: AfterToolCallFn | None = None  # 工具执行后钩子。
    retryPolicy: RetryPolicyFn | None = None  # 错误重试策略。
    registry: ProviderRegistry | None = None  # provider 注册表。
    traceRecorder: Any | None = None  # 可选 trace 记录器。

    def __init__(
        self,
        systemPrompt: str | None = None,
        model: Model | str | None = None,
        thinking: ReasoningLevel | None = None,
        tools: list[AgentTool] | None = None,
        stream: AgentStreamFn | None = None,
        convert_to_llm: ConvertToLlmFn | None = None,
        transform_context: TransformContextFn | None = None,
        get_steering_messages: GetSteeringMessagesFn | None = None,
        get_follow_up_messages: GetFollowUpMessagesFn | None = None,
        toolExecutionMode: ToolExecutionMode = "serial",
        beforeToolCall: BeforeToolCallFn | None = None,
        afterToolCall: AfterToolCallFn | None = None,
        retryPolicy: RetryPolicyFn | None = None,
        registry: ProviderRegistry | None = None,
        traceRecorder: Any | None = None,
        *,
        steer: GetSteeringMessagesFn | None = None,
        followUp: GetFollowUpMessagesFn | None = None,
    ) -> None:
        self.systemPrompt = systemPrompt  # 系统提示词。
        self.model = model  # 当前模型。
        self.thinking = thinking  # 推理等级。
        self.tools = list(tools or [])  # 可用工具。
        self.stream = stream  # 流式函数。
        self.convert_to_llm = convert_to_llm  # 消息转换函数。
        self.transform_context = transform_context  # 上下文变换函数。
        self.get_steering_messages = get_steering_messages or steer  # steering 提供函数。
        self.get_follow_up_messages = get_follow_up_messages or followUp  # follow-up 提供函数。
        self.toolExecutionMode = toolExecutionMode  # 工具执行模式。
        self.beforeToolCall = beforeToolCall  # 工具前置钩子。
        self.afterToolCall = afterToolCall  # 工具后置钩子。
        self.retryPolicy = retryPolicy  # 重试策略。
        self.registry = registry  # provider 注册表。
        self.traceRecorder = traceRecorder  # trace 记录器。

    @property
    def steer(self) -> GetSteeringMessagesFn | None:
        """兼容旧实现读取 steer。"""

        return self.get_steering_messages

    @steer.setter
    def steer(self, value: GetSteeringMessagesFn | None) -> None:
        self.get_steering_messages = value

    @property
    def followUp(self) -> GetFollowUpMessagesFn | None:
        """兼容旧实现读取 followUp。"""

        return self.get_follow_up_messages

    @followUp.setter
    def followUp(self, value: GetFollowUpMessagesFn | None) -> None:
        self.get_follow_up_messages = value


@dataclass(slots=True)
class AgentEvent:
    """定义对外暴露的统一 Agent 事件对象。"""

    type: AgentEventType  # 事件类型。
    state: AgentState | None = None  # 事件发生时的状态快照。
    payload: AgentPayload = field(default_factory=dict)  # 事件载荷。

    @property
    def message(self) -> Any:
        """兼容旧实现读取消息对象。"""

        return self.payload.get("message")

    @property
    def messageDelta(self) -> str | None:
        """兼容旧实现读取消息增量。"""

        return self.payload.get("messageDelta")

    @property
    def toolCall(self) -> ToolCall | None:
        """兼容旧实现读取工具调用。"""

        return self.payload.get("toolCall")

    @property
    def toolResult(self) -> ToolResultMessage | None:
        """兼容旧实现读取工具结果。"""

        return self.payload.get("toolResult")

    @property
    def error(self) -> str | None:
        """兼容旧实现读取错误文本。"""

        error = self.payload.get("error")
        if isinstance(error, AgentError):
            return error.message
        if isinstance(error, str):
            return error
        return None


def default_convert_to_llm(messages: list[Any], state: AgentState) -> list[ConversationMessage]:
    """默认把内部消息列表中过滤出的标准对话消息送给模型。"""

    _ = state
    return [message for message in messages if isinstance(message, (AssistantMessage, ToolResultMessage)) or getattr(message, "role", None) == "user"]


def default_transform_context(context: AgentContext, state: AgentState) -> AgentContext:
    """默认直接返回原始上下文，不做额外变换。"""

    _ = state
    return context


def default_retry_policy(context: RetryContext) -> RetryDecision:
    """默认不自动重试。"""

    _ = context
    return RetryDecision(shouldRetry=False, delaySeconds=0.0)


def to_llm_context(state: AgentState, loop: AgentLoopConfig) -> Context:
    """根据当前状态与配置构造真正送给模型的 Context。"""

    convert_fn = loop.convert_to_llm or default_convert_to_llm
    messages = convert_fn(state.messages, state)
    if asyncio.iscoroutine(messages):
        raise RuntimeError("to_llm_context must not be called with async convert_to_llm")
    context = AgentContext(
        systemPrompt=state.systemPrompt,
        messages=list(messages),
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
    transformed = transform_fn(context, state)
    if asyncio.iscoroutine(transformed):
        raise RuntimeError("to_llm_context must not be called with async transform_context")
    return Context(systemPrompt=transformed.systemPrompt, messages=transformed.messages, tools=transformed.tools)
