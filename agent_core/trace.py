from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai.types import AssistantMessage, ConversationMessage, Context, TextContent, ThinkingContent, ToolCall, ToolResultMessage

from .types import AgentContext, AgentEvent, AgentLoopConfig, AgentState


def _utc_now() -> str:
    """返回当前 UTC 时间的 ISO 字符串表示。"""

    return datetime.now(UTC).isoformat()


def _truncate(text: str, *, limit: int = 500) -> tuple[str, bool]:
    """按给定上限截断文本，并返回是否发生截断。"""

    if len(text) <= limit:
        return text, False
    return text[:limit] + "...", True


def _message_text(message: Any) -> str:
    """尽量把不同消息对象归一化为纯文本。"""

    if message is None:
        return ""
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        parts: list[str] = []
        for item in message:
            parts.append(_message_text(item))
        return "".join(parts)
    if isinstance(message, AssistantMessage):
        return _message_text(message.content)
    if isinstance(message, ToolResultMessage):
        return _message_text(message.content)
    if hasattr(message, "text"):
        return str(message.text)
    if hasattr(message, "thinking"):
        return str(message.thinking)
    if hasattr(message, "content"):
        return _message_text(message.content)
    return str(message)


def _message_role(message: Any) -> str:
    """返回消息对象对应的角色名。"""

    return str(getattr(message, "role", message.__class__.__name__.lower()))


@dataclass(slots=True)
class TraceToolCall:
    """描述一次工具调用的最小可回放信息。"""

    id: str  # 工具调用唯一 ID。
    name: str  # 工具名称。
    arguments: str  # 工具参数文本。


@dataclass(slots=True)
class TraceContextSnapshot:
    """记录某一轮真正送给模型的上下文摘要。"""

    turn_index: int  # 所属 turn 序号。
    system_prompt: str  # 当前 system prompt 摘要文本。
    system_prompt_truncated: bool  # system prompt 是否被截断。
    message_roles: list[str]  # 送给模型的消息角色序列。
    messages_preview: list[str]  # 最近消息的可读预览。
    tools: list[str]  # 当前对模型可见的工具列表。


@dataclass(slots=True)
class TraceEvent:
    """定义一条统一 trace 事件。"""

    type: str  # 事件类型，如 `message_end`、`tool_execution_start`。
    turn_index: int  # 事件所属 turn 序号。
    message_text: str = ""  # 事件关联的消息文本。
    thinking_text: str = ""  # 事件关联的 thinking 文本。
    tool_name: str = ""  # 事件关联的工具名。
    tool_arguments: str = ""  # 工具参数文本。
    tool_result: str = ""  # 工具结果文本摘要。
    error_kind: str = ""  # 错误分类，如 `provider_error`。
    error_message: str = ""  # 错误消息文本。
    retry_count: int = 0  # 事件发生时累计的重试次数。
    message_id: str = ""  # 可选的消息标识。
    metadata: dict[str, Any] = field(default_factory=dict)  # 附加元信息。


@dataclass(slots=True)
class TraceTurn:
    """定义一次 turn 的完整 trace 片段。"""

    turn_index: int  # turn 序号。
    context: TraceContextSnapshot | None = None  # 送给模型的上下文快照。
    events: list[TraceEvent] = field(default_factory=list)  # 本轮收集到的事件时间线。


@dataclass(slots=True)
class TraceOutcome:
    """描述一次运行最终的结束状态。"""

    status: str  # 结束状态，如 `completed`、`aborted`、`error`。
    error_kind: str = ""  # 若失败，对应错误分类。
    error_message: str = ""  # 若失败，对应错误消息。


@dataclass(slots=True)
class TraceRecord:
    """定义一次完整 agent 运行的 trace 记录。"""

    run_id: str  # 本次运行唯一 ID。
    started_at: str  # 运行开始时间。
    finished_at: str = ""  # 运行结束时间。
    model_id: str = ""  # 本次运行使用的模型 ID。
    thinking: str | None = None  # 本次运行使用的 thinking 等级。
    tool_execution_mode: str = "serial"  # 工具执行模式。
    turns: list[TraceTurn] = field(default_factory=list)  # 全部 turn trace。
    outcome: TraceOutcome = field(default_factory=lambda: TraceOutcome(status="running"))  # 运行最终结果。
    retry_events: list[TraceEvent] = field(default_factory=list)  # 单独汇总的 retry 事件。
    abort_events: list[TraceEvent] = field(default_factory=list)  # 单独汇总的 abort 事件。
    compaction_events: list[TraceEvent] = field(default_factory=list)  # 上层补充写入的 compaction 事件。
    session_metadata: dict[str, Any] = field(default_factory=dict)  # 产品层补充的会话元数据。


class TraceSerializer:
    """负责 trace 的 JSON / Markdown 序列化。"""

    @staticmethod
    def dump_json(record: TraceRecord, path: Path) -> None:
        """将 trace 记录写入 JSON 文件。"""

        path.write_text(json.dumps(asdict(record), ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def render_markdown(record: TraceRecord) -> str:
        """把 trace 记录渲染成人类可读的 Markdown 回放文本。"""

        lines = [
            f"# Trace Replay: {record.run_id}",
            "",
            "## Run Summary",
            f"- model: `{record.model_id}`",
            f"- thinking: `{record.thinking}`",
            f"- tool_execution_mode: `{record.tool_execution_mode}`",
            f"- outcome: `{record.outcome.status}`",
        ]
        if record.outcome.error_message:
            lines.append(f"- error: `{record.outcome.error_message}`")
        lines.extend(
            [
                "",
                "## Input Context Summary",
            ]
        )
        for turn in record.turns:
            lines.append(f"### Turn {turn.turn_index}")
            if turn.context is not None:
                lines.append(f"- roles: {', '.join(turn.context.message_roles)}")
                lines.append(f"- tools: {', '.join(turn.context.tools) or '(none)'}")
                lines.append("- messages:")
                for preview in turn.context.messages_preview:
                    lines.append(f"  - {preview}")
            lines.append("- timeline:")
            for event in turn.events:
                desc = event.type
                if event.message_text:
                    desc += f" | message={event.message_text}"
                if event.tool_name:
                    desc += f" | tool={event.tool_name}"
                if event.tool_arguments:
                    desc += f" | args={event.tool_arguments}"
                if event.tool_result:
                    desc += f" | result={event.tool_result}"
                if event.error_message:
                    desc += f" | error={event.error_message}"
                lines.append(f"  - {desc}")
            lines.append("")
        if record.retry_events:
            lines.append("## Retry Records")
            for event in record.retry_events:
                lines.append(f"- turn {event.turn_index}: {event.error_message} -> retry_count={event.retry_count}")
            lines.append("")
        if record.abort_events:
            lines.append("## Abort Records")
            for event in record.abort_events:
                lines.append(f"- turn {event.turn_index}: {event.error_message}")
            lines.append("")
        if record.compaction_events:
            lines.append("## Compaction Records")
            for event in record.compaction_events:
                lines.append(f"- {event.metadata.get('action', 'compaction')}: {event.metadata}")
            lines.append("")
        lines.extend(
            [
                "## Final Result",
                f"- status: `{record.outcome.status}`",
                f"- error_kind: `{record.outcome.error_kind or 'none'}`",
                f"- error_message: `{record.outcome.error_message or 'none'}`",
            ]
        )
        return "\n".join(lines) + "\n"

    @staticmethod
    def dump_markdown(record: TraceRecord, path: Path) -> None:
        """将 trace 记录写入 Markdown 回放文件。"""

        path.write_text(TraceSerializer.render_markdown(record), encoding="utf-8")


class TraceReplayLoader:
    """负责从落盘的 trace 文件恢复 `TraceRecord`。"""

    @staticmethod
    def load(path: str | Path) -> TraceRecord:
        """从 JSON trace 文件加载一份完整 trace 记录。"""

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        turns = [
            TraceTurn(
                turn_index=item["turn_index"],
                context=TraceContextSnapshot(**item["context"]) if item.get("context") else None,
                events=[TraceEvent(**event) for event in item.get("events", [])],
            )
            for item in data.get("turns", [])
        ]
        return TraceRecord(
            run_id=data["run_id"],
            started_at=data["started_at"],
            finished_at=data.get("finished_at", ""),
            model_id=data.get("model_id", ""),
            thinking=data.get("thinking"),
            tool_execution_mode=data.get("tool_execution_mode", "serial"),
            turns=turns,
            outcome=TraceOutcome(**data.get("outcome", {"status": "unknown"})),
            retry_events=[TraceEvent(**event) for event in data.get("retry_events", [])],
            abort_events=[TraceEvent(**event) for event in data.get("abort_events", [])],
            compaction_events=[TraceEvent(**event) for event in data.get("compaction_events", [])],
            session_metadata=dict(data.get("session_metadata", {})),
        )


class AgentTraceCollector:
    """消费 `AgentEvent` 并构建统一 trace 记录。"""

    def __init__(self, loop: AgentLoopConfig, state: AgentState | None = None) -> None:
        """根据 loop 配置和可选初始状态初始化 collector。"""

        model_id = ""
        if loop.model is not None:
            model_id = str(getattr(loop.model, "id", loop.model))
        elif state is not None and state.model is not None:
            model_id = str(getattr(state.model, "id", state.model))
        self.record = TraceRecord(
            run_id=uuid.uuid4().hex[:12],
            started_at=_utc_now(),
            model_id=model_id,
            thinking=loop.thinking or (state.thinking if state is not None else None),
            tool_execution_mode=loop.toolExecutionMode,
        )
        self._turns: dict[int, TraceTurn] = {}

    def _ensure_turn(self, turn_index: int) -> TraceTurn:
        """确保指定 turn 对应的 `TraceTurn` 已存在。"""

        turn = self._turns.get(turn_index)
        if turn is None:
            turn = TraceTurn(turn_index=turn_index)
            self._turns[turn_index] = turn
            self.record.turns.append(turn)
            self.record.turns.sort(key=lambda item: item.turn_index)
        return turn

    def record_context_snapshot(self, turn_index: int, context: AgentContext | Context) -> None:
        """记录一次真正送给模型的上下文摘要。"""

        system_prompt, prompt_truncated = _truncate(str(context.systemPrompt or ""))
        previews: list[str] = []
        for message in list(context.messages)[-6:]:
            text, _ = _truncate(_message_text(message), limit=160)
            previews.append(f"{_message_role(message)}: {text}")
        snapshot = TraceContextSnapshot(
            turn_index=turn_index,
            system_prompt=system_prompt,
            system_prompt_truncated=prompt_truncated,
            message_roles=[_message_role(message) for message in context.messages],
            messages_preview=previews,
            tools=[tool.name for tool in getattr(context, "tools", [])],
        )
        self._ensure_turn(turn_index).context = snapshot

    def record_retry_decision(self, turn_index: int, *, error_message: str, retry_count: int, should_retry: bool, delay_seconds: float) -> None:
        """记录一次 provider 错误后的重试决策。"""

        event = TraceEvent(
            type="retry_decision",
            turn_index=turn_index,
            error_message=error_message,
            retry_count=retry_count,
            metadata={"should_retry": should_retry, "delay_seconds": delay_seconds},
        )
        self.record.retry_events.append(event)
        self._ensure_turn(turn_index).events.append(event)

    def record_before_tool_result(self, turn_index: int, tool_call: ToolCall, *, outcome: str, error_message: str = "") -> None:
        """记录 beforeToolCall 阶段的 skip / error 结果。"""

        event = TraceEvent(
            type="before_tool_result",
            turn_index=turn_index,
            tool_name=tool_call.name,
            tool_arguments=tool_call.arguments_text,
            error_message=error_message,
            metadata={"outcome": outcome},
        )
        self._ensure_turn(turn_index).events.append(event)

    def record_abort(self, turn_index: int, reason: str) -> None:
        """记录一次运行级 abort 事件。"""

        event = TraceEvent(type="abort", turn_index=turn_index, error_message=reason)
        self.record.abort_events.append(event)
        self._ensure_turn(turn_index).events.append(event)

    def record_compaction(self, action: str, details: dict[str, Any]) -> None:
        """记录由上层补充写入的 compaction 事件。"""

        event = TraceEvent(type="compaction", turn_index=details.get("turn_index", 0), metadata={"action": action, **details})
        self.record.compaction_events.append(event)

    def consume_event(self, event: AgentEvent) -> None:
        """消费一条 `AgentEvent`，并转换成统一 trace 事件。"""

        turn_index = 0
        if event.state is not None:
            turn_index = event.state.runtime_flags.turnIndex
        if "turnIndex" in event.payload:
            turn_index = int(event.payload["turnIndex"])
        tool_call = event.toolCall
        tool_result = event.toolResult
        message_text, _ = _truncate(_message_text(event.message))
        thinking_text = ""
        if isinstance(event.message, AssistantMessage):
            thinking_text, _ = _truncate(_message_text([item for item in event.message.content if isinstance(item, ThinkingContent)]))
        tool_result_text = ""
        if tool_result is not None:
            tool_result_text, _ = _truncate(_message_text(tool_result.content))
        trace_event = TraceEvent(
            type=event.type,
            turn_index=turn_index,
            message_text=message_text,
            thinking_text=thinking_text,
            tool_name=tool_call.name if tool_call is not None else (tool_result.toolName if tool_result is not None else ""),
            tool_arguments=tool_call.arguments_text if tool_call is not None else "",
            tool_result=tool_result_text,
            error_kind=str(getattr(event.state.error, "kind", "")) if event.state and event.state.error else "",
            error_message=event.error or "",
            retry_count=event.state.runtime_flags.retryCount if event.state is not None else 0,
            message_id=str(event.payload.get("message_id", "")),
            metadata={"tool_result_error": bool(getattr(tool_result, "isError", False))} if tool_result is not None else {},
        )
        self._ensure_turn(turn_index).events.append(trace_event)
        if event.type == "agent_end":
            status = "completed"
            if trace_event.error_kind == "aborted":
                status = "aborted"
            elif trace_event.error_message:
                status = "error"
            self.record.outcome = TraceOutcome(status=status, error_kind=trace_event.error_kind, error_message=trace_event.error_message)

    def finalize(self, *, session_metadata: dict[str, Any] | None = None) -> TraceRecord:
        """结束收集，并返回最终 trace 记录。"""

        self.record.finished_at = _utc_now()
        if session_metadata:
            self.record.session_metadata = dict(session_metadata)
        if self.record.outcome.status == "running":
            self.record.outcome = TraceOutcome(status="completed")
        return self.record


def build_trace_listener(collector: AgentTraceCollector):
    """构造一个可直接注册给 `Agent` 的 trace 监听器。"""

    def _listener(event: AgentEvent) -> None:
        """把事件转交给 collector 处理。"""

        collector.consume_event(event)
    return _listener


__all__ = [
    "AgentTraceCollector",
    "TraceContextSnapshot",
    "TraceEvent",
    "TraceOutcome",
    "TraceRecord",
    "TraceReplayLoader",
    "TraceSerializer",
    "TraceToolCall",
    "TraceTurn",
    "build_trace_listener",
]
