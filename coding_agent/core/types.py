from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from ai.types import AssistantMessage, ConversationMessage, Model, ToolCall, ToolResultMessage, UserMessage
from agent_core import AgentEvent, AgentTool

ReasoningLevel = Literal["low", "medium", "high"]
SessionEventType = Literal[
    "status",
    "thinking",
    "message_start",
    "message_delta",
    "message_end",
    "tool_start",
    "tool_update",
    "tool_end",
    "error",
]
SessionPanel = Literal["main", "status", "thinking", "tool", "error"]
StatusLevel = Literal["info", "success", "warning", "error"]
ToolMode = Literal["workspace-write", "read-only"]


@dataclass(slots=True)
class ToolPolicy:
    """定义内置工具的默认限制与安全策略。"""

    max_read_chars: int = 12000
    max_command_chars: int = 12000
    max_ls_entries: int = 200
    max_find_entries: int = 200
    allow_bash: bool = True


@dataclass(slots=True)
class CompactionSettings:
    """token 驱动的压缩设置。"""

    enabled: bool = True
    reserve_tokens: int = 16384
    keep_recent_tokens: int = 20000
    compact_model: str | None = None


@dataclass(slots=True)
class BranchSummarySettings:
    """分支摘要设置。"""

    reserve_tokens: int = 16384
    skip_prompt: bool = False


@dataclass(slots=True)
class CodingAgentSettings:
    """定义 coding-agent 运行时使用的合并后设置。"""

    default_model: str = "openai:gpt-5"
    default_thinking: ReasoningLevel = "medium"
    theme: str = "default"
    system_prompt_override: str | None = None
    tool_policy: ToolPolicy = field(default_factory=ToolPolicy)
    compaction: CompactionSettings = field(default_factory=CompactionSettings)
    branch_summary: BranchSummarySettings = field(default_factory=BranchSummarySettings)
    # Legacy fields kept for backwards compatibility while the codebase migrates.
    auto_compact: bool = True
    compact_threshold: float = 0.8
    compact_keep_turns: int = 4
    compact_model: str | None = None


@dataclass(slots=True)
class SkillResource:
    name: str
    description: str
    path: Path


@dataclass(slots=True)
class PromptResource:
    name: str
    path: Path
    content: str


@dataclass(slots=True)
class ThemeResource:
    name: str
    path: Path
    data: dict[str, Any]


@dataclass(slots=True)
class ExtensionResource:
    name: str
    path: Path
    module_path: Path | None = None
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExtensionCommand:
    name: str
    handler: Callable[..., Any] | None = None
    description: str = ""
    source: str = ""


@dataclass(slots=True)
class ExtensionRuntime:
    tools: list[AgentTool] = field(default_factory=list)
    commands: list[ExtensionCommand] = field(default_factory=list)
    provider_factories: dict[str, Callable[..., Any]] = field(default_factory=dict)
    event_listeners: list[Callable[[AgentEvent], Any]] = field(default_factory=list)
    prompt_fragments: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ToolExecutionContext:
    tool_name: str
    workspace_root: Path
    cwd: Path
    arguments: dict[str, Any] = field(default_factory=dict)
    mode: ToolMode = "workspace-write"


@dataclass(slots=True)
class ToolSecurityPolicy:
    before_execute: Callable[[ToolExecutionContext], None] | None = None
    after_execute: Callable[[ToolExecutionContext, str], str] | None = None


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    builder: Callable[[Path, Path, CodingAgentSettings], AgentTool]
    group: str = "general"
    built_in: bool = False
    mode: ToolMode = "workspace-write"
    source: str = "core"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ResourceBundle:
    skills: list[SkillResource] = field(default_factory=list)
    prompts: dict[str, PromptResource] = field(default_factory=dict)
    themes: dict[str, ThemeResource] = field(default_factory=dict)
    agents_context: str | None = None
    extensions: list[ExtensionResource] = field(default_factory=list)
    extension_runtime: ExtensionRuntime = field(default_factory=ExtensionRuntime)


@dataclass(slots=True)
class SessionContext:
    """描述构建系统提示时所需的上下文信息。"""

    workspace_root: Path
    cwd: Path
    model: Model
    thinking: ReasoningLevel | None
    settings: CodingAgentSettings
    resources: ResourceBundle
    tools_markdown: str
    extra_prompt_fragments: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SessionEvent:
    type: SessionEventType
    message: str = ""
    delta: str = ""
    tool_name: str = ""
    error: str | None = None
    panel: SessionPanel = "main"
    status_level: StatusLevel = "info"
    is_transient: bool = False
    tool_arguments: str = ""
    tool_output_preview: str = ""
    message_id: str = ""
    turn_id: str = ""
    source: str = ""
    render_group: str = ""
    render_order: int = 0
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SessionHeader:
    type: Literal["session"] = "session"
    version: int = 1
    id: str = ""
    timestamp: str = ""
    cwd: str = ""
    parent_session: str | None = None
    title: str = ""
    model_id: str = ""


@dataclass(slots=True)
class SessionEntryBase:
    id: str
    parent_id: str | None
    timestamp: str


@dataclass(slots=True)
class SessionMessageEntry(SessionEntryBase):
    type: Literal["message"] = field(default="message", init=False)
    message: ConversationMessage = field(default_factory=UserMessage)


@dataclass(slots=True)
class ThinkingLevelChangeEntry(SessionEntryBase):
    type: Literal["thinking_level_change"] = field(default="thinking_level_change", init=False)
    thinking_level: str = "medium"


@dataclass(slots=True)
class ModelChangeEntry(SessionEntryBase):
    type: Literal["model_change"] = field(default="model_change", init=False)
    provider: str = ""
    model_id: str = ""


@dataclass(slots=True)
class CompactionEntry(SessionEntryBase):
    type: Literal["compaction"] = field(default="compaction", init=False)
    summary: str = ""
    first_kept_entry_id: str = ""
    tokens_before: int = 0
    details: dict[str, Any] | None = None
    from_hook: bool = False


@dataclass(slots=True)
class BranchSummaryEntry(SessionEntryBase):
    type: Literal["branch_summary"] = field(default="branch_summary", init=False)
    from_id: str = ""
    summary: str = ""
    details: dict[str, Any] | None = None
    from_hook: bool = False


@dataclass(slots=True)
class CustomEntry(SessionEntryBase):
    type: Literal["custom"] = field(default="custom", init=False)
    custom_type: str = ""
    data: Any | None = None


@dataclass(slots=True)
class CustomMessageEntry(SessionEntryBase):
    type: Literal["custom_message"] = field(default="custom_message", init=False)
    custom_type: str = ""
    content: str | list[dict[str, Any]] = ""
    details: dict[str, Any] | None = None
    display: bool = True


@dataclass(slots=True)
class LabelEntry(SessionEntryBase):
    type: Literal["label"] = field(default="label", init=False)
    target_id: str = ""
    label: str | None = None


@dataclass(slots=True)
class SessionInfoEntry(SessionEntryBase):
    type: Literal["session_info"] = field(default="session_info", init=False)
    name: str | None = None


SessionEntry = (
    SessionMessageEntry
    | ThinkingLevelChangeEntry
    | ModelChangeEntry
    | CompactionEntry
    | BranchSummaryEntry
    | CustomEntry
    | CustomMessageEntry
    | LabelEntry
    | SessionInfoEntry
)


@dataclass(slots=True)
class SessionTreeNode:
    entry: SessionEntry
    children: list["SessionTreeNode"] = field(default_factory=list)
    label: str | None = None


@dataclass(slots=True)
class SessionConversationContext:
    messages: list[ConversationMessage] = field(default_factory=list)
    thinking_level: str = "off"
    model: dict[str, str] | None = None


@dataclass(slots=True)
class SessionInfo:
    path: str
    id: str
    cwd: str
    name: str | None
    parent_session_path: str | None
    created_at: str
    modified_at: str
    message_count: int
    first_message: str
    all_messages_text: str
    leaf_id: str | None = None
    title: str = ""
    model_id: str = ""


@dataclass(slots=True)
class SessionSnapshot:
    session_id: str
    session_file: Path
    cwd: Path
    model_id: str
    leaf_id: str | None = None
    entries: list[SessionEntry] = field(default_factory=list)
    header: SessionHeader | None = None
    title: str = ""

    @property
    def branch_id(self) -> str:
        """Legacy compatibility shim."""

        return self.leaf_id or "main"


@dataclass(slots=True)
class CompactResult:
    summary: str
    compacted_count: int
    first_kept_entry_id: str
    tokens_before: int
    details: dict[str, Any] | None = None
    branch_id: str = ""
    node_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ContextStats:
    estimated_tokens: int
    limit: int
    ratio: float


def assistant_from_parts(content: str, thinking: str = "") -> AssistantMessage:
    return AssistantMessage(content=content, thinking=thinking)


def conversation_to_node_payload(message: ConversationMessage) -> dict[str, Any]:
    """Legacy helper kept for old migration/tests."""

    payload: dict[str, Any] = {
        "role": getattr(message, "role", ""),
        "content": getattr(message, "content", ""),
        "metadata": dict(getattr(message, "metadata", {})),
    }
    if getattr(message, "role", "") == "assistant":
        payload["thinking"] = getattr(message, "thinking", "")
        payload["tool_calls"] = [
            {
                "id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
                "metadata": dict(tool_call.metadata),
            }
            for tool_call in getattr(message, "toolCalls", [])
        ]
    if getattr(message, "role", "") == "tool":
        payload["tool_name"] = getattr(message, "toolName", "")
        payload["tool_call_id"] = getattr(message, "toolCallId", "")
    return payload


def serialize_message(message: ConversationMessage) -> dict[str, Any]:
    """将统一消息对象转换成可持久化字典。"""

    if isinstance(message, UserMessage):
        return {
            "role": "user",
            "content": str(message.content),
            "metadata": dict(message.metadata),
            "timestamp": message.timestamp,
        }
    if isinstance(message, AssistantMessage):
        return {
            "role": "assistant",
            "content": str(message.content),
            "thinking": message.thinking,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                    "metadata": dict(tool_call.metadata),
                }
                for tool_call in message.toolCalls
            ],
            "metadata": dict(message.metadata),
            "usage": message.usage,
            "stop_reason": message.stopReason,
            "response_id": message.responseId,
            "error_message": message.errorMessage,
            "timestamp": message.timestamp,
        }
    if isinstance(message, ToolResultMessage):
        return {
            "role": "tool",
            "tool_call_id": message.toolCallId,
            "tool_name": message.toolName,
            "content": str(message.content),
            "metadata": dict(message.metadata),
            "is_error": message.isError,
            "details": message.details,
            "timestamp": message.timestamp,
        }
    raise TypeError(f"Unsupported message type: {type(message)!r}")


def deserialize_message(data: dict[str, Any]) -> ConversationMessage:
    """从持久化字典恢复统一消息对象。"""

    role = str(data.get("role", "user"))
    if role == "assistant":
        return AssistantMessage(
            content=str(data.get("content", "")),
            thinking=str(data.get("thinking", "")),
            toolCalls=[
                ToolCall(
                    id=str(item.get("id", "")),
                    name=str(item.get("name", "")),
                    arguments=item.get("arguments", {}),
                    metadata=dict(item.get("metadata", {})),
                )
                for item in data.get("tool_calls", [])
            ],
            metadata=dict(data.get("metadata", {})),
            usage=data.get("usage"),
            stopReason=data.get("stop_reason"),
            responseId=data.get("response_id"),
            errorMessage=data.get("error_message"),
            timestamp=int(data.get("timestamp") or 0) or None,
        )
    if role == "tool":
        return ToolResultMessage(
            toolCallId=str(data.get("tool_call_id", "")),
            toolName=str(data.get("tool_name", "")),
            content=str(data.get("content", "")),
            metadata=dict(data.get("metadata", {})),
            isError=bool(data.get("is_error", False)),
            details=data.get("details"),
            timestamp=int(data.get("timestamp") or 0),
        )
    return UserMessage(
        content=str(data.get("content", "")),
        metadata=dict(data.get("metadata", {})),
        timestamp=int(data.get("timestamp") or 0),
    )
