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

    max_read_chars: int = 12000  # 单次读取类工具允许返回的最大字符数。
    max_command_chars: int = 12000  # 单次命令类工具允许返回的最大字符数。
    max_ls_entries: int = 200  # `ls` 类工具默认最多返回的条目数。
    max_find_entries: int = 200  # `find` 类工具默认最多返回的条目数。
    allow_bash: bool = True  # 是否允许启用 `bash` 工具。


@dataclass(slots=True)
class CompactionSettings:
    """token 驱动的压缩设置。"""

    enabled: bool = True  # 是否启用上下文压缩。
    reserve_tokens: int = 16384  # 为模型最终回答预留的 token 预算。
    keep_recent_tokens: int = 20000  # 压缩时希望保留为原始明细的最近 token 预算。
    compact_model: str | None = None  # 可选的压缩专用模型 ID。


@dataclass(slots=True)
class BranchSummarySettings:
    """分支摘要设置。"""

    reserve_tokens: int = 16384  # 分支摘要阶段额外预留的 token 预算。
    skip_prompt: bool = False  # 是否跳过分支摘要提示词扩展。


@dataclass(slots=True)
class CodingAgentSettings:
    """定义 coding-agent 运行时使用的合并后设置。"""

    default_model: str = "openai:gpt-5"  # 默认使用的模型 ID。
    default_thinking: ReasoningLevel = "medium"  # 默认思考强度。
    theme: str = "default"  # 交互界面默认主题。
    system_prompt_override: str | None = None  # 可选的系统提示词整体覆盖内容。
    tool_policy: ToolPolicy = field(default_factory=ToolPolicy)  # 工具执行限制策略。
    compaction: CompactionSettings = field(default_factory=CompactionSettings)  # 上下文压缩配置。
    branch_summary: BranchSummarySettings = field(default_factory=BranchSummarySettings)  # 分支摘要配置。
    # Legacy fields kept for backwards compatibility while the codebase migrates.
    auto_compact: bool = True  # 旧版自动压缩开关，保留兼容。
    compact_threshold: float = 0.8  # 旧版压缩阈值，保留兼容。
    compact_keep_turns: int = 4  # 旧版保留轮次数配置，保留兼容。
    compact_model: str | None = None  # 旧版压缩模型字段，保留兼容。


@dataclass(slots=True)
class SkillResource:
    name: str  # skill 名称。
    description: str  # skill 描述。
    path: Path  # skill 文件路径。


@dataclass(slots=True)
class PromptResource:
    name: str  # prompt 资源名。
    path: Path  # prompt 文件路径。
    content: str  # prompt 文件内容。


@dataclass(slots=True)
class ThemeResource:
    name: str  # 主题名称。
    path: Path  # 主题文件路径。
    data: dict[str, Any]  # 主题配置数据。


@dataclass(slots=True)
class ExtensionResource:
    name: str  # 扩展名称。
    path: Path  # 扩展根路径或 manifest 路径。
    module_path: Path | None = None  # 扩展入口模块路径。
    source: str = ""  # 扩展来源标识。
    metadata: dict[str, Any] = field(default_factory=dict)  # 扩展元信息。


@dataclass(slots=True)
class ExtensionCommand:
    name: str  # 扩展命令名。
    handler: Callable[..., Any] | None = None  # 命令处理函数。
    description: str = ""  # 命令描述。
    source: str = ""  # 命令来源扩展。


@dataclass(slots=True)
class ExtensionRuntime:
    tools: list[AgentTool] = field(default_factory=list)  # 扩展贡献的工具集合。
    commands: list[ExtensionCommand] = field(default_factory=list)  # 扩展贡献的命令集合。
    provider_factories: dict[str, Callable[..., Any]] = field(default_factory=dict)  # 扩展注册的 provider 工厂。
    event_listeners: list[Callable[[AgentEvent], Any]] = field(default_factory=list)  # 扩展订阅的事件监听器。
    prompt_fragments: list[str] = field(default_factory=list)  # 追加到系统提示词末尾的片段。


@dataclass(slots=True)
class ToolExecutionContext:
    tool_name: str  # 当前执行的工具名。
    workspace_root: Path  # 工作区根目录。
    cwd: Path  # 当前工具执行目录。
    arguments: dict[str, Any] = field(default_factory=dict)  # 工具调用参数。
    mode: ToolMode = "workspace-write"  # 当前工具模式。


@dataclass(slots=True)
class ToolSecurityPolicy:
    before_execute: Callable[[ToolExecutionContext], None] | None = None  # 执行前校验钩子。
    after_execute: Callable[[ToolExecutionContext, str], str] | None = None  # 执行后结果处理钩子。


@dataclass(slots=True)
class ToolDefinition:
    name: str  # 工具名称。
    description: str  # 工具说明。
    builder: Callable[[Path, Path, CodingAgentSettings], AgentTool]  # 构造工具实例的工厂函数。
    group: str = "general"  # 工具分组。
    built_in: bool = False  # 是否为内置工具。
    mode: ToolMode = "workspace-write"  # 工具默认执行模式。
    source: str = "core"  # 工具来源。
    metadata: dict[str, Any] = field(default_factory=dict)  # 工具附加元信息。


@dataclass(slots=True)
class ResourceBundle:
    skills: list[SkillResource] = field(default_factory=list)  # 已加载的 skills。
    prompts: dict[str, PromptResource] = field(default_factory=dict)  # 已加载的 prompts。
    themes: dict[str, ThemeResource] = field(default_factory=dict)  # 已加载的 themes。
    agents_context: str | None = None  # `AGENTS.md` 等补充上下文文本。
    extensions: list[ExtensionResource] = field(default_factory=list)  # 已发现的扩展资源。
    extension_runtime: ExtensionRuntime = field(default_factory=ExtensionRuntime)  # 已汇总的扩展运行时能力。


@dataclass(slots=True)
class SessionContext:
    """描述构建系统提示时所需的上下文信息。"""

    workspace_root: Path  # 工作区根目录。
    cwd: Path  # 当前会话目录。
    model: Model  # 当前模型对象。
    thinking: ReasoningLevel | None  # 当前思考级别。
    settings: CodingAgentSettings  # 当前生效设置。
    resources: ResourceBundle  # 当前已加载资源集合。
    tools_markdown: str  # 工具说明的 markdown 文本。
    extra_prompt_fragments: list[str] = field(default_factory=list)  # 额外追加的 prompt 片段。


@dataclass(slots=True)
class SessionEvent:
    type: SessionEventType  # 事件类型。
    message: str = ""  # 事件关联的完整文本。
    delta: str = ""  # 流式增量文本。
    tool_name: str = ""  # 关联工具名。
    error: str | None = None  # 错误描述。
    panel: SessionPanel = "main"  # 默认渲染面板。
    status_level: StatusLevel = "info"  # 状态级别。
    is_transient: bool = False  # 是否为临时态事件。
    tool_arguments: str = ""  # 工具参数文本。
    tool_output_preview: str = ""  # 工具输出预览。
    message_id: str = ""  # 消息标识。
    turn_id: str = ""  # 所属轮次标识。
    source: str = ""  # 事件来源。
    render_group: str = ""  # UI 渲染分组。
    render_order: int = 0  # UI 渲染顺序。
    payload: dict[str, Any] = field(default_factory=dict)  # 扩展负载数据。


@dataclass(slots=True)
class SessionHeader:
    type: Literal["session"] = "session"  # 记录类型，固定为 session 头。
    version: int = 1  # 会话文件格式版本。
    id: str = ""  # 会话 ID。
    timestamp: str = ""  # 会话创建时间。
    cwd: str = ""  # 会话工作目录。
    parent_session: str | None = None  # 父会话文件路径。
    title: str = ""  # 会话标题。
    model_id: str = ""  # 会话创建时的模型 ID。


@dataclass(slots=True)
class SessionEntryBase:
    id: str  # 条目 ID。
    parent_id: str | None  # 父节点 ID。
    timestamp: str  # 条目时间戳。


@dataclass(slots=True)
class SessionMessageEntry(SessionEntryBase):
    type: Literal["message"] = field(default="message", init=False)  # 条目类型。
    message: ConversationMessage = field(default_factory=UserMessage)  # 存储的真实对话消息。


@dataclass(slots=True)
class ThinkingLevelChangeEntry(SessionEntryBase):
    type: Literal["thinking_level_change"] = field(default="thinking_level_change", init=False)  # 条目类型。
    thinking_level: str = "medium"  # 新的 thinking 级别。


@dataclass(slots=True)
class ModelChangeEntry(SessionEntryBase):
    type: Literal["model_change"] = field(default="model_change", init=False)  # 条目类型。
    provider: str = ""  # 目标 provider。
    model_id: str = ""  # 目标模型 ID。


@dataclass(slots=True)
class CompactionEntry(SessionEntryBase):
    type: Literal["compaction"] = field(default="compaction", init=False)  # 条目类型。
    summary: str = ""  # 压缩后生成的摘要文本。
    first_kept_entry_id: str = ""  # 压缩边界之后第一个保留的原始条目 ID。
    tokens_before: int = 0  # 压缩前估算的 token 数。
    details: dict[str, Any] | None = None  # 附加压缩细节。
    from_hook: bool = False  # 是否由 hook 触发生成。


@dataclass(slots=True)
class BranchSummaryEntry(SessionEntryBase):
    type: Literal["branch_summary"] = field(default="branch_summary", init=False)  # 条目类型。
    from_id: str = ""  # 摘要所对应的分支起点。
    summary: str = ""  # 分支摘要文本。
    details: dict[str, Any] | None = None  # 附加分支信息。
    from_hook: bool = False  # 是否由 hook 触发生成。


@dataclass(slots=True)
class CustomEntry(SessionEntryBase):
    type: Literal["custom"] = field(default="custom", init=False)  # 条目类型。
    custom_type: str = ""  # 自定义记录类型。
    data: Any | None = None  # 自定义负载数据。


@dataclass(slots=True)
class CustomMessageEntry(SessionEntryBase):
    type: Literal["custom_message"] = field(default="custom_message", init=False)  # 条目类型。
    custom_type: str = ""  # 自定义消息类型。
    content: str | list[dict[str, Any]] = ""  # 自定义消息内容。
    details: dict[str, Any] | None = None  # 自定义附加细节。
    display: bool = True  # 是否允许在界面显示。


@dataclass(slots=True)
class LabelEntry(SessionEntryBase):
    type: Literal["label"] = field(default="label", init=False)  # 条目类型。
    target_id: str = ""  # 被标记的目标节点 ID。
    label: str | None = None  # 标记文本；为空表示清除。


@dataclass(slots=True)
class SessionInfoEntry(SessionEntryBase):
    type: Literal["session_info"] = field(default="session_info", init=False)  # 条目类型。
    name: str | None = None  # 会话展示名称。


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
    entry: SessionEntry  # 当前树节点对应的条目。
    children: list["SessionTreeNode"] = field(default_factory=list)  # 子节点列表。
    label: str | None = None  # 当前节点标签。


@dataclass(slots=True)
class SessionConversationContext:
    messages: list[ConversationMessage] = field(default_factory=list)  # 恢复后的消息列表。
    thinking_level: str = "off"  # 当前分支最终生效的 thinking 级别。
    model: dict[str, str] | None = None  # 当前分支最终生效的模型信息。


@dataclass(slots=True)
class SessionInfo:
    path: str  # session 文件路径。
    id: str  # session ID。
    cwd: str  # 工作目录。
    name: str | None  # 用户命名的会话名。
    parent_session_path: str | None  # 父会话路径。
    created_at: str  # 创建时间。
    modified_at: str  # 最近更新时间。
    message_count: int  # 消息条数。
    first_message: str  # 第一条用户消息摘要。
    all_messages_text: str  # 全量消息拼接文本，便于搜索。
    leaf_id: str | None = None  # 当前叶子节点 ID。
    title: str = ""  # 标题。
    model_id: str = ""  # 当前模型 ID。

@dataclass(slots=True)
class CompactResult:
    summary: str  # 生成的摘要文本。
    compacted_count: int  # 本次被压缩的消息数量。
    first_kept_entry_id: str  # 压缩后第一个保留的原始条目 ID。
    tokens_before: int  # 压缩前估算的 token 数。
    details: dict[str, Any] | None = None  # 附加细节。
    branch_id: str = ""  # 兼容旧接口的分支标识。
    node_ids: list[str] = field(default_factory=list)  # 兼容旧接口的节点列表。


@dataclass(slots=True)
class ContextStats:
    estimated_tokens: int  # 当前上下文估算 token 数。
    limit: int  # 模型上下文上限。
    ratio: float  # 估算占比。


def assistant_from_parts(content: str, thinking: str = "") -> AssistantMessage:
    """根据文本内容和思考内容快速构造 assistant 消息。"""

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
