from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from ai.types import AssistantMessage, ConversationMessage, Model
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

    max_read_chars: int = 12000  # 单次读取工具允许返回的最大字符数。
    max_command_chars: int = 12000  # 单次命令工具允许接收的最大命令长度。
    max_ls_entries: int = 200  # `ls` 类工具最大返回条目数。
    max_find_entries: int = 200  # `find` 类工具最大返回匹配数。
    allow_bash: bool = True  # 是否允许启用 bash 工具。


@dataclass(slots=True)
class CodingAgentSettings:
    """定义 coding-agent 运行时使用的合并后设置。"""

    default_model: str = "openai:gpt-5"  # 默认模型 ID。
    default_thinking: ReasoningLevel = "medium"  # 默认思考强度。
    auto_compact: bool = True  # 是否自动触发上下文压缩。
    compact_threshold: float = 0.8  # 接近窗口上限多少比例时触发压缩。
    compact_keep_turns: int = 4  # 压缩时保留的最近 turn 数。
    compact_model: str | None = None  # 执行压缩摘要时使用的模型。
    theme: str = "default"  # 交互界面主题名。
    system_prompt_override: str | None = None  # 覆盖默认系统提示词的文本。
    tool_policy: ToolPolicy = field(default_factory=ToolPolicy)  # 工具限制策略。


@dataclass(slots=True)
class SkillResource:
    """表示一个已发现的技能摘要。"""

    name: str  # 技能名。
    description: str  # 技能描述。
    path: Path  # 技能文件路径。


@dataclass(slots=True)
class PromptResource:
    """表示一个已加载的提示模板资源。"""

    name: str  # 提示资源名。
    path: Path  # 提示文件路径。
    content: str  # 提示文本内容。


@dataclass(slots=True)
class ThemeResource:
    """表示一个已加载的主题资源。"""

    name: str  # 主题名。
    path: Path  # 主题文件路径。
    data: dict[str, Any]  # 主题样式配置。


@dataclass(slots=True)
class ExtensionResource:
    """表示一个已扫描但未执行的扩展资源。"""

    name: str  # 扩展名。
    path: Path  # 扩展目录或清单文件路径。
    module_path: Path | None = None  # 可执行扩展模块路径。
    source: str = ""  # 扩展来源标识。
    metadata: dict[str, Any] = field(default_factory=dict)  # 扩展清单元信息。


@dataclass(slots=True)
class ExtensionCommand:
    """表示扩展注册的一条命令描述。"""

    name: str  # 命令名。
    handler: Callable[..., Any] | None = None  # 命令处理函数。
    description: str = ""  # 命令描述。
    source: str = ""  # 命令来源扩展。


@dataclass(slots=True)
class ExtensionRuntime:
    """表示一次扩展装配后得到的运行时贡献结果。"""

    tools: list[AgentTool] = field(default_factory=list)  # 扩展贡献的工具。
    commands: list[ExtensionCommand] = field(default_factory=list)  # 扩展贡献的命令。
    provider_factories: dict[str, Callable[..., Any]] = field(default_factory=dict)  # 扩展注册的 provider 工厂。
    event_listeners: list[Callable[[AgentEvent], Any]] = field(default_factory=list)  # 扩展订阅的事件监听器。
    prompt_fragments: list[str] = field(default_factory=list)  # 扩展追加的系统提示片段。


@dataclass(slots=True)
class ControlMessage:
    """表示 steering 或 follow-up 这类显式控制消息。"""

    kind: Literal["steering", "follow_up", "custom"]  # 控制消息类型。
    content: str  # 控制消息正文。
    metadata: dict[str, Any] = field(default_factory=dict)  # 控制消息附加元信息。


@dataclass(slots=True)
class ToolExecutionContext:
    """描述一次工具执行前后可见的安全上下文。"""

    tool_name: str  # 当前执行的工具名。
    workspace_root: Path  # 工作区根目录。
    cwd: Path  # 工具执行时的当前目录。
    arguments: dict[str, Any] = field(default_factory=dict)  # 工具参数。
    mode: ToolMode = "workspace-write"  # 工具权限模式。


@dataclass(slots=True)
class ToolSecurityPolicy:
    """定义工具执行前后的统一安全策略接口。"""

    before_execute: Callable[[ToolExecutionContext], None] | None = None  # 执行前校验/拦截钩子。
    after_execute: Callable[[ToolExecutionContext, str], str] | None = None  # 执行后清洗结果钩子。


@dataclass(slots=True)
class ToolDefinition:
    """描述一个可构建的工具定义。"""

    name: str  # 工具定义名。
    description: str  # 工具说明。
    builder: Callable[[Path, Path, CodingAgentSettings], AgentTool]  # 工具构造函数。
    group: str = "general"  # 工具分组。
    built_in: bool = False  # 是否为内置工具。
    mode: ToolMode = "workspace-write"  # 工具默认权限模式。
    source: str = "core"  # 工具来源。
    metadata: dict[str, Any] = field(default_factory=dict)  # 工具定义元信息。


@dataclass(slots=True)
class ResourceBundle:
    """聚合一次资源扫描得到的全部资源。"""

    skills: list[SkillResource] = field(default_factory=list)  # 已加载技能。
    prompts: dict[str, PromptResource] = field(default_factory=dict)  # 已加载提示模板。
    themes: dict[str, ThemeResource] = field(default_factory=dict)  # 已加载主题。
    agents_context: str | None = None  # 工作区中的 `AGENTS.md` 等上下文。
    extensions: list[ExtensionResource] = field(default_factory=list)  # 扫描到的扩展资源。
    extension_runtime: ExtensionRuntime = field(default_factory=ExtensionRuntime)  # 扩展装配后的运行时贡献。


@dataclass(slots=True)
class SessionContext:
    """描述构建系统提示时所需的上下文信息。"""

    workspace_root: Path  # 工作区根目录。
    cwd: Path  # 当前会话目录。
    model: Model  # 当前模型。
    thinking: ReasoningLevel | None  # 当前思考等级。
    settings: CodingAgentSettings  # 当前生效设置。
    resources: ResourceBundle  # 已加载资源集合。
    tools_markdown: str  # 工具列表的 Markdown 渲染文本。
    extra_prompt_fragments: list[str] = field(default_factory=list)  # 额外系统提示片段。


@dataclass(slots=True)
class SessionEvent:
    """定义给交互层消费的统一会话事件。"""

    type: SessionEventType  # 事件类型。
    message: str = ""  # 主消息文本。
    delta: str = ""  # 流式增量文本。
    tool_name: str = ""  # 相关工具名。
    error: str | None = None  # 错误文本。
    panel: SessionPanel = "main"  # 建议展示面板。
    status_level: StatusLevel = "info"  # 状态等级。
    is_transient: bool = False  # 是否为临时状态，不适合长期保留。
    tool_arguments: str = ""  # 工具参数文本。
    tool_output_preview: str = ""  # 工具输出摘要。
    message_id: str = ""  # 事件关联消息 ID。
    turn_id: str = ""  # 所属 turn ID。
    source: str = ""  # 事件来源。
    render_group: str = ""  # 渲染归组标识。
    render_order: int = 0  # 渲染顺序权重。
    payload: dict[str, Any] = field(default_factory=dict)  # 附加事件数据。


@dataclass(slots=True)
class PersistedMessageNode:
    """表示落盘后的单条会话节点数据。"""

    id: str  # 节点 ID。
    role: str  # 消息角色。
    content: str  # 节点正文。
    parent_id: str | None  # 父节点 ID。
    branch_id: str  # 所属分支。
    message_type: str = "message"  # 节点消息类型。
    tool_name: str | None = None  # 工具结果节点中的工具名。
    tool_call_id: str | None = None  # 对应的工具调用 ID。
    thinking: str = ""  # assistant 的思考文本。
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # assistant 携带的工具调用列表。
    metadata: dict[str, Any] = field(default_factory=dict)  # 节点附加元信息。


@dataclass(slots=True)
class SessionSnapshot:
    """表示从会话文件中恢复出的完整快照。"""

    session_id: str  # 会话 ID。
    branch_id: str  # 当前活动分支。
    cwd: Path  # 会话目录。
    model_id: str  # 当前模型 ID。
    nodes: list[PersistedMessageNode] = field(default_factory=list)  # 会话节点列表。
    summaries: list[dict[str, Any]] = field(default_factory=list)  # 分支摘要记录。


@dataclass(slots=True)
class CompactResult:
    """表示一次上下文压缩的结果。"""

    summary: str  # 生成出的摘要文本。
    compacted_count: int  # 被压缩掉的节点数。
    branch_id: str  # 发生压缩的分支。
    node_ids: list[str] = field(default_factory=list)  # 被摘要覆盖的节点 ID 列表。


@dataclass(slots=True)
class ContextStats:
    """表示当前上下文的粗略 token 统计信息。"""

    estimated_tokens: int  # 当前估算 token 总量。
    limit: int  # 模型窗口上限。
    ratio: float  # 当前占用比例。


def assistant_from_parts(content: str, thinking: str = "") -> AssistantMessage:
    """根据文本与思考内容快捷构造 assistant 消息。"""

    return AssistantMessage(content=content, thinking=thinking)


def conversation_to_node_payload(message: ConversationMessage) -> dict[str, Any]:
    """把统一消息对象转换为可持久化的节点载荷。"""

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
