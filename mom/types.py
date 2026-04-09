from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


def utc_now_iso() -> str:
    """返回当前 UTC 时间的 ISO 8601 字符串。"""
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ChatAttachment:
    """聊天消息中的附件信息。"""

    original_name: str  # 附件原始文件名。
    local_path: str = ""  # 附件在工作区中的本地相对路径。
    mime_type: str | None = None  # 附件 MIME 类型。
    file_key: str | None = None  # 平台侧附件标识。
    message_id: str | None = None  # 所属消息 ID。
    size: int | None = None  # 附件大小。
    metadata: dict[str, Any] = field(default_factory=dict)  # 附件附加元信息。


@dataclass(slots=True)
class ChatEvent:
    """统一后的聊天事件模型，作为 mom 内部的核心输入。"""

    platform: str  # 来源平台。
    chat_id: str  # 会话或频道 ID。
    message_id: str  # 消息唯一 ID。
    sender_id: str  # 发送者 ID。
    sender_name: str  # 发送者名称。
    text: str  # 规范化后的消息正文。
    attachments: list[ChatAttachment] = field(default_factory=list)  # 附件列表。
    is_direct: bool = False  # 是否为私聊。
    is_trigger: bool = False  # 是否应触发机器人执行。
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))  # 事件发生时间。
    chat_name: str | None = None  # 会话名称。
    mentions: list[str] = field(default_factory=list)  # 消息中出现的提及列表。
    metadata: dict[str, Any] = field(default_factory=dict)  # 事件附加元信息。


@dataclass(slots=True)
class ChatUser:
    """聊天参与者信息。"""

    id: str  # 用户 ID。
    name: str  # 用户名。
    display_name: str | None = None  # 展示名。


@dataclass(slots=True)
class ChatInfo:
    """聊天会话的基础信息。"""

    id: str  # 会话 ID。
    name: str  # 会话名称。


RespondFn = Callable[[str, bool], Awaitable[str | None]]
ReplaceFn = Callable[[str], Awaitable[str | None]]
DetailFn = Callable[[str], Awaitable[str | None]]
UploadFn = Callable[[str, str | None], Awaitable[str | None]]
WorkingFn = Callable[[bool], Awaitable[None]]
DeleteFn = Callable[[], Awaitable[None]]


@dataclass(slots=True)
class ChatContext:
    """runner 与外部聊天平台之间的交互抽象。"""

    message: ChatEvent  # 当前触发消息。
    chat_name: str | None  # 当前会话名。
    users: list[ChatUser]  # 已知用户列表。
    chats: list[ChatInfo]  # 已知会话列表。
    respond: RespondFn  # 发送或更新主回复的方法。
    replace_message: ReplaceFn  # 替换主消息的方法。
    respond_detail: DetailFn  # 发送明细消息的方法。
    upload_file: UploadFn  # 上传文件的方法。
    set_working: WorkingFn  # 设置工作中状态的方法。
    delete_message: DeleteFn  # 删除消息的方法。
    metadata: dict[str, Any] = field(default_factory=dict)  # 运行期附加元信息。


@dataclass(slots=True)
class RunResult:
    """一次 Agent 执行结束后的汇总结果。"""

    stop_reason: Literal["completed", "aborted", "error"] = "completed"  # 停止原因。
    error_message: str | None = None  # 错误消息。
    main_message_id: str | None = None  # 主消息 ID。
    final_text: str = ""  # 最终回复文本。
    detail_count: int = 0  # 发送的 detail 消息数。
    suppressed_events_count: int = 0  # 被隐藏的中间事件数。


@dataclass(slots=True)
class MomRenderConfig:
    """mom 输出渲染策略配置。"""

    render_mode: Literal["final_only", "streaming"] = "final_only"  # 渲染模式。
    placeholder_text: str = "处理中…"  # 执行中占位文案。
    show_intermediate_updates: bool = False  # 是否展示流式中间文本。
    show_tool_details: bool = False  # 是否展示工具详情。
    show_thinking: bool = False  # 是否展示 thinking 事件。


@dataclass(slots=True)
class SessionRef:
    """频道与 Agent 会话之间的持久化引用。"""

    session_id: str  # 底层会话 ID。
    branch_id: str = "main"  # 当前分支 ID。
    synced_message_ids: list[str] = field(default_factory=list)  # 已同步进会话的消息 ID。


@dataclass(slots=True)
class ChannelState:
    """频道级运行态，记录是否忙碌、排队消息和停止状态。"""

    running: bool = False  # 当前频道是否正在处理任务。
    runner: Any | None = None  # 复用的 runner 对象。
    store: Any | None = None  # 关联存储对象。
    stop_requested: bool = False  # 是否收到停止请求。
    stop_message_id: str | None = None  # 停止提示消息 ID。
    queued_events: list[ChatEvent] = field(default_factory=list)  # 排队等待的事件。
    current_message_id: str | None = None  # 当前处理中的消息 ID。
    recent_incoming_message_ids: list[str] = field(default_factory=list)  # 最近处理过的输入消息 ID。


@dataclass(slots=True)
class LoggedChatMessage:
    """写入频道日志文件的标准消息结构。"""

    platform: str  # 平台名。
    chat_id: str  # 会话 ID。
    message_id: str  # 消息 ID。
    sender_id: str  # 发送者 ID。
    sender_name: str  # 发送者名称。
    text: str  # 文本正文。
    is_bot: bool  # 是否为机器人消息。
    created_at: str = field(default_factory=utc_now_iso)  # 记录创建时间。
    attachments: list[dict[str, Any]] = field(default_factory=list)  # 附件列表。
    direct: bool = False  # 是否为私聊消息。
    trigger: bool = False  # 是否触发执行。
    response_kind: str = "message"  # 回复类型。
    metadata: dict[str, Any] = field(default_factory=dict)  # 附加元信息。


@dataclass(slots=True)
class MomPaths:
    """mom 在工作区下使用的目录与文件路径集合。"""

    workspace_root: Path  # 工作区根目录。
    root: Path  # `.mom` 根目录。
    channels_dir: Path  # 频道目录根路径。
    events_dir: Path  # 事件目录。
    sessions_dir: Path  # 会话目录。
    settings_file: Path  # mom 设置文件。
    channel_index_file: Path  # 频道与会话索引文件。

    def ensure_exists(self) -> None:
        """确保 mom 运行依赖的目录和配置文件已经存在。"""
        self.root.mkdir(parents=True, exist_ok=True)
        self.channels_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        if not self.settings_file.exists():
            self.settings_file.write_text("{}\n", encoding="utf-8")
        if not self.channel_index_file.exists():
            self.channel_index_file.write_text("{}\n", encoding="utf-8")
