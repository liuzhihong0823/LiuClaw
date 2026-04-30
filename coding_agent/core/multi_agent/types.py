from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


TeamMemberStatus = Literal["working", "idle", "shutdown"]
ProtocolStatus = Literal["pending", "approved", "rejected", "expired"]
ProtocolKind = Literal["shutdown", "plan"]


@dataclass(slots=True)
class Envelope:
    """定义团队收件箱中的单条消息信封。"""

    id: str  # 消息唯一 ID。
    sender: str  # 发送方名称。
    recipient: str  # 接收方名称。
    message_type: str  # 消息类型，例如 message、shutdown_request。
    content: str  # 主要消息内容。
    timestamp: float  # 发送时间戳。
    request_id: str | None = None  # 关联协议请求 ID，可为空。
    metadata: dict[str, Any] = field(default_factory=dict)  # 额外结构化字段。


@dataclass(slots=True)
class ProtocolRequest:
    """定义一个需要 request_id 追踪的协议请求。"""

    request_id: str  # 协议请求 ID。
    kind: ProtocolKind  # 请求种类，目前支持 shutdown / plan。
    status: ProtocolStatus  # 当前协议状态。
    sender: str  # 发起方。
    recipient: str  # 接收方。
    content: str = ""  # 附带文本，例如计划正文或关停说明。
    created_at: float = 0.0  # 创建时间。
    updated_at: float = 0.0  # 最近更新时间。
    response: str = ""  # 审批或响应文本。
    metadata: dict[str, Any] = field(default_factory=dict)  # 额外元信息。


@dataclass(slots=True)
class TeamMemberState:
    """定义团队成员的持久化状态。"""

    name: str  # 成员名称。
    role: str  # 成员角色。
    status: TeamMemberStatus  # 当前状态。
    session_id: str = ""  # 成员关联的会话 ID。
    session_file: str = ""  # 成员关联的会话文件路径。
    parent_session: str = ""  # 父会话路径，便于追踪来源。
    model_id: str = ""  # 当前成员默认使用的模型。
    thinking: str = ""  # 当前成员默认使用的思考等级。
    last_error: str = ""  # 最近一次运行错误。
    last_polled_at: float = 0.0  # 最近一次 idle 轮询 inbox 的时间。
    updated_at: float = 0.0  # 最近一次状态更新时间。


@dataclass(slots=True)
class SpawnResult:
    """定义 spawn teammate 的返回结果。"""

    name: str  # 成员名称。
    role: str  # 成员角色。
    status: TeamMemberStatus  # spawn 之后的状态。
    created: bool  # 是否新建了成员。
    session_id: str = ""  # 关联的会话 ID。
    session_file: str = ""  # 关联的会话文件路径。
