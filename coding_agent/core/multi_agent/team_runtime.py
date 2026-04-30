from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_core import AgentTool

from .message_bus import MessageBus
from .protocols import ProtocolTracker
from .teammate import TeammateHandle
from .types import Envelope, ProtocolRequest, SpawnResult, TeamMemberState

if TYPE_CHECKING:
    from ..agent_session import AgentSession
    from ..model_registry import ModelRegistry
    from ..resource_loader import ResourceLoader
    from ..session_manager import SessionManager
    from ..tools import ToolRegistry


@dataclass(slots=True)
class TeamSharedState:
    """定义同一团队在内存中的共享状态。"""

    handles: dict[str, TeammateHandle] = field(default_factory=dict)  # 仅当前进程内活跃的 worker 句柄。


class TeamRuntime:
    """负责 lead / worker 的团队级编排与持久化控制面。"""

    def __init__(
        self,
        *,
        owner_session: "AgentSession",
        workspace_root: Path,
        model_registry: "ModelRegistry | None" = None,
        owner_name: str = "lead",
        owner_role: str = "lead",
        shared_state: TeamSharedState | None = None,
        idle_poll_interval: float = 2.0,
    ) -> None:
        """为某个会话创建或恢复团队运行时。"""

        self.owner_session = owner_session  # 挂载当前 runtime 的会话。
        self.workspace_root = workspace_root  # 团队控制面所在工作区。
        self.model_registry = model_registry  # 用于按模型 ID 恢复 worker 模型。
        self.owner_name = owner_name  # 当前 runtime 对应的成员名称。
        self.owner_role = owner_role  # 当前 runtime 对应的成员角色。
        self.shared_state = shared_state or TeamSharedState()  # 当前进程内的共享活跃句柄。
        self.idle_poll_interval = idle_poll_interval  # idle 阶段轮询 inbox 的时间间隔。
        self.team_root = workspace_root / ".liuclaw" / "team"  # 团队控制面根目录。
        self.runtime_dir = self.team_root / "runtime"  # 团队运行时附属目录。
        self.inbox_dir = self.team_root / "inbox"  # 收件箱目录。
        self.config_path = self.team_root / "config.json"  # 团队名册配置。
        self.protocols_path = self.runtime_dir / "protocols.json"  # 协议状态文件。
        self.team_root.mkdir(parents=True, exist_ok=True)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.bus = MessageBus(self.inbox_dir)  # 团队消息总线。
        self.protocols = ProtocolTracker(self.protocols_path)  # 团队协议追踪器。
        self._ensure_config()

    def register_tools(self, tool_registry: "ToolRegistry") -> None:
        """向工具注册表追加团队协作工具。"""

        tool_registry.register_tool(self._build_spawn_tool(), source="team", group="multi-agent", mode="workspace-write")
        tool_registry.register_tool(self._build_send_message_tool(), source="team", group="multi-agent", mode="workspace-write")
        tool_registry.register_tool(self._build_read_inbox_tool(), source="team", group="multi-agent", mode="workspace-write")
        tool_registry.register_tool(self._build_list_team_tool(), source="team", group="multi-agent", mode="read-only")
        tool_registry.register_tool(self._build_shutdown_request_tool(), source="team", group="multi-agent", mode="workspace-write")
        tool_registry.register_tool(self._build_shutdown_response_tool(), source="team", group="multi-agent", mode="workspace-write")
        tool_registry.register_tool(self._build_submit_plan_tool(), source="team", group="multi-agent", mode="workspace-write")
        tool_registry.register_tool(self._build_review_plan_tool(), source="team", group="multi-agent", mode="workspace-write")
        tool_registry.register_tool(self._build_list_requests_tool(), source="team", group="multi-agent", mode="read-only")

    def spawn(
        self,
        name: str,
        role: str,
        task_prompt: str,
        *,
        model: str | None = None,
        thinking: str | None = None,
    ) -> SpawnResult:
        """创建或复用一个持久 teammate，并立即启动一轮任务。"""

        self._require_lead()
        created = False
        member = self._find_member(name)
        if member is None:
            created = True
            member = TeamMemberState(name=name, role=role, status="idle")
        elif member.status == "working":
            raise RuntimeError(f"Teammate '{name}' is currently working")
        else:
            member.role = role
        handle = self.shared_state.handles.get(name)
        if handle is None:
            handle = self._create_teammate_handle(name=name, role=role, model_id=model, thinking=thinking)
            self.shared_state.handles[name] = handle
        else:
            handle.role = role
        member.role = role
        member.session_id = handle.session.session_id or ""
        member.session_file = handle.session.session_file or ""
        member.parent_session = self.owner_session.session_file or ""
        member.model_id = handle.session.model.id
        member.thinking = str(handle.session.thinking or "")
        if handle.is_running and handle.is_polling:
            member.status = "idle"
        else:
            member.status = "working"
        member.updated_at = time.time()
        self._upsert_member(member)
        handle.pending_shutdown = False
        self.start_or_resume_worker(name, task_prompt)
        return SpawnResult(
            name=name,
            role=role,
            status=member.status,
            created=created,
            session_id=member.session_id,
            session_file=member.session_file,
        )

    def send_message(
        self,
        to_agent: str,
        content: str,
        *,
        message_type: str = "message",
        request_id: str | None = None,
        metadata: dict | None = None,
    ) -> Envelope:
        """以当前 owner 身份向其他成员发送消息。"""

        envelope = self.bus.send(
            self.owner_name,
            to_agent,
            content,
            message_type=message_type,
            request_id=request_id,
            metadata=metadata,
        )
        return envelope

    def list_members(self) -> list[TeamMemberState]:
        """返回团队当前的持久化名册。"""

        return sorted(self._load_members(), key=lambda item: item.name)

    def peek_inbox(self, name: str) -> list[Envelope]:
        """读取指定成员的收件箱摘要，不清空。"""

        return self.bus.peek(name)

    def read_inbox(self, name: str | None = None) -> list[Envelope]:
        """读取并清空某个成员的收件箱。"""

        return self.bus.read_and_drain(name or self.owner_name)

    def shutdown(self, name: str, request_id: str | None = None) -> ProtocolRequest:
        """向指定 worker 发起优雅关停请求。"""

        self._require_lead()
        member = self._find_member(name)
        if member is None:
            raise ValueError(f"Unknown teammate '{name}'")
        request = self.protocols.create_request(
            kind="shutdown",
            sender=self.owner_name,
            recipient=name,
            content="Please shut down gracefully.",
            request_id=request_id,
        )
        self.bus.send(
            self.owner_name,
            name,
            request.content,
            message_type="shutdown_request",
            request_id=request.request_id,
        )
        return request

    def list_protocol_requests(self) -> list[ProtocolRequest]:
        """返回当前团队的协议请求列表。"""

        return self.protocols.list_requests()

    def start_or_resume_worker(self, name: str, prompt: str) -> None:
        """启动一个新的 worker，或向已有 idle worker 投递新任务。"""

        handle = self.shared_state.handles.get(name)
        if handle is None:
            raise ValueError(f"Unknown teammate '{name}'")
        member = self._find_member(name)
        if member is None:
            raise ValueError(f"Unknown teammate '{name}'")
        if handle.is_running:
            self.send_message(
                name,
                prompt,
                message_type="message",
                metadata={"assignment": True, "source": "spawn"},
            )
            return
        handle.start_worker(prompt)

    def drain_inbox_for_worker(self, name: str) -> list[Envelope]:
        """由目标 worker 自己读取并清空其 inbox。"""

        return self.bus.read_and_drain(name)

    def format_idle_resume_prompt(self, name: str, role: str, messages: list[Envelope]) -> str:
        """把 idle 阶段读到的消息统一包装成恢复 prompt。"""

        identity_lines = [
            "<identity>",
            f"You are '{name}', role: {role}, team member in a persistent multi-agent system.",
            "You are resuming work because new inbox messages arrived.",
            "If a shutdown request is present, finish your current reasoning and use respond_shutdown.",
            "</identity>",
        ]
        lines = [*identity_lines, "<team_inbox>"]
        for message in messages:
            lines.append(
                json.dumps(
                    {
                        "type": message.message_type,
                        "from": message.sender,
                        "to": message.recipient,
                        "content": message.content,
                        "request_id": message.request_id,
                        "metadata": message.metadata,
                    },
                    ensure_ascii=False,
                )
            )
        lines.append("</team_inbox>")
        return "\n".join(lines)

    def _create_teammate_handle(
        self,
        *,
        name: str,
        role: str,
        model_id: str | None,
        thinking: str | None,
    ) -> TeammateHandle:
        """创建一个新的 worker 句柄与独立 AgentSession。"""

        from ..agent_session import AgentSession
        from ..resource_loader import ResourceLoader
        from ..session_manager import SessionManager

        base_session = self.owner_session
        session_manager = SessionManager(base_session.session_manager.sessions_dir)
        session_manager.create_session(
            cwd=base_session.cwd,
            model_id=model_id or base_session.model.id,
            title=f"worker:{name}",
            parent_session=base_session.session_file,
        )
        resource_loader = ResourceLoader(
            skills_dir=base_session.resource_loader.skills_dir,
            prompts_dir=base_session.resource_loader.prompts_dir,
            themes_dir=base_session.resource_loader.themes_dir,
            extensions_dir=base_session.resource_loader.extensions_dir,
            workspace_root=base_session.workspace_root,
        )
        resolved_model = base_session.model if not model_id or model_id == base_session.model.id else self.model_registry.get(model_id)
        worker_session = AgentSession(
            workspace_root=base_session.workspace_root,
            cwd=base_session.cwd,
            model=resolved_model,
            thinking=thinking or base_session.thinking,
            settings=base_session.settings,
            session_manager=session_manager,
            resource_loader=resource_loader,
            model_registry=self.model_registry,
            stream_fn=base_session._stream_fn,
        )
        worker_runtime = TeamRuntime(
            owner_session=worker_session,
            workspace_root=self.workspace_root,
            model_registry=self.model_registry,
            owner_name=name,
            owner_role="worker",
            shared_state=self.shared_state,
            idle_poll_interval=self.idle_poll_interval,
        )
        worker_session.attach_team_runtime(worker_runtime)
        worker_session.set_prompt_fragments(
            [
                "\n".join(
                    [
                        "团队身份：",
                        f"- 你的名字: {name}",
                        f"- 你的角色: {role}",
                        "- 你是持久 teammate，不是一次性 subagent。",
                        "- 收到 shutdown_request 时，应先收尾，再调用 respond_shutdown。",
                        "- 执行高风险修改前，应先调用 submit_plan，等待审批结果。",
                        "- 若收件箱带来新消息，应优先处理该消息。",
                    ]
                )
            ]
        )
        return TeammateHandle(name=name, role=role, session=worker_session, runtime=worker_runtime)

    def _set_member_status(self, name: str, status: str) -> None:
        """更新指定成员的状态。"""

        member = self._find_member(name)
        if member is None:
            return
        member.status = status
        member.updated_at = time.time()
        self._upsert_member(member)

    def _mark_member_polled(self, name: str) -> None:
        """记录某个成员最近一次执行 idle 轮询的时间。"""

        member = self._find_member(name)
        if member is None:
            return
        member.last_polled_at = time.time()
        member.updated_at = member.last_polled_at
        self._upsert_member(member)

    def _set_member_error(self, name: str, error: str) -> None:
        """更新指定成员最近一次错误信息。"""

        member = self._find_member(name)
        if member is None:
            return
        member.last_error = error
        member.updated_at = time.time()
        self._upsert_member(member)

    def _require_lead(self) -> None:
        """限制某些操作只能由 lead 发起。"""

        if self.owner_role != "lead":
            raise RuntimeError("This operation is only available to the lead session")

    def _ensure_config(self) -> None:
        """确保团队配置文件存在。"""

        if self.config_path.exists():
            return
        self.config_path.write_text(json.dumps({"team_name": "default", "members": []}, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_config(self) -> dict[str, Any]:
        """加载团队配置文件。"""

        self._ensure_config()
        raw = self.config_path.read_text(encoding="utf-8").strip()
        data = json.loads(raw) if raw else {"team_name": "default", "members": []}
        data.setdefault("team_name", "default")
        data.setdefault("members", [])
        return data

    def _save_config(self, data: dict[str, Any]) -> None:
        """保存团队配置文件。"""

        self.config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_members(self) -> list[TeamMemberState]:
        """从配置文件恢复全部成员。"""

        config = self._load_config()
        members: list[TeamMemberState] = []
        for item in config.get("members", []):
            members.append(
                TeamMemberState(
                    name=str(item.get("name", "")),
                    role=str(item.get("role", "")),
                    status=str(item.get("status", "idle")),
                    session_id=str(item.get("session_id", "")),
                    session_file=str(item.get("session_file", "")),
                    parent_session=str(item.get("parent_session", "")),
                    model_id=str(item.get("model_id", "")),
                    thinking=str(item.get("thinking", "")),
                    last_error=str(item.get("last_error", "")),
                    last_polled_at=float(item.get("last_polled_at", 0.0)),
                    updated_at=float(item.get("updated_at", 0.0)),
                )
            )
        return members

    def _upsert_member(self, member: TeamMemberState) -> None:
        """把成员状态写回配置文件。"""

        config = self._load_config()
        members = [item for item in self._load_members() if item.name != member.name]
        members.append(member)
        config["members"] = [self._member_to_record(item) for item in sorted(members, key=lambda current: current.name)]
        self._save_config(config)

    def _find_member(self, name: str) -> TeamMemberState | None:
        """按名称查找成员。"""

        for member in self._load_members():
            if member.name == name:
                return member
        return None

    @staticmethod
    def _member_to_record(member: TeamMemberState) -> dict[str, Any]:
        """把 TeamMemberState 转成持久化字典。"""

        return {
            "name": member.name,
            "role": member.role,
            "status": member.status,
            "session_id": member.session_id,
            "session_file": member.session_file,
            "parent_session": member.parent_session,
            "model_id": member.model_id,
            "thinking": member.thinking,
            "last_error": member.last_error,
            "last_polled_at": member.last_polled_at,
            "updated_at": member.updated_at,
        }

    def _build_spawn_tool(self) -> AgentTool:
        """构造 spawn_agent 工具。"""

        async def execute(arguments: str, context) -> str:
            payload = json.loads(arguments or "{}")
            result = self.spawn(
                payload["name"],
                payload["role"],
                payload["task_prompt"],
                model=payload.get("model"),
                thinking=payload.get("thinking"),
            )
            return json.dumps(
                {
                    "name": result.name,
                    "role": result.role,
                    "status": result.status,
                    "created": result.created,
                    "session_id": result.session_id,
                },
                ensure_ascii=False,
            )

        return AgentTool(
            name="spawn_agent",
            description="Spawn a persistent teammate that uses its own AgentSession.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                    "task_prompt": {"type": "string"},
                    "model": {"type": "string"},
                    "thinking": {"type": "string"},
                },
                "required": ["name", "role", "task_prompt"],
            },
            execute=execute,
        )

    def _build_send_message_tool(self) -> AgentTool:
        """构造 send_message 工具。"""

        async def execute(arguments: str, context) -> str:
            payload = json.loads(arguments or "{}")
            envelope = self.send_message(
                payload["to"],
                payload["content"],
                message_type=payload.get("message_type", "message"),
                request_id=payload.get("request_id"),
                metadata=dict(payload.get("metadata", {})),
            )
            return json.dumps({"id": envelope.id, "to": envelope.recipient}, ensure_ascii=False)

        return AgentTool(
            name="send_message",
            description="Send a structured message to a teammate inbox.",
            inputSchema={
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "content": {"type": "string"},
                    "message_type": {"type": "string"},
                    "request_id": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                "required": ["to", "content"],
            },
            execute=execute,
        )

    def _build_read_inbox_tool(self) -> AgentTool:
        """构造 read_inbox 工具。"""

        async def execute(arguments: str, context) -> str:
            payload = json.loads(arguments or "{}")
            target = payload.get("name") or self.owner_name
            messages = self.read_inbox(target)
            return json.dumps([asdict(message) for message in messages], ensure_ascii=False)

        return AgentTool(
            name="read_inbox",
            description="Read and drain the current teammate inbox.",
            inputSchema={"type": "object", "properties": {"name": {"type": "string"}}},
            execute=execute,
        )

    def _build_list_team_tool(self) -> AgentTool:
        """构造 list_team 工具。"""

        async def execute(arguments: str, context) -> str:
            _ = arguments, context
            return json.dumps([asdict(member) for member in self.list_members()], ensure_ascii=False)

        return AgentTool(
            name="list_team",
            description="List persistent teammate roster and statuses.",
            inputSchema={"type": "object", "properties": {}},
            execute=execute,
        )

    def _build_shutdown_request_tool(self) -> AgentTool:
        """构造 request_shutdown 工具。"""

        async def execute(arguments: str, context) -> str:
            payload = json.loads(arguments or "{}")
            request = self.shutdown(payload["name"], request_id=payload.get("request_id"))
            return json.dumps(asdict(request), ensure_ascii=False)

        return AgentTool(
            name="request_shutdown",
            description="Ask a teammate to perform graceful shutdown using a request_id protocol.",
            inputSchema={
                "type": "object",
                "properties": {"name": {"type": "string"}, "request_id": {"type": "string"}},
                "required": ["name"],
            },
            execute=execute,
        )

    def _build_shutdown_response_tool(self) -> AgentTool:
        """构造 respond_shutdown 工具。"""

        async def execute(arguments: str, context) -> str:
            payload = json.loads(arguments or "{}")
            request = self.protocols.update_request(
                payload["request_id"],
                status="approved" if payload.get("approve", True) else "rejected",
                response=str(payload.get("reason", "")),
            )
            if payload.get("approve", True):
                handle = self.shared_state.handles.get(self.owner_name)
                if handle is not None:
                    handle.request_shutdown_after_current_turn()
                else:
                    self._set_member_status(self.owner_name, "shutdown")
            self.bus.send(
                self.owner_name,
                request.sender,
                str(payload.get("reason", "")),
                message_type="shutdown_response",
                request_id=request.request_id,
                metadata={"approve": bool(payload.get("approve", True))},
            )
            return json.dumps(asdict(request), ensure_ascii=False)

        return AgentTool(
            name="respond_shutdown",
            description="Respond to a graceful shutdown request.",
            inputSchema={
                "type": "object",
                "properties": {
                    "request_id": {"type": "string"},
                    "approve": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["request_id", "approve"],
            },
            execute=execute,
        )

    def _build_submit_plan_tool(self) -> AgentTool:
        """构造 submit_plan 工具。"""

        async def execute(arguments: str, context) -> str:
            payload = json.loads(arguments or "{}")
            request = self.protocols.create_request(
                kind="plan",
                sender=self.owner_name,
                recipient=payload.get("to", "lead"),
                content=payload["plan"],
            )
            self.bus.send(
                self.owner_name,
                request.recipient,
                request.content,
                message_type="plan_submit",
                request_id=request.request_id,
            )
            return json.dumps(asdict(request), ensure_ascii=False)

        return AgentTool(
            name="submit_plan",
            description="Submit a plan for approval before a risky change.",
            inputSchema={
                "type": "object",
                "properties": {"plan": {"type": "string"}, "to": {"type": "string"}},
                "required": ["plan"],
            },
            execute=execute,
        )

    def _build_review_plan_tool(self) -> AgentTool:
        """构造 review_plan 工具。"""

        async def execute(arguments: str, context) -> str:
            self._require_lead()
            payload = json.loads(arguments or "{}")
            request = self.protocols.update_request(
                payload["request_id"],
                status="approved" if payload.get("approve", True) else "rejected",
                response=str(payload.get("feedback", "")),
            )
            self.bus.send(
                self.owner_name,
                request.recipient,
                str(payload.get("feedback", "")),
                message_type="plan_approval_response",
                request_id=request.request_id,
                metadata={"approve": bool(payload.get("approve", True))},
            )
            return json.dumps(asdict(request), ensure_ascii=False)

        return AgentTool(
            name="review_plan",
            description="Approve or reject a previously submitted plan.",
            inputSchema={
                "type": "object",
                "properties": {
                    "request_id": {"type": "string"},
                    "approve": {"type": "boolean"},
                    "feedback": {"type": "string"},
                },
                "required": ["request_id", "approve"],
            },
            execute=execute,
        )

    def _build_list_requests_tool(self) -> AgentTool:
        """构造 list_protocol_requests 工具。"""

        async def execute(arguments: str, context) -> str:
            _ = arguments, context
            return json.dumps([asdict(request) for request in self.list_protocol_requests()], ensure_ascii=False)

        return AgentTool(
            name="list_protocol_requests",
            description="List team protocol requests tracked by request_id.",
            inputSchema={"type": "object", "properties": {}},
            execute=execute,
        )
