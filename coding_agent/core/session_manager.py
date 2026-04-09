from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai.types import AssistantMessage, ConversationMessage, ToolCall, ToolResultMessage, UserMessage

from .types import ControlMessage, PersistedMessageNode, SessionSnapshot, conversation_to_node_payload


class SessionManager:
    """负责会话元信息、事件流与上下文恢复。"""

    def __init__(self, sessions_dir: Path) -> None:
        """初始化会话根目录。"""

        self.sessions_dir = sessions_dir  # 会话数据根目录。
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self, *, cwd: Path, model_id: str, title: str | None = None) -> SessionSnapshot:
        """创建新会话并初始化元信息与事件文件。"""

        session_id = uuid.uuid4().hex[:12]
        branch_id = "main"
        now = datetime.now(UTC).isoformat()
        snapshot = SessionSnapshot(session_id=session_id, branch_id=branch_id, cwd=cwd, model_id=model_id)
        self._write_meta(
            session_id,
            {
                "session_id": session_id,
                "title": title or session_id,
                "current_branch": branch_id,
                "cwd": str(cwd),
                "model_id": model_id,
                "created_at": now,
                "updated_at": now,
            },
        )
        self._ensure_events_file(session_id)
        return snapshot

    def load_session(self, session_id: str) -> SessionSnapshot:
        """通过回放事件流恢复指定会话的快照。"""

        meta = self._read_meta(session_id)
        snapshot = SessionSnapshot(
            session_id=session_id,
            branch_id=str(meta.get("current_branch", "main")),
            cwd=Path(str(meta["cwd"])),
            model_id=str(meta["model_id"]),
        )
        for event in self.iter_events(session_id):
            if event["type"] == "message":
                snapshot.nodes.append(PersistedMessageNode(**event["payload"]))
            elif event["type"] == "control":
                snapshot.nodes.append(
                    PersistedMessageNode(
                        id=str(event["payload"]["id"]),
                        role="control",
                        content=str(event["payload"]["content"]),
                        parent_id=event["payload"].get("parent_id"),
                        branch_id=str(event["payload"]["branch_id"]),
                        message_type="control",
                        metadata=dict(event["payload"].get("metadata", {})),
                    )
                )
            elif event["type"] == "summary":
                snapshot.summaries.append(event["payload"])
            elif event["type"] == "branch_switch":
                snapshot.branch_id = str(event["payload"]["to_branch"])
        return snapshot

    def append_message(
        self,
        session_id: str,
        *,
        message: ConversationMessage,
        branch_id: str,
        parent_id: str | None,
        node_id: str | None = None,
    ) -> PersistedMessageNode:
        """向会话事件流追加一条消息节点。"""

        node = PersistedMessageNode(
            id=node_id or uuid.uuid4().hex[:12],
            parent_id=parent_id,
            branch_id=branch_id,
            **conversation_to_node_payload(message),
        )
        self._append_event(session_id, {"type": "message", "payload": asdict(node)})
        self._touch_meta(session_id, message=message)
        return node

    def append_summary(
        self,
        session_id: str,
        *,
        branch_id: str,
        summary: str,
        node_ids: list[str],
    ) -> None:
        """向会话事件流追加一条摘要事件。"""

        self._append_event(
            session_id,
            {"type": "summary", "payload": {"branch_id": branch_id, "summary": summary, "node_ids": node_ids}},
        )
        self._touch_meta(session_id)

    def append_control(
        self,
        session_id: str,
        *,
        message: ControlMessage,
        branch_id: str,
        parent_id: str | None,
        node_id: str | None = None,
    ) -> PersistedMessageNode:
        """追加一条显式控制消息事件，不混入普通业务消息流。"""

        node = PersistedMessageNode(
            id=node_id or uuid.uuid4().hex[:12],
            role="control",
            content=message.content,
            parent_id=parent_id,
            branch_id=branch_id,
            message_type="control",
            metadata={"control_kind": message.kind, **dict(message.metadata)},
        )
        self._append_event(
            session_id,
            {
                "type": "control",
                "payload": {
                    "id": node.id,
                    "content": node.content,
                    "parent_id": parent_id,
                    "branch_id": branch_id,
                    "metadata": dict(node.metadata),
                },
            },
        )
        self._touch_meta(session_id)
        return node

    def switch_branch(self, session_id: str, from_branch: str, to_branch: str) -> None:
        """记录分支切换事件，并更新当前分支元信息。"""

        self._append_event(
            session_id,
            {"type": "branch_switch", "payload": {"from_branch": from_branch, "to_branch": to_branch}},
        )
        meta = self._read_meta(session_id)
        meta["current_branch"] = to_branch
        self._write_meta(session_id, meta)

    def list_recent_sessions(self, *, limit: int = 10, cwd: Path | None = None) -> list[dict[str, Any]]:
        """列出最近使用的会话摘要，用于交互层恢复与补全。"""

        items: list[dict[str, Any]] = []
        if not self.sessions_dir.exists():
            return items
        for session_dir in self.sessions_dir.iterdir():
            meta_file = session_dir / "meta.json"
            if not meta_file.exists():
                continue
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if cwd is not None and Path(str(meta.get("cwd", ""))).resolve() != cwd.resolve():
                continue
            items.append(meta)
        items.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return items[:limit]

    def build_context_messages(self, session_id: str, branch_id: str | None = None) -> list[ConversationMessage]:
        """构建发送给模型的上下文消息列表。"""

        snapshot = self.load_session(session_id)
        active_branch = branch_id or snapshot.branch_id
        messages: list[ConversationMessage] = []
        summary = self.latest_summary(snapshot, active_branch)
        summarized_node_ids = set(summary.get("node_ids", [])) if summary else set()
        if summary:
            messages.append(
                UserMessage(
                    content=f"[会话摘要]\n{summary['summary']}",
                    metadata={"summary": True, "branch_id": active_branch},
                )
            )
        for node in snapshot.nodes:
            if node.branch_id != active_branch:
                continue
            if node.role == "control":
                continue
            if node.id in summarized_node_ids:
                continue
            messages.append(self._node_to_message(node))
        return messages

    def latest_summary(self, snapshot: SessionSnapshot, branch_id: str) -> dict[str, Any] | None:
        """获取某个分支最近一次生成的摘要。"""

        for item in reversed(snapshot.summaries):
            if item["branch_id"] == branch_id:
                return item
        return None

    def iter_events(self, session_id: str):
        """返回指定会话的全部事件记录。"""

        events_file = self._events_file(session_id)
        if not events_file.exists():
            return []
        with events_file.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def _ensure_events_file(self, session_id: str) -> None:
        """确保会话事件文件已存在。"""

        events = self._events_file(session_id)
        events.parent.mkdir(parents=True, exist_ok=True)
        if not events.exists():
            events.write_text("", encoding="utf-8")

    def _append_event(self, session_id: str, event: dict[str, Any]) -> None:
        """以 JSONL 形式向会话事件文件追加一条事件。"""

        self._ensure_events_file(session_id)
        with self._events_file(session_id).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _session_dir(self, session_id: str) -> Path:
        """返回会话对应的目录路径。"""

        return self.sessions_dir / session_id

    def _events_file(self, session_id: str) -> Path:
        """返回会话事件文件路径。"""

        return self._session_dir(session_id) / "events.jsonl"

    def _meta_file(self, session_id: str) -> Path:
        """返回会话元信息文件路径。"""

        return self._session_dir(session_id) / "meta.json"

    def _write_meta(self, session_id: str, data: dict[str, Any]) -> None:
        """写入会话元信息文件。"""

        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        self._meta_file(session_id).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _read_meta(self, session_id: str) -> dict[str, Any]:
        """读取会话元信息文件。"""

        return json.loads(self._meta_file(session_id).read_text(encoding="utf-8"))

    def _touch_meta(self, session_id: str, message: ConversationMessage | None = None) -> None:
        """在会话活跃时刷新元信息中的更新时间与标题。"""

        meta = self._read_meta(session_id)
        meta["updated_at"] = datetime.now(UTC).isoformat()
        if message is not None and isinstance(message, UserMessage) and meta.get("title") == session_id:
            text = message.content.strip().replace("\n", " ")
            meta["title"] = text[:40] if text else session_id
        if message is not None and hasattr(message, "role") and getattr(message, "role", "") != "tool":
            meta["model_id"] = meta.get("model_id", meta.get("model"))
        self._write_meta(session_id, meta)

    @staticmethod
    def _node_to_message(node: PersistedMessageNode) -> ConversationMessage:
        """把持久化节点恢复成统一消息对象。"""

        if node.role == "user":
            return UserMessage(content=node.content, metadata=dict(node.metadata))
        if node.role == "assistant":
            return AssistantMessage(
                content=node.content,
                thinking=node.thinking,
                toolCalls=[
                    ToolCall(
                        id=str(item["id"]),
                        name=str(item["name"]),
                        arguments=str(item.get("arguments", "")),
                        metadata=dict(item.get("metadata", {})),
                    )
                    for item in node.tool_calls
                ],
                metadata=dict(node.metadata),
            )
        return ToolResultMessage(
            toolCallId=node.tool_call_id or "",
            toolName=node.tool_name or "",
            content=node.content,
            metadata=dict(node.metadata),
        )
