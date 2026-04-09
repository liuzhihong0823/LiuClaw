from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from coding_agent.core.session_manager import SessionManager

from .types import ChatAttachment, ChatEvent, LoggedChatMessage, MomPaths, SessionRef


def build_mom_paths(workspace_root: Path) -> MomPaths:
    """根据工作区根目录构造 mom 使用的所有路径。"""
    root = workspace_root / ".mom"
    return MomPaths(
        workspace_root=workspace_root,
        root=root,
        channels_dir=root / "channels",
        events_dir=root / "events",
        sessions_dir=root / "sessions",
        settings_file=root / "settings.json",
        channel_index_file=root / "channel_index.json",
    )


class MomStore:
    def __init__(self, workspace_root: Path) -> None:
        """初始化 mom 存储层，并确保工作目录结构存在。"""
        self.workspace_root = workspace_root.resolve()  # 规范化后的工作区根目录。
        self.paths = build_mom_paths(self.workspace_root)  # mom 使用的路径集合。
        self.paths.ensure_exists()

    def channel_dir(self, chat_id: str) -> Path:
        """返回频道目录，不存在时自动创建。"""
        path = self.paths.channels_dir / chat_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def channel_log_path(self, chat_id: str) -> Path:
        """返回频道日志文件路径。"""
        return self.channel_dir(chat_id) / "log.jsonl"

    def channel_memory_path(self, chat_id: str) -> Path:
        """返回频道记忆文件路径，不存在时创建空文件。"""
        path = self.channel_dir(chat_id) / "MEMORY.md"
        if not path.exists():
            path.write_text("", encoding="utf-8")
        return path

    def attachments_dir(self, chat_id: str) -> Path:
        """返回频道附件目录。"""
        path = self.channel_dir(chat_id) / "attachments"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def scratch_dir(self, chat_id: str) -> Path:
        """返回频道临时工作目录。"""
        path = self.channel_dir(chat_id) / "scratch"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def sessions_manager(self) -> SessionManager:
        """创建会话管理器，用于访问 mom 的会话存储。"""
        return SessionManager(self.paths.sessions_dir)

    def load_channel_index(self) -> dict[str, Any]:
        """读取频道到会话引用的索引文件。"""
        raw = self.paths.channel_index_file.read_text(encoding="utf-8").strip()
        if not raw:
            return {}
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}

    def load_settings(self) -> dict[str, Any]:
        """读取 mom 的本地设置文件。"""
        raw = self.paths.settings_file.read_text(encoding="utf-8").strip()
        if not raw:
            return {}
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}

    def save_channel_index(self, index: dict[str, Any]) -> None:
        """持久化频道索引信息。"""
        self.paths.channel_index_file.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def get_or_create_session_ref(self, chat_id: str, session_manager: SessionManager, model_id: str) -> SessionRef:
        """获取频道对应会话；若不存在则创建一条新的 Agent 会话引用。"""
        index = self.load_channel_index()
        item = index.get(chat_id)
        if isinstance(item, dict) and item.get("session_id"):
            return SessionRef(
                session_id=str(item["session_id"]),
                branch_id=str(item.get("branch_id", "main")),
                synced_message_ids=[str(value) for value in item.get("synced_message_ids", [])],
            )
        snapshot = session_manager.create_session(cwd=self.channel_dir(chat_id), model_id=model_id, title=f"mom:{chat_id}")
        ref = SessionRef(session_id=snapshot.session_id, branch_id=snapshot.branch_id, synced_message_ids=[])
        self.save_session_ref(chat_id, ref)
        return ref

    def save_session_ref(self, chat_id: str, ref: SessionRef) -> None:
        """保存频道与会话之间的引用关系。"""
        index = self.load_channel_index()
        index[chat_id] = {
            "session_id": ref.session_id,
            "branch_id": ref.branch_id,
            "synced_message_ids": list(dict.fromkeys(ref.synced_message_ids)),
        }
        self.save_channel_index(index)

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """清洗附件文件名，避免出现非法或危险字符。"""
        sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", filename).strip("._")
        return sanitized or "attachment"

    def register_attachment(self, chat_id: str, message_id: str, attachment: ChatAttachment) -> ChatAttachment:
        """为附件分配本地落盘路径，并回填到附件对象中。"""
        suffix = self.sanitize_filename(attachment.original_name)
        target = self.attachments_dir(chat_id) / f"{message_id}_{suffix}"
        attachment.local_path = str(target.relative_to(self.workspace_root))
        attachment.message_id = message_id
        return attachment

    def write_attachment_bytes(self, attachment: ChatAttachment, payload: bytes) -> Path:
        """把附件二进制内容写入本地文件。"""
        target = self.workspace_root / attachment.local_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return target

    def append_log(self, chat_id: str, entry: LoggedChatMessage) -> None:
        """向频道日志追加一条 JSONL 记录。"""
        path = self.channel_log_path(chat_id)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

    def has_logged_message(self, chat_id: str, message_id: str, *, is_bot: bool | None = None) -> bool:
        """检查某条消息是否已经写入日志，可按机器人/用户消息过滤。"""
        if not message_id:
            return False
        for entry in reversed(self.read_log_entries(chat_id)):
            if str(entry.get("message_id") or "") != message_id:
                continue
            if is_bot is None:
                return True
            return bool(entry.get("is_bot")) is is_bot
        return False

    def log_event(self, event: ChatEvent) -> None:
        """记录用户侧事件到频道日志，并自动跳过重复消息。"""
        if self.has_logged_message(event.chat_id, event.message_id, is_bot=False):
            return
        self.append_log(
            event.chat_id,
            LoggedChatMessage(
                platform=event.platform,
                chat_id=event.chat_id,
                message_id=event.message_id,
                sender_id=event.sender_id,
                sender_name=event.sender_name,
                text=event.text,
                is_bot=False,
                attachments=[asdict(item) for item in event.attachments],
                direct=event.is_direct,
                trigger=event.is_trigger,
                metadata=dict(event.metadata),
            ),
        )

    def log_bot_message(
        self,
        chat_id: str,
        *,
        message_id: str,
        text: str,
        response_kind: str = "message",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """记录机器人回复到频道日志。"""
        self.append_log(
            chat_id,
            LoggedChatMessage(
                platform="feishu",
                chat_id=chat_id,
                message_id=message_id,
                sender_id="bot",
                sender_name="mom",
                text=text,
                is_bot=True,
                response_kind=response_kind,
                metadata=dict(metadata or {}),
            ),
        )

    def read_log_entries(self, chat_id: str) -> list[dict[str, Any]]:
        """读取频道完整日志并返回解析后的记录列表。"""
        path = self.channel_log_path(chat_id)
        if not path.exists():
            return []
        items: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items
