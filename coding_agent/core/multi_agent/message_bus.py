from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from .types import Envelope


class MessageBus:
    """提供 append-only JSONL 收件箱读写能力。"""

    def __init__(self, inbox_dir: Path) -> None:
        """初始化收件箱目录。"""

        self.inbox_dir = inbox_dir  # 每个成员一个 jsonl 文件。
        self.inbox_dir.mkdir(parents=True, exist_ok=True)

    def send(
        self,
        sender: str,
        recipient: str,
        content: str,
        *,
        message_type: str = "message",
        request_id: str | None = None,
        metadata: dict | None = None,
    ) -> Envelope:
        """向指定成员的收件箱追加一条消息。"""

        envelope = Envelope(
            id=uuid.uuid4().hex[:12],
            sender=sender,
            recipient=recipient,
            message_type=message_type,
            content=content,
            timestamp=time.time(),
            request_id=request_id,
            metadata=dict(metadata or {}),
        )
        path = self._path_for(recipient)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self._to_record(envelope), ensure_ascii=False) + "\n")
        return envelope

    def peek(self, recipient: str) -> list[Envelope]:
        """只读取收件箱内容，不执行清空。"""

        path = self._path_for(recipient)
        if not path.exists():
            return []
        messages: list[Envelope] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            messages.append(self._from_record(json.loads(raw)))
        return messages

    def read_and_drain(self, recipient: str) -> list[Envelope]:
        """读取指定收件箱并立即清空。"""

        messages = self.peek(recipient)
        self._path_for(recipient).write_text("", encoding="utf-8")
        return messages

    def _path_for(self, recipient: str) -> Path:
        """返回某个成员对应的 inbox 文件路径。"""

        return self.inbox_dir / f"{recipient}.jsonl"

    @staticmethod
    def _to_record(envelope: Envelope) -> dict:
        """把 Envelope 转成可持久化字典。"""

        return {
            "id": envelope.id,
            "sender": envelope.sender,
            "recipient": envelope.recipient,
            "message_type": envelope.message_type,
            "content": envelope.content,
            "timestamp": envelope.timestamp,
            "request_id": envelope.request_id,
            "metadata": envelope.metadata,
        }

    @staticmethod
    def _from_record(record: dict) -> Envelope:
        """把持久化记录恢复成 Envelope。"""

        return Envelope(
            id=str(record.get("id", "")),
            sender=str(record.get("sender", "")),
            recipient=str(record.get("recipient", "")),
            message_type=str(record.get("message_type", "message")),
            content=str(record.get("content", "")),
            timestamp=float(record.get("timestamp", 0.0)),
            request_id=record.get("request_id"),
            metadata=dict(record.get("metadata", {})),
        )
