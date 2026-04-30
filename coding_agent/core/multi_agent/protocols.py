from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from .types import ProtocolKind, ProtocolRequest


class ProtocolTracker:
    """负责用 request_id 持久化追踪协议状态。"""

    def __init__(self, path: Path) -> None:
        """初始化协议状态文件。"""

        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._save({})

    def create_request(
        self,
        *,
        kind: ProtocolKind,
        sender: str,
        recipient: str,
        content: str = "",
        metadata: dict | None = None,
        request_id: str | None = None,
    ) -> ProtocolRequest:
        """创建一个新的协议请求并落盘。"""

        now = time.time()
        request = ProtocolRequest(
            request_id=request_id or uuid.uuid4().hex[:8],
            kind=kind,
            status="pending",
            sender=sender,
            recipient=recipient,
            content=content,
            created_at=now,
            updated_at=now,
            metadata=dict(metadata or {}),
        )
        data = self._load()
        data[request.request_id] = self._to_record(request)
        self._save(data)
        return request

    def update_request(
        self,
        request_id: str,
        *,
        status: str,
        response: str = "",
        metadata: dict | None = None,
    ) -> ProtocolRequest:
        """更新某个协议请求的状态。"""

        data = self._load()
        if request_id not in data:
            raise KeyError(f"Unknown request_id '{request_id}'")
        record = dict(data[request_id])
        record["status"] = status
        record["response"] = response
        record["updated_at"] = time.time()
        if metadata:
            merged = dict(record.get("metadata", {}))
            merged.update(metadata)
            record["metadata"] = merged
        data[request_id] = record
        self._save(data)
        return self._from_record(record)

    def get_request(self, request_id: str) -> ProtocolRequest | None:
        """按 request_id 读取单个协议请求。"""

        record = self._load().get(request_id)
        if record is None:
            return None
        return self._from_record(record)

    def list_requests(self) -> list[ProtocolRequest]:
        """返回全部协议请求，按最近更新时间倒序。"""

        requests = [self._from_record(record) for record in self._load().values()]
        return sorted(requests, key=lambda item: item.updated_at, reverse=True)

    def _load(self) -> dict[str, dict]:
        """加载协议状态文件。"""

        raw = self.path.read_text(encoding="utf-8").strip()
        if not raw:
            return {}
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("protocols.json must contain an object")
        return {str(key): dict(value) for key, value in data.items()}

    def _save(self, data: dict[str, dict]) -> None:
        """保存协议状态文件。"""

        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _to_record(request: ProtocolRequest) -> dict:
        """把 ProtocolRequest 转成持久化字典。"""

        return {
            "request_id": request.request_id,
            "kind": request.kind,
            "status": request.status,
            "sender": request.sender,
            "recipient": request.recipient,
            "content": request.content,
            "created_at": request.created_at,
            "updated_at": request.updated_at,
            "response": request.response,
            "metadata": request.metadata,
        }

    @staticmethod
    def _from_record(record: dict) -> ProtocolRequest:
        """把持久化记录恢复成 ProtocolRequest。"""

        return ProtocolRequest(
            request_id=str(record.get("request_id", "")),
            kind=str(record.get("kind", "plan")),
            status=str(record.get("status", "pending")),
            sender=str(record.get("sender", "")),
            recipient=str(record.get("recipient", "")),
            content=str(record.get("content", "")),
            created_at=float(record.get("created_at", 0.0)),
            updated_at=float(record.get("updated_at", 0.0)),
            response=str(record.get("response", "")),
            metadata=dict(record.get("metadata", {})),
        )
