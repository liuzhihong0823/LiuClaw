from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import ChatEvent


@dataclass(slots=True)
class EventRecord:
    """事件文件中定义的通用事件记录结构。"""

    name: str  # 事件名称。
    data: dict[str, Any]  # 事件载荷数据。


class EventsWatcher:
    def __init__(self, events_dir: Path, dispatch) -> None:
        """初始化事件目录监听器，按轮询方式消费本地事件文件。"""
        self.events_dir = events_dir
        self.dispatch = dispatch
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def scan_once(self) -> None:
        """扫描一次事件目录，并把到期事件转发给上层处理器。"""
        self.events_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.events_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            event = self._build_chat_event(path, payload)
            if event is None:
                continue
            await self.dispatch(event)
            if payload.get("type") in {"immediate", "one-shot"}:
                path.unlink(missing_ok=True)

    def _build_chat_event(self, path: Path, payload: dict[str, Any]) -> ChatEvent | None:
        """把事件文件内容转换成内部 ChatEvent，并处理触发时机。"""
        event_type = payload.get("type")
        if event_type not in {"immediate", "one-shot", "periodic"}:
            return None
        if event_type == "one-shot":
            at_value = payload.get("at")
            if not at_value:
                return None
            at = datetime.fromisoformat(str(at_value).replace("Z", "+00:00"))
            if at > datetime.now(timezone.utc):
                return None
        if event_type == "periodic":
            interval = int(payload.get("interval_seconds", 0) or 0)
            last_run = payload.get("last_run")
            if interval <= 0:
                return None
            if last_run:
                last_dt = datetime.fromisoformat(str(last_run).replace("Z", "+00:00"))
                elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds()
                if elapsed < interval:
                    return None
            payload["last_run"] = datetime.now(timezone.utc).isoformat()
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return ChatEvent(
            platform="feishu",
            chat_id=str(payload["channelId"]),
            message_id=f"event:{path.stem}",
            sender_id="system",
            sender_name="system",
            text=str(payload["text"]),
            is_direct=False,
            is_trigger=True,
            metadata={"synthetic": True, "event_type": event_type, "event_file": path.name},
        )

    async def run(self, interval_seconds: float = 1.0) -> None:
        """持续轮询事件目录，直到收到停止信号。"""
        while not self._stop.is_set():
            await self.scan_once()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval_seconds)
            except asyncio.TimeoutError:
                continue

    def start(self, interval_seconds: float = 1.0) -> asyncio.Task:
        """启动后台轮询任务。"""
        self._stop.clear()
        self._task = asyncio.create_task(self.run(interval_seconds=interval_seconds))
        return self._task

    async def stop(self) -> None:
        """停止后台轮询任务并等待退出。"""
        self._stop.set()
        if self._task is not None:
            await self._task
