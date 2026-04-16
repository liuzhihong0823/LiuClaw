from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ai import UserMessage
from coding_agent.core.session_manager import SessionManager

from .types import SessionRef


def _format_log_entry_for_agent(entry: dict) -> str:
    """把日志中的历史消息转换成适合注入 Agent 上下文的文本格式。"""
    created_at = str(entry.get("created_at") or "")
    sender = str(entry.get("sender_name") or entry.get("sender_id") or "unknown")
    prefix = ""
    if created_at:
        try:
            stamp = datetime.fromisoformat(created_at.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S")
            prefix = f"[{stamp}] "
        except ValueError:
            prefix = ""
    body = f"{prefix}[{sender}]: {entry.get('text', '')}".strip()
    attachments = entry.get("attachments") or []
    if attachments:
        lines = ["", "<attachments>"]
        for attachment in attachments:
            local_path = attachment.get("local_path") or attachment.get("local") or ""
            original = attachment.get("original_name") or attachment.get("original") or "attachment"
            lines.append(f"- {original}: {local_path}")
        lines.append("</attachments>")
        body = "\n".join([body, *lines])
    return body


def sync_channel_log_to_session(
    session_manager: SessionManager,
    session_ref: SessionRef,
    channel_dir: Path,
    exclude_message_id: str | None = None,
) -> int:
    """把频道日志里的用户消息同步到会话树，避免 Agent 丢失历史上下文。"""
    log_file = channel_dir / "log.jsonl"
    if not log_file.exists():
        return 0

    synced = set(session_ref.synced_message_ids)
    inserted = 0
    parent_id = session_ref.leaf_id
    snapshot = session_manager.load_session(session_ref.session_file)
    if snapshot.leaf_id:
        parent_id = snapshot.leaf_id

    with log_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = __import__("json").loads(line)
            message_id = str(entry.get("message_id") or "")
            if not message_id:
                continue
            if message_id == exclude_message_id:
                continue
            if entry.get("is_bot"):
                continue
            if message_id in synced:
                continue
            node = session_manager.append_message(
                session_ref.session_file,
                message=UserMessage(content=_format_log_entry_for_agent(entry), metadata={"synced_from_log": True, "message_id": message_id}),
                parent_id=parent_id,
            )
            parent_id = node.id
            session_ref.leaf_id = node.id
            session_ref.session_id = snapshot.session_id
            session_ref.synced_message_ids.append(message_id)
            synced.add(message_id)
            inserted += 1
    return inserted
