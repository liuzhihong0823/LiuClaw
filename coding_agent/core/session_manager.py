from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai.types import AssistantMessage, ConversationMessage, UserMessage

from .types import (
    BranchSummaryEntry,
    CompactionEntry,
    CustomEntry,
    CustomMessageEntry,
    LabelEntry,
    ModelChangeEntry,
    SessionConversationContext,
    SessionEntry,
    SessionHeader,
    SessionInfo,
    SessionInfoEntry,
    SessionMessageEntry,
    SessionSnapshot,
    SessionTreeNode,
    ThinkingLevelChangeEntry,
    deserialize_message,
    serialize_message,
)

CURRENT_SESSION_VERSION = 1


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_session_dir_name(cwd: Path) -> str:
    raw = str(cwd.resolve()).lstrip("/\\")
    return f"--{raw.replace('/', '-').replace('\\', '-').replace(':', '-')}--"


class SessionManager:
    """树状、追加式的会话存储管理器。"""

    def __init__(self, sessions_dir: Path) -> None:
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self, *, cwd: Path, model_id: str, title: str | None = None, parent_session: str | None = None) -> SessionSnapshot:
        session_id = uuid.uuid4().hex[:12]
        session_dir = self._session_dir_for_cwd(cwd)
        session_dir.mkdir(parents=True, exist_ok=True)
        session_file = session_dir / f"{datetime.now(UTC).strftime('%Y-%m-%dT%H-%M-%S-%fZ')}_{session_id}.jsonl"
        header = SessionHeader(
            version=CURRENT_SESSION_VERSION,
            id=session_id,
            timestamp=_iso_now(),
            cwd=str(cwd.resolve()),
            parent_session=parent_session,
            title=title or session_id,
            model_id=model_id,
        )
        session_file.write_text(json.dumps(self._header_to_record(header), ensure_ascii=False) + "\n", encoding="utf-8")
        return SessionSnapshot(
            session_id=session_id,
            session_file=session_file,
            cwd=cwd.resolve(),
            model_id=model_id,
            leaf_id=None,
            entries=[],
            header=header,
            title=header.title,
        )

    def continue_recent(self, cwd: Path) -> SessionSnapshot:
        recent = self.list_recent_sessions(limit=1, cwd=cwd)
        if recent:
            return self.load_session(recent[0]["session_file"])
        return self.create_session(cwd=cwd, model_id="")

    def open(self, session_file: str | Path) -> SessionSnapshot:
        return self.load_session(session_file)

    def load_session(self, session_ref: str | Path) -> SessionSnapshot:
        session_file = self.resolve_session_file(session_ref)
        if session_file is None:
            raise FileNotFoundError(f"Unknown session reference: {session_ref}")

        records = self._load_records(session_file)
        if not records:
            raise ValueError(f"Invalid session file: {session_file}")
        header = self._record_to_header(records[0])
        entries: list[SessionEntry] = [self._record_to_entry(item) for item in records[1:]]
        leaf_id = entries[-1].id if entries else None
        return SessionSnapshot(
            session_id=header.id,
            session_file=session_file,
            cwd=Path(header.cwd).resolve(),
            model_id=header.model_id,
            leaf_id=leaf_id,
            entries=entries,
            header=header,
            title=header.title,
        )

    def resolve_session_file(self, session_ref: str | Path | None) -> Path | None:
        if session_ref is None:
            return None
        ref = Path(str(session_ref))
        if ref.suffix == ".jsonl" and ref.exists():
            return ref.resolve()
        if ref.exists() and ref.is_file():
            return ref.resolve()
        candidate = self._find_session_file_by_id(str(session_ref))
        if candidate is not None:
            return candidate
        legacy_dir = self.sessions_dir / str(session_ref)
        if legacy_dir.is_dir():
            return self._migrate_legacy_session(legacy_dir)
        return None

    def list_recent_sessions(self, *, limit: int = 10, cwd: Path | None = None) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        self._migrate_all_legacy_sessions()
        for session_file in self._iter_session_files(cwd=cwd):
            info = self._build_session_info(session_file)
            if info is None:
                continue
            items.append(
                {
                    "session_id": info.id,
                    "session_file": info.path,
                    "leaf_id": info.leaf_id,
                    "cwd": info.cwd,
                    "title": info.title or info.name or info.first_message,
                    "model_id": info.model_id,
                    "updated_at": info.modified_at,
                    "created_at": info.created_at,
                    "message_count": info.message_count,
                }
            )
        items.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return items[:limit]

    def append_message(self, session_ref: str | Path, *, message: ConversationMessage, parent_id: str | None = None) -> SessionMessageEntry:
        snapshot = self.load_session(session_ref)
        entry = SessionMessageEntry(
            id=self._generate_entry_id(snapshot.entries),
            parent_id=parent_id if parent_id is not None else snapshot.leaf_id,
            timestamp=_iso_now(),
            message=message,
        )
        self._append_record(snapshot.session_file, self._entry_to_record(entry))
        if isinstance(message, UserMessage):
            self._update_header(snapshot.session_file, title=str(message.content).strip().replace("\n", " ")[:80] or snapshot.title)
        return entry

    def append_thinking_level_change(self, session_ref: str | Path, thinking_level: str) -> ThinkingLevelChangeEntry:
        snapshot = self.load_session(session_ref)
        entry = ThinkingLevelChangeEntry(
            id=self._generate_entry_id(snapshot.entries),
            parent_id=snapshot.leaf_id,
            timestamp=_iso_now(),
            thinking_level=thinking_level,
        )
        self._append_record(snapshot.session_file, self._entry_to_record(entry))
        return entry

    def append_model_change(self, session_ref: str | Path, provider: str, model_id: str) -> ModelChangeEntry:
        snapshot = self.load_session(session_ref)
        entry = ModelChangeEntry(
            id=self._generate_entry_id(snapshot.entries),
            parent_id=snapshot.leaf_id,
            timestamp=_iso_now(),
            provider=provider,
            model_id=model_id,
        )
        self._append_record(snapshot.session_file, self._entry_to_record(entry))
        self._update_header(snapshot.session_file, model_id=model_id)
        return entry

    def append_compaction(
        self,
        session_ref: str | Path,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int,
        details: dict[str, Any] | None = None,
        from_hook: bool = False,
    ) -> CompactionEntry:
        snapshot = self.load_session(session_ref)
        entry = CompactionEntry(
            id=self._generate_entry_id(snapshot.entries),
            parent_id=snapshot.leaf_id,
            timestamp=_iso_now(),
            summary=summary,
            first_kept_entry_id=first_kept_entry_id,
            tokens_before=tokens_before,
            details=details,
            from_hook=from_hook,
        )
        self._append_record(snapshot.session_file, self._entry_to_record(entry))
        return entry

    def append_branch_summary(
        self,
        session_ref: str | Path,
        from_id: str,
        summary: str,
        details: dict[str, Any] | None = None,
        from_hook: bool = False,
    ) -> BranchSummaryEntry:
        snapshot = self.load_session(session_ref)
        entry = BranchSummaryEntry(
            id=self._generate_entry_id(snapshot.entries),
            parent_id=snapshot.leaf_id,
            timestamp=_iso_now(),
            from_id=from_id,
            summary=summary,
            details=details,
            from_hook=from_hook,
        )
        self._append_record(snapshot.session_file, self._entry_to_record(entry))
        return entry

    def append_custom_entry(self, session_ref: str | Path, custom_type: str, data: Any | None = None) -> CustomEntry:
        snapshot = self.load_session(session_ref)
        entry = CustomEntry(
            id=self._generate_entry_id(snapshot.entries),
            parent_id=snapshot.leaf_id,
            timestamp=_iso_now(),
            custom_type=custom_type,
            data=data,
        )
        self._append_record(snapshot.session_file, self._entry_to_record(entry))
        return entry

    def append_custom_message_entry(
        self,
        session_ref: str | Path,
        custom_type: str,
        content: str | list[dict[str, Any]],
        display: bool,
        details: dict[str, Any] | None = None,
    ) -> CustomMessageEntry:
        snapshot = self.load_session(session_ref)
        entry = CustomMessageEntry(
            id=self._generate_entry_id(snapshot.entries),
            parent_id=snapshot.leaf_id,
            timestamp=_iso_now(),
            custom_type=custom_type,
            content=content,
            details=details,
            display=display,
        )
        self._append_record(snapshot.session_file, self._entry_to_record(entry))
        return entry

    def append_label_change(self, session_ref: str | Path, target_id: str, label: str | None) -> LabelEntry:
        snapshot = self.load_session(session_ref)
        entry = LabelEntry(
            id=self._generate_entry_id(snapshot.entries),
            parent_id=snapshot.leaf_id,
            timestamp=_iso_now(),
            target_id=target_id,
            label=label,
        )
        self._append_record(snapshot.session_file, self._entry_to_record(entry))
        return entry

    def append_session_info(self, session_ref: str | Path, name: str | None) -> SessionInfoEntry:
        snapshot = self.load_session(session_ref)
        entry = SessionInfoEntry(
            id=self._generate_entry_id(snapshot.entries),
            parent_id=snapshot.leaf_id,
            timestamp=_iso_now(),
            name=name,
        )
        self._append_record(snapshot.session_file, self._entry_to_record(entry))
        if name:
            self._update_header(snapshot.session_file, title=name)
        return entry

    def get_entries(self, session_ref: str | Path) -> list[SessionEntry]:
        return self.load_session(session_ref).entries

    def get_leaf_id(self, session_ref: str | Path) -> str | None:
        return self.load_session(session_ref).leaf_id

    def get_entry(self, session_ref: str | Path, entry_id: str) -> SessionEntry | None:
        for entry in self.load_session(session_ref).entries:
            if entry.id == entry_id:
                return entry
        return None

    def get_branch(self, session_ref: str | Path, from_id: str | None = None) -> list[SessionEntry]:
        snapshot = self.load_session(session_ref)
        by_id = {entry.id: entry for entry in snapshot.entries}
        path: list[SessionEntry] = []
        current = by_id.get(from_id or snapshot.leaf_id or "")
        while current is not None:
            path.insert(0, current)
            current = by_id.get(current.parent_id or "")
        return path

    def get_tree(self, session_ref: str | Path) -> list[SessionTreeNode]:
        snapshot = self.load_session(session_ref)
        labels: dict[str, str] = {}
        for entry in snapshot.entries:
            if isinstance(entry, LabelEntry):
                if entry.label:
                    labels[entry.target_id] = entry.label
                else:
                    labels.pop(entry.target_id, None)
        nodes = {entry.id: SessionTreeNode(entry=entry, label=labels.get(entry.id)) for entry in snapshot.entries}
        roots: list[SessionTreeNode] = []
        for entry in snapshot.entries:
            node = nodes[entry.id]
            if entry.parent_id and entry.parent_id in nodes:
                nodes[entry.parent_id].children.append(node)
            else:
                roots.append(node)
        return roots

    def branch(self, session_ref: str | Path, branch_from_id: str) -> SessionSnapshot:
        snapshot = self.load_session(session_ref)
        if not any(entry.id == branch_from_id for entry in snapshot.entries):
            raise ValueError(f"Entry {branch_from_id} not found")
        snapshot.leaf_id = branch_from_id
        return snapshot

    def branch_with_summary(
        self,
        session_ref: str | Path,
        branch_from_id: str | None,
        summary: str,
        details: dict[str, Any] | None = None,
        from_hook: bool = False,
    ) -> BranchSummaryEntry:
        snapshot = self.load_session(session_ref)
        entry = BranchSummaryEntry(
            id=self._generate_entry_id(snapshot.entries),
            parent_id=branch_from_id,
            timestamp=_iso_now(),
            from_id=branch_from_id or "root",
            summary=summary,
            details=details,
            from_hook=from_hook,
        )
        self._append_record(snapshot.session_file, self._entry_to_record(entry))
        return entry

    def create_branched_session(self, session_ref: str | Path, leaf_id: str) -> Path:
        snapshot = self.load_session(session_ref)
        path_entries = self.get_branch(snapshot.session_file, leaf_id)
        if not path_entries:
            raise ValueError(f"Entry {leaf_id} not found")
        branched = self.create_session(cwd=snapshot.cwd, model_id=snapshot.model_id, title=snapshot.title, parent_session=str(snapshot.session_file))
        content = [json.dumps(self._header_to_record(branched.header), ensure_ascii=False)]
        for entry in path_entries:
            content.append(json.dumps(self._entry_to_record(entry), ensure_ascii=False))
        branched.session_file.write_text("\n".join(content) + "\n", encoding="utf-8")
        return branched.session_file

    def build_session_context(self, session_ref: str | Path, leaf_id: str | None = None) -> SessionConversationContext:
        snapshot = self.load_session(session_ref)
        path = self.get_branch(snapshot.session_file, leaf_id)
        messages: list[ConversationMessage] = []
        thinking_level = "off"
        model: dict[str, str] | None = None
        compaction: CompactionEntry | None = None

        for entry in path:
            if isinstance(entry, ThinkingLevelChangeEntry):
                thinking_level = entry.thinking_level
            elif isinstance(entry, ModelChangeEntry):
                model = {"provider": entry.provider, "model_id": entry.model_id}
            elif isinstance(entry, SessionMessageEntry) and isinstance(entry.message, AssistantMessage):
                model = {"provider": entry.message.metadata.get("provider", ""), "model_id": entry.message.metadata.get("model", "")}
            elif isinstance(entry, CompactionEntry):
                compaction = entry

        def append_message(entry: SessionEntry) -> None:
            if isinstance(entry, SessionMessageEntry):
                messages.append(entry.message)
            elif isinstance(entry, BranchSummaryEntry) and entry.summary:
                messages.append(UserMessage(content=f"[Branch Summary]\n{entry.summary}", metadata={"branch_summary": True, "from_id": entry.from_id}))
            elif isinstance(entry, CustomMessageEntry):
                messages.append(UserMessage(content=str(entry.content), metadata={"custom_message": entry.custom_type, "display": entry.display, **(entry.details or {})}))

        if compaction is not None:
            messages.append(UserMessage(content=f"[Session Summary]\n{compaction.summary}", metadata={"summary": True}))
            compaction_idx = next((index for index, item in enumerate(path) if item.id == compaction.id), -1)
            found_first_kept = False
            for index in range(compaction_idx):
                entry = path[index]
                if entry.id == compaction.first_kept_entry_id:
                    found_first_kept = True
                if found_first_kept:
                    append_message(entry)
            for entry in path[compaction_idx + 1 :]:
                append_message(entry)
        else:
            for entry in path:
                append_message(entry)

        return SessionConversationContext(messages=messages, thinking_level=thinking_level, model=model)

    def build_context_messages(self, session_ref: str | Path, branch_id: str | None = None) -> list[ConversationMessage]:
        return self.build_session_context(session_ref, branch_id).messages

    def iter_events(self, session_ref: str | Path) -> list[dict[str, Any]]:
        session_file = self.resolve_session_file(session_ref)
        if session_file is None or not session_file.exists():
            return []
        return [json.loads(line) for line in session_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    def get_header(self, session_ref: str | Path) -> SessionHeader | None:
        snapshot = self.load_session(session_ref)
        return snapshot.header

    def _load_records(self, session_file: Path) -> list[dict[str, Any]]:
        if not session_file.exists():
            return []
        items: list[dict[str, Any]] = []
        for line in session_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return items

    def _append_record(self, session_file: Path, record: dict[str, Any]) -> None:
        with session_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _update_header(self, session_file: Path, *, title: str | None = None, model_id: str | None = None) -> None:
        records = self._load_records(session_file)
        if not records:
            return
        header = records[0]
        if title is not None and title:
            header["title"] = title
        if model_id is not None:
            header["model_id"] = model_id
        records[0] = header
        session_file.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n", encoding="utf-8")

    def _header_to_record(self, header: SessionHeader) -> dict[str, Any]:
        return {
            "type": "session",
            "version": header.version,
            "id": header.id,
            "timestamp": header.timestamp,
            "cwd": header.cwd,
            "parent_session": header.parent_session,
            "title": header.title,
            "model_id": header.model_id,
        }

    def _record_to_header(self, record: dict[str, Any]) -> SessionHeader:
        return SessionHeader(
            version=int(record.get("version", CURRENT_SESSION_VERSION)),
            id=str(record.get("id", "")),
            timestamp=str(record.get("timestamp", "")),
            cwd=str(record.get("cwd", "")),
            parent_session=record.get("parent_session"),
            title=str(record.get("title", "")),
            model_id=str(record.get("model_id", "")),
        )

    def _entry_to_record(self, entry: SessionEntry) -> dict[str, Any]:
        base = {"type": entry.type, "id": entry.id, "parent_id": entry.parent_id, "timestamp": entry.timestamp}
        if isinstance(entry, SessionMessageEntry):
            base["message"] = serialize_message(entry.message)
        elif isinstance(entry, ThinkingLevelChangeEntry):
            base["thinking_level"] = entry.thinking_level
        elif isinstance(entry, ModelChangeEntry):
            base["provider"] = entry.provider
            base["model_id"] = entry.model_id
        elif isinstance(entry, CompactionEntry):
            base["summary"] = entry.summary
            base["first_kept_entry_id"] = entry.first_kept_entry_id
            base["tokens_before"] = entry.tokens_before
            base["details"] = entry.details
            base["from_hook"] = entry.from_hook
        elif isinstance(entry, BranchSummaryEntry):
            base["from_id"] = entry.from_id
            base["summary"] = entry.summary
            base["details"] = entry.details
            base["from_hook"] = entry.from_hook
        elif isinstance(entry, CustomEntry):
            base["custom_type"] = entry.custom_type
            base["data"] = entry.data
        elif isinstance(entry, CustomMessageEntry):
            base["custom_type"] = entry.custom_type
            base["content"] = entry.content
            base["details"] = entry.details
            base["display"] = entry.display
        elif isinstance(entry, LabelEntry):
            base["target_id"] = entry.target_id
            base["label"] = entry.label
        elif isinstance(entry, SessionInfoEntry):
            base["name"] = entry.name
        return base

    def _record_to_entry(self, record: dict[str, Any]) -> SessionEntry:
        base = {
            "id": str(record.get("id", "")),
            "parent_id": record.get("parent_id"),
            "timestamp": str(record.get("timestamp", "")),
        }
        entry_type = str(record.get("type", "message"))
        if entry_type == "message":
            return SessionMessageEntry(message=deserialize_message(dict(record.get("message", {}))), **base)
        if entry_type == "thinking_level_change":
            return ThinkingLevelChangeEntry(thinking_level=str(record.get("thinking_level", "off")), **base)
        if entry_type == "model_change":
            return ModelChangeEntry(provider=str(record.get("provider", "")), model_id=str(record.get("model_id", "")), **base)
        if entry_type == "compaction":
            return CompactionEntry(
                summary=str(record.get("summary", "")),
                first_kept_entry_id=str(record.get("first_kept_entry_id", "")),
                tokens_before=int(record.get("tokens_before", 0)),
                details=record.get("details"),
                from_hook=bool(record.get("from_hook", False)),
                **base,
            )
        if entry_type == "branch_summary":
            return BranchSummaryEntry(
                from_id=str(record.get("from_id", "")),
                summary=str(record.get("summary", "")),
                details=record.get("details"),
                from_hook=bool(record.get("from_hook", False)),
                **base,
            )
        if entry_type == "custom":
            return CustomEntry(custom_type=str(record.get("custom_type", "")), data=record.get("data"), **base)
        if entry_type == "custom_message":
            return CustomMessageEntry(
                custom_type=str(record.get("custom_type", "")),
                content=record.get("content", ""),
                details=record.get("details"),
                display=bool(record.get("display", True)),
                **base,
            )
        if entry_type == "label":
            return LabelEntry(target_id=str(record.get("target_id", "")), label=record.get("label"), **base)
        if entry_type == "session_info":
            return SessionInfoEntry(name=record.get("name"), **base)
        raise ValueError(f"Unsupported session entry type: {entry_type}")

    def _iter_session_files(self, *, cwd: Path | None = None):
        if cwd is not None:
            session_dir = self._session_dir_for_cwd(cwd)
            if not session_dir.exists():
                return []
            return sorted(session_dir.glob("*.jsonl"))
        return sorted(self.sessions_dir.glob("**/*.jsonl"))

    def _find_session_file_by_id(self, session_id: str) -> Path | None:
        for item in self.sessions_dir.glob("**/*.jsonl"):
            if item.stem.endswith(f"_{session_id}") or item.stem == session_id:
                return item.resolve()
            records = self._load_records(item)
            if records and str(records[0].get("id", "")) == session_id:
                return item.resolve()
        return None

    def _session_dir_for_cwd(self, cwd: Path) -> Path:
        return self.sessions_dir / _safe_session_dir_name(cwd)

    def _generate_entry_id(self, entries: list[SessionEntry]) -> str:
        existing = {entry.id for entry in entries}
        while True:
            candidate = uuid.uuid4().hex[:8]
            if candidate not in existing:
                return candidate

    def _build_session_info(self, session_file: Path) -> SessionInfo | None:
        records = self._load_records(session_file)
        if not records or records[0].get("type") != "session":
            return None
        header = self._record_to_header(records[0])
        entries = [self._record_to_entry(item) for item in records[1:]]
        message_entries = [entry for entry in entries if isinstance(entry, SessionMessageEntry)]
        texts = [str(entry.message.content) for entry in message_entries if getattr(entry.message, "role", "") in {"user", "assistant"}]
        first_message = next((str(entry.message.content) for entry in message_entries if getattr(entry.message, "role", "") == "user"), "(no messages)")
        name = None
        for entry in reversed(entries):
            if isinstance(entry, SessionInfoEntry):
                name = entry.name or None
                break
        modified = entries[-1].timestamp if entries else header.timestamp
        return SessionInfo(
            path=str(session_file),
            id=header.id,
            cwd=header.cwd,
            name=name,
            parent_session_path=header.parent_session,
            created_at=header.timestamp,
            modified_at=modified,
            message_count=len(message_entries),
            first_message=first_message,
            all_messages_text=" ".join(texts),
            leaf_id=entries[-1].id if entries else None,
            title=header.title,
            model_id=header.model_id,
        )

    def _migrate_all_legacy_sessions(self) -> None:
        for child in self.sessions_dir.iterdir():
            if child.is_dir() and (child / "events.jsonl").exists() and (child / "meta.json").exists():
                self._migrate_legacy_session(child)

    def _migrate_legacy_session(self, legacy_dir: Path) -> Path:
        meta_file = legacy_dir / "meta.json"
        events_file = legacy_dir / "events.jsonl"
        if not meta_file.exists() or not events_file.exists():
            raise FileNotFoundError(f"Legacy session missing meta/events: {legacy_dir}")
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        cwd = Path(str(meta.get("cwd", self.sessions_dir))).resolve()
        session_dir = self._session_dir_for_cwd(cwd)
        session_dir.mkdir(parents=True, exist_ok=True)
        session_file = session_dir / f"migrated_{legacy_dir.name}.jsonl"
        if session_file.exists():
            return session_file

        header = SessionHeader(
            version=CURRENT_SESSION_VERSION,
            id=str(meta.get("session_id", legacy_dir.name)),
            timestamp=str(meta.get("created_at", _iso_now())),
            cwd=str(cwd),
            parent_session=None,
            title=str(meta.get("title", legacy_dir.name)),
            model_id=str(meta.get("model_id", meta.get("model", ""))),
        )
        records = [self._header_to_record(header)]
        last_entry_id: str | None = None
        first_kept_entry_id: str | None = None
        for raw_line in events_file.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            event = json.loads(raw_line)
            if event.get("type") == "message":
                payload = dict(event.get("payload", {}))
                message = deserialize_message(
                    {
                        "role": payload.get("role", "user"),
                        "content": payload.get("content", ""),
                        "metadata": payload.get("metadata", {}),
                        "thinking": payload.get("thinking", ""),
                        "tool_calls": payload.get("tool_calls", []),
                        "tool_call_id": payload.get("tool_call_id", ""),
                        "tool_name": payload.get("tool_name", ""),
                    }
                )
                entry = SessionMessageEntry(
                    id=str(payload.get("id", uuid.uuid4().hex[:8])),
                    parent_id=last_entry_id,
                    timestamp=_iso_now(),
                    message=message,
                )
                records.append(self._entry_to_record(entry))
                last_entry_id = entry.id
                if first_kept_entry_id is None:
                    first_kept_entry_id = entry.id
            elif event.get("type") == "summary":
                payload = dict(event.get("payload", {}))
                entry = CompactionEntry(
                    id=uuid.uuid4().hex[:8],
                    parent_id=last_entry_id,
                    timestamp=_iso_now(),
                    summary=str(payload.get("summary", "")),
                    first_kept_entry_id=first_kept_entry_id or "",
                    tokens_before=0,
                    details={"migrated_node_ids": payload.get("node_ids", [])},
                    from_hook=False,
                )
                records.append(self._entry_to_record(entry))
                last_entry_id = entry.id
        session_file.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n", encoding="utf-8")
        return session_file
