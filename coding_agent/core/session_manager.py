from __future__ import annotations

import json
import uuid
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
    SessionTreeNode,
    ThinkingLevelChangeEntry,
    deserialize_message,
    serialize_message,
)

CURRENT_SESSION_VERSION = 1


def _iso_now() -> str:
    """返回当前 UTC 时间的 ISO 字符串表示。"""

    return datetime.now(UTC).isoformat()


def _safe_session_dir_name(cwd: Path) -> str:
    """将工作目录转换为可安全落盘的 session 目录名。"""

    raw = str(cwd.resolve()).lstrip("/\\")
    return f"--{raw.replace('/', '-').replace('\\', '-').replace(':', '-')}--"


FileEntry = SessionHeader | SessionEntry


class SessionManager:
    """树状、追加式的会话存储管理器。"""

    def __init__(self, sessions_dir: Path) -> None:
        """初始化会话存储目录，并准备当前会话的内存状态。"""

        self.sessions_dir = sessions_dir  # 会话根目录。
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.session_file: Path | None = None  # 当前打开的会话文件路径。
        self.session_id: str = ""  # 当前打开的会话 ID。
        self.cwd: Path | None = None  # 当前会话工作目录。
        self.file_entries: list[FileEntry] = []  # 当前会话的完整条目列表（包含 header）。
        self.by_id: dict[str, SessionEntry] = {}  # 条目 ID 到条目对象的索引。
        self.labels_by_id: dict[str, str] = {}  # 已解析的条目标签索引。
        self.leaf_id: str | None = None  # 当前活动分支的叶子节点 ID。
        self.flushed: bool = True  # 当前内存状态是否已全部落盘。

    @property
    def branch_id(self) -> str:
        """兼容旧字段名，实际值等价于当前 leaf ID。"""

        return self.leaf_id or "main"

    @property
    def model_id(self) -> str:
        """返回当前会话头里记录的模型 ID。"""

        header = self.get_header()
        return header.model_id if header is not None else ""

    @property
    def title(self) -> str:
        """返回当前会话标题。"""

        header = self.get_header()
        return header.title if header is not None else ""

    def create_session(self, *, cwd: Path, model_id: str, title: str | None = None, parent_session: str | None = None) -> SessionManager:
        """创建一个新的空会话文件，并将当前 manager 切换到该会话。"""

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
        self.session_file = session_file.resolve()
        self.session_id = header.id
        self.cwd = Path(header.cwd).resolve()
        self.file_entries = [header]
        self.by_id = {}
        self.labels_by_id = {}
        self.leaf_id = None
        self.flushed = True
        return self

    def continue_recent(self, cwd: Path) -> SessionManager:
        """继续当前目录最近一次会话；若不存在则新建。"""

        recent = self.list_recent_sessions(limit=1, cwd=cwd)
        if recent:
            return self.open(recent[0]["session_file"])
        return self.create_session(cwd=cwd, model_id="")

    def open(self, session_file: str | Path) -> SessionManager:
        """按文件路径或引用打开一个已有会话。"""

        self.set_session_file(session_file)
        return self

    def load_session(self, session_ref: str | Path) -> SessionManager:
        """兼容旧调用方，内部直接切换到指定会话并返回当前 manager。"""

        return self.open(session_ref)

    def set_session_file(self, session_ref: str | Path) -> None:
        """切换到指定会话文件，并重建当前会话的内存索引。"""

        resolved = self.resolve_session_file(session_ref)
        if resolved is None:
            raise FileNotFoundError(f"Unknown session reference: {session_ref}")
        self.session_file = resolved
        self.file_entries = self.load_entries_from_file(resolved)
        if not self.file_entries:
            raise ValueError(f"Invalid session file: {resolved}")
        self._build_index()
        self.flushed = True

    def load_entries_from_file(self, file_path: str | Path) -> list[FileEntry]:
        """读取并校验 JSONL 会话文件，恢复为内存条目列表。"""

        path = Path(file_path)
        if not path.exists():
            return []
        entries: list[FileEntry] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if not entries:
                if record.get("type") != "session" or not isinstance(record.get("id"), str):
                    return []
                entries.append(self._record_to_header(record))
                continue
            entries.append(self._record_to_entry(record))
        return entries

    def resolve_session_file(self, session_ref: str | Path | None) -> Path | None:
        """将会话引用解析成真实的 session 文件路径。"""

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
        return None

    def list_recent_sessions(self, *, limit: int = 10, cwd: Path | None = None) -> list[dict[str, Any]]:
        """列出最近更新的会话摘要信息。"""

        items: list[dict[str, Any]] = []
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

    def append_message(self, message: ConversationMessage, parent_id: str | None = None) -> SessionMessageEntry:
        """向当前会话树追加一条普通消息节点。"""

        self._require_open_session()
        entry = SessionMessageEntry(
            id=self._generate_entry_id(),
            parent_id=parent_id if parent_id is not None else self.leaf_id,
            timestamp=_iso_now(),
            message=message,
        )
        self._append_entry(entry)
        if isinstance(message, UserMessage):
            self._update_header(title=str(message.content).strip().replace("\n", " ")[:80] or self.title)
        return entry

    def append_thinking_level_change(self, thinking_level: str) -> ThinkingLevelChangeEntry:
        """记录一次 thinking 级别变更。"""

        self._require_open_session()
        entry = ThinkingLevelChangeEntry(
            id=self._generate_entry_id(),
            parent_id=self.leaf_id,
            timestamp=_iso_now(),
            thinking_level=thinking_level,
        )
        self._append_entry(entry)
        return entry

    def append_model_change(self, provider: str, model_id: str) -> ModelChangeEntry:
        """记录一次模型切换，并同步更新会话头信息。"""

        self._require_open_session()
        entry = ModelChangeEntry(
            id=self._generate_entry_id(),
            parent_id=self.leaf_id,
            timestamp=_iso_now(),
            provider=provider,
            model_id=model_id,
        )
        self._append_entry(entry)
        self._update_header(model_id=model_id)
        return entry

    def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int,
        details: dict[str, Any] | None = None,
        from_hook: bool = False,
    ) -> CompactionEntry:
        """向当前会话追加一条压缩摘要记录。"""

        self._require_open_session()
        entry = CompactionEntry(
            id=self._generate_entry_id(),
            parent_id=self.leaf_id,
            timestamp=_iso_now(),
            summary=summary,
            first_kept_entry_id=first_kept_entry_id,
            tokens_before=tokens_before,
            details=details,
            from_hook=from_hook,
        )
        self._append_entry(entry)
        return entry

    def append_branch_summary(
        self,
        from_id: str,
        summary: str,
        details: dict[str, Any] | None = None,
        from_hook: bool = False,
    ) -> BranchSummaryEntry:
        """向当前会话追加一条分支摘要记录。"""

        self._require_open_session()
        entry = BranchSummaryEntry(
            id=self._generate_entry_id(),
            parent_id=self.leaf_id,
            timestamp=_iso_now(),
            from_id=from_id,
            summary=summary,
            details=details,
            from_hook=from_hook,
        )
        self._append_entry(entry)
        return entry

    def append_custom_entry(self, custom_type: str, data: Any | None = None) -> CustomEntry:
        """追加一条不参与上下文恢复的自定义记录。"""

        self._require_open_session()
        entry = CustomEntry(
            id=self._generate_entry_id(),
            parent_id=self.leaf_id,
            timestamp=_iso_now(),
            custom_type=custom_type,
            data=data,
        )
        self._append_entry(entry)
        return entry

    def append_custom_message_entry(
        self,
        custom_type: str,
        content: str | list[dict[str, Any]],
        display: bool,
        details: dict[str, Any] | None = None,
    ) -> CustomMessageEntry:
        """追加一条会被解释进模型上下文的自定义消息记录。"""

        self._require_open_session()
        entry = CustomMessageEntry(
            id=self._generate_entry_id(),
            parent_id=self.leaf_id,
            timestamp=_iso_now(),
            custom_type=custom_type,
            content=content,
            details=details,
            display=display,
        )
        self._append_entry(entry)
        return entry

    def append_label_change(self, target_id: str, label: str | None) -> LabelEntry:
        """为指定节点追加标签或清除标签。"""

        self._require_open_session()
        entry = LabelEntry(
            id=self._generate_entry_id(),
            parent_id=self.leaf_id,
            timestamp=_iso_now(),
            target_id=target_id,
            label=label,
        )
        self._append_entry(entry)
        return entry

    def append_session_info(self, name: str | None) -> SessionInfoEntry:
        """记录会话展示名称等附加信息。"""

        self._require_open_session()
        entry = SessionInfoEntry(
            id=self._generate_entry_id(),
            parent_id=self.leaf_id,
            timestamp=_iso_now(),
            name=name,
        )
        self._append_entry(entry)
        if name:
            self._update_header(title=name)
        return entry

    def get_entries(self) -> list[SessionEntry]:
        """返回当前会话中的全部条目。"""

        return [entry for entry in self.file_entries[1:] if isinstance(entry, self._session_entry_types())]

    def get_leaf_id(self) -> str | None:
        """返回当前会话叶子节点 ID。"""

        return self.leaf_id

    def get_entry(self, entry_id: str) -> SessionEntry | None:
        """按条目 ID 查找当前会话中的单个节点。"""

        return self.by_id.get(entry_id)

    def get_branch(self, from_id: str | None = None) -> list[SessionEntry]:
        """从指定叶子回溯出一条完整分支路径。"""

        current = self.by_id.get(from_id or self.leaf_id or "")
        path: list[SessionEntry] = []
        while current is not None:
            path.insert(0, current)
            current = self.by_id.get(current.parent_id or "")
        return path

    def get_tree(self) -> list[SessionTreeNode]:
        """将当前会话条目重建为树结构节点列表。"""

        entries = self.get_entries()
        node_map = {entry.id: SessionTreeNode(entry=entry, label=self.labels_by_id.get(entry.id)) for entry in entries}
        roots: list[SessionTreeNode] = []
        for entry in entries:
            node = node_map[entry.id]
            if entry.parent_id is None or entry.parent_id not in node_map:
                roots.append(node)
                continue
            node_map[entry.parent_id].children.append(node)
        for node in node_map.values():
            node.children.sort(key=lambda item: item.entry.timestamp)
        roots.sort(key=lambda item: item.entry.timestamp)
        return roots

    def branch(self, branch_from_id: str) -> None:
        """将当前会话视角切换到指定节点对应的分支。"""

        if branch_from_id not in self.by_id:
            raise ValueError(f"Entry {branch_from_id} not found")
        self.leaf_id = branch_from_id

    def reset_leaf(self) -> None:
        """将当前分支指针重置到根之前的位置。"""

        self.leaf_id = None

    def branch_with_summary(
        self,
        branch_from_id: str | None,
        summary: str,
        details: dict[str, Any] | None = None,
        from_hook: bool = False,
    ) -> BranchSummaryEntry:
        """切换到指定分支点，并追加一条摘要化的分支说明。"""

        if branch_from_id is not None and branch_from_id not in self.by_id:
            raise ValueError(f"Entry {branch_from_id} not found")
        self.leaf_id = branch_from_id
        entry = BranchSummaryEntry(
            id=self._generate_entry_id(),
            parent_id=branch_from_id,
            timestamp=_iso_now(),
            from_id=branch_from_id or "root",
            summary=summary,
            details=details,
            from_hook=from_hook,
        )
        self._append_entry(entry)
        return entry

    def create_branched_session(self, leaf_id: str) -> Path:
        """基于某个叶子节点复制出一份只包含单一路径的新会话文件。"""

        self._require_open_session()
        path_entries = self.get_branch(leaf_id)
        if not path_entries:
            raise ValueError(f"Entry {leaf_id} not found")
        parent_session = str(self.session_file) if self.session_file is not None else None
        branch_manager = SessionManager(self.sessions_dir).create_session(
            cwd=self.cwd or Path.cwd(),
            model_id=self.model_id,
            title=self.title,
            parent_session=parent_session,
        )
        content = [json.dumps(branch_manager._header_to_record(branch_manager.get_header()), ensure_ascii=False)]
        for entry in path_entries:
            content.append(json.dumps(self._entry_to_record(entry), ensure_ascii=False))
        if branch_manager.session_file is None:
            raise RuntimeError("Branched session file was not created")
        branch_manager.session_file.write_text("\n".join(content) + "\n", encoding="utf-8")
        return branch_manager.session_file

    def build_session_context(self, leaf_id: str | None = None) -> SessionConversationContext:
        """将当前会话分支恢复为可直接送给模型的上下文对象。"""

        path = self.get_branch(leaf_id)
        if leaf_id is None and self.leaf_id is None:
            path = []
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
                model = {
                    "provider": entry.message.metadata.get("provider", ""),
                    "model_id": entry.message.metadata.get("model", ""),
                }
            elif isinstance(entry, CompactionEntry):
                compaction = entry

        def append_message(entry: SessionEntry) -> None:
            if isinstance(entry, SessionMessageEntry):
                messages.append(entry.message)
            elif isinstance(entry, BranchSummaryEntry) and entry.summary:
                messages.append(
                    UserMessage(
                        content=f"[Branch Summary]\n{entry.summary}",
                        metadata={"branch_summary": True, "from_id": entry.from_id},
                    )
                )
            elif isinstance(entry, CustomMessageEntry):
                payload = {"custom_message": entry.custom_type, "display": entry.display, **(entry.details or {})}
                messages.append(UserMessage(content=str(entry.content), metadata=payload))

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

    def build_context_messages(self, leaf_id: str | None = None) -> list[ConversationMessage]:
        """仅提取当前会话上下文中的消息列表。"""

        return self.build_session_context(leaf_id).messages

    def iter_events(self, session_ref: str | Path | None = None) -> list[dict[str, Any]]:
        """按原始 JSON 行读取会话文件中的事件记录，仅用于调试查看。"""

        session_file = self.resolve_session_file(session_ref) if session_ref is not None else self.session_file
        if session_file is None or not session_file.exists():
            return []
        return [json.loads(line) for line in session_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    def get_header(self) -> SessionHeader | None:
        """返回当前会话头信息。"""

        if not self.file_entries or not isinstance(self.file_entries[0], SessionHeader):
            return None
        return self.file_entries[0]

    def _build_index(self) -> None:
        """根据当前 file_entries 重建条目索引、标签索引与 leaf 指针。"""

        header = self.get_header()
        if header is None:
            raise ValueError("Session file missing header")
        self.session_id = header.id
        self.cwd = Path(header.cwd).resolve()
        self.by_id = {}
        self.labels_by_id = {}
        self.leaf_id = None
        for entry in self.get_entries():
            self.by_id[entry.id] = entry
            self.leaf_id = entry.id
            if isinstance(entry, LabelEntry):
                if entry.label:
                    self.labels_by_id[entry.target_id] = entry.label
                else:
                    self.labels_by_id.pop(entry.target_id, None)

    def _append_entry(self, entry: SessionEntry) -> None:
        """将条目追加到文件和内存索引中，并推进当前 leaf。"""

        self._require_open_session()
        if self.session_file is None:
            raise RuntimeError("No active session file")
        self.file_entries.append(entry)
        self.by_id[entry.id] = entry
        if isinstance(entry, LabelEntry):
            if entry.label:
                self.labels_by_id[entry.target_id] = entry.label
            else:
                self.labels_by_id.pop(entry.target_id, None)
        with self.session_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self._entry_to_record(entry), ensure_ascii=False) + "\n")
        self.leaf_id = entry.id
        self.flushed = True

    def _update_header(self, *, title: str | None = None, model_id: str | None = None) -> None:
        """重写首行头信息，用于同步标题或模型 ID。"""

        self._require_open_session()
        header = self.get_header()
        if header is None or self.session_file is None:
            return
        if title is not None and title:
            header.title = title
        if model_id is not None:
            header.model_id = model_id
        self._rewrite_file()

    def _rewrite_file(self) -> None:
        """将当前内存里的完整会话状态整份重写回 session 文件。"""

        if self.session_file is None:
            raise RuntimeError("No active session file")
        lines: list[str] = []
        for entry in self.file_entries:
            if isinstance(entry, SessionHeader):
                lines.append(json.dumps(self._header_to_record(entry), ensure_ascii=False))
            else:
                lines.append(json.dumps(self._entry_to_record(entry), ensure_ascii=False))
        self.session_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.flushed = True

    def _require_open_session(self) -> None:
        """确保当前 manager 已绑定到一个有效会话。"""

        if self.session_file is None or self.get_header() is None:
            raise RuntimeError("No active session")

    def _header_to_record(self, header: SessionHeader | None) -> dict[str, Any]:
        """将会话头对象转换为可持久化字典。"""

        if header is None:
            raise ValueError("Session header is required")
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
        """将持久化字典恢复为会话头对象。"""

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
        """将统一条目对象转换成 JSONL 记录。"""

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
        """将 JSONL 记录反序列化为统一条目对象。"""

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

    def _iter_session_files(self, *, cwd: Path | None = None) -> list[Path]:
        """遍历指定目录或全部工作区下的 session 文件。"""

        if cwd is not None:
            session_dir = self._session_dir_for_cwd(cwd)
            if not session_dir.exists():
                return []
            return sorted(session_dir.glob("*.jsonl"))
        return sorted(self.sessions_dir.glob("**/*.jsonl"))

    def _find_session_file_by_id(self, session_id: str) -> Path | None:
        """按 session ID 搜索对应的 session 文件。"""

        for item in self.sessions_dir.glob("**/*.jsonl"):
            if item.stem.endswith(f"_{session_id}") or item.stem == session_id:
                return item.resolve()
            entries = self.load_entries_from_file(item)
            header = entries[0] if entries else None
            if isinstance(header, SessionHeader) and header.id == session_id:
                return item.resolve()
        return None

    def _session_dir_for_cwd(self, cwd: Path) -> Path:
        """根据工作目录计算其 session 存放目录。"""

        return self.sessions_dir / _safe_session_dir_name(cwd)

    def _generate_entry_id(self) -> str:
        """生成一个在当前会话中唯一的条目 ID。"""

        while True:
            candidate = uuid.uuid4().hex[:8]
            if candidate not in self.by_id:
                return candidate

    def _build_session_info(self, session_file: Path) -> SessionInfo | None:
        """从 session 文件提炼出用于列表展示的摘要信息。"""

        entries = self.load_entries_from_file(session_file)
        if not entries or not isinstance(entries[0], SessionHeader):
            return None
        header = entries[0]
        session_entries = [entry for entry in entries[1:] if isinstance(entry, self._session_entry_types())]
        message_entries = [entry for entry in session_entries if isinstance(entry, SessionMessageEntry)]
        texts = [
            str(entry.message.content)
            for entry in message_entries
            if getattr(entry.message, "role", "") in {"user", "assistant"}
        ]
        first_message = next(
            (str(entry.message.content) for entry in message_entries if getattr(entry.message, "role", "") == "user"),
            "(no messages)",
        )
        name = None
        for entry in reversed(session_entries):
            if isinstance(entry, SessionInfoEntry):
                name = entry.name or None
                break
        modified = session_entries[-1].timestamp if session_entries else header.timestamp
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
            leaf_id=session_entries[-1].id if session_entries else None,
            title=header.title,
            model_id=header.model_id,
        )

    @staticmethod
    def _session_entry_types() -> tuple[type[Any], ...]:
        """返回所有合法会话条目类型，用于 isinstance 过滤。"""

        return (
            SessionMessageEntry,
            ThinkingLevelChangeEntry,
            ModelChangeEntry,
            CompactionEntry,
            BranchSummaryEntry,
            CustomEntry,
            CustomMessageEntry,
            LabelEntry,
            SessionInfoEntry,
        )
