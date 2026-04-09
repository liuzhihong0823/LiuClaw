from __future__ import annotations

from ai.types import Context, Model
from ai.utils.context_window import detect_context_overflow

from ..session_manager import SessionManager
from ..types import CodingAgentSettings, CompactResult, ContextStats
from .branch_summary import summarize_branch_on_switch
from .compactor import SessionCompactor
from .triggers import should_compact


class CompactionCoordinator:
    """统一管理阈值压缩、溢出恢复、手动压缩和分支摘要。"""

    def __init__(self, session_manager: SessionManager, settings: CodingAgentSettings, keep_turns: int | None = None) -> None:
        """初始化压缩协调器。"""

        self.session_manager = session_manager  # 会话管理器。
        self.settings = settings  # 当前压缩相关设置。
        self.compactor = SessionCompactor(session_manager, keep_turns=keep_turns or settings.compact_keep_turns)  # 实际执行压缩的对象。

    def compact_manual(self, session_id: str, branch_id: str | None = None) -> CompactResult:
        """手动触发压缩。"""

        return self.compactor.compact_session(session_id, branch_id)

    def maybe_compact_for_threshold(self, session_id: str, branch_id: str, model: Model, context: Context) -> CompactResult | None:
        """在发送前根据阈值判断是否触发压缩。"""

        if not self.settings.auto_compact:
            return None
        report = detect_context_overflow(model, context)
        ratio = report.total_tokens / report.limit if report.limit else 0.0
        stats = ContextStats(estimated_tokens=report.total_tokens, limit=report.limit, ratio=ratio)
        if should_compact(stats, self.settings, model):
            return self.compactor.compact_session(session_id, branch_id)
        return None

    def recover_from_overflow(self, session_id: str, branch_id: str) -> CompactResult | None:
        """在上下文溢出时尝试压缩并恢复。"""

        result = self.compactor.compact_session(session_id, branch_id)
        if result.compacted_count <= 0:
            return None
        return result

    def summarize_branch(self, session_id: str, branch_id: str) -> str:
        """为分支切换生成摘要。"""

        return summarize_branch_on_switch(self.compactor, session_id, branch_id)
