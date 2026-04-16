from __future__ import annotations

from ai.types import Model

from ..types import CodingAgentSettings, ContextStats


def should_compact(context_stats: ContextStats, settings: CodingAgentSettings, model: Model) -> bool:
    """根据 token 预算判断是否触发压缩。"""

    compaction = settings.compaction
    if not compaction.enabled:
        return False
    return context_stats.estimated_tokens >= max(0, model.contextWindow - compaction.reserve_tokens)
