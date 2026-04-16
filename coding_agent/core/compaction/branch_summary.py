from __future__ import annotations

from .compactor import SessionCompactor


async def summarize_branch_on_switch(
    compactor: SessionCompactor,
    session_ref: str,
    old_leaf_id: str | None,
    target_leaf_id: str | None,
) -> str:
    """在切换到另一条分支前，为离开的路径生成摘要。"""

    if old_leaf_id is None or old_leaf_id == target_leaf_id:
        return ""
    snapshot = compactor.session_manager.load_session(session_ref)
    old_branch = compactor.session_manager.get_branch(snapshot.session_file, old_leaf_id)
    target_branch_ids = {entry.id for entry in compactor.session_manager.get_branch(snapshot.session_file, target_leaf_id)}
    entries_to_summarize = [entry for entry in old_branch if entry.id not in target_branch_ids]
    messages = compactor._messages_from_entries(entries_to_summarize, include_compactions=True)
    if not messages:
        return ""
    return await compactor._summarize_messages(messages, custom_instructions="Summarize only the abandoned branch.", previous_summary=None)
