from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .types import BranchSummarySettings, CodingAgentSettings, CompactionSettings, ToolPolicy


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并两层配置，后者覆盖前者。"""

    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class SettingsManager:
    """负责读取、合并并保存 coding-agent 设置。"""

    def __init__(self, global_settings_file: Path, project_settings_file: Path | None = None) -> None:
        """初始化全局与项目级设置文件位置。"""

        self.global_settings_file = global_settings_file  # 全局设置文件路径。
        self.project_settings_file = project_settings_file  # 项目级设置文件路径。

    def load(self) -> CodingAgentSettings:
        """读取并合并全局设置与项目设置。"""

        global_data = self._load_json(self.global_settings_file)
        project_data = self._load_json(self.project_settings_file) if self.project_settings_file else {}
        merged = _deep_merge(global_data, project_data)
        tool_policy = ToolPolicy(**merged.get("tool_policy", {}))
        compaction_raw = dict(merged.get("compaction", {}))
        compaction = CompactionSettings(
            enabled=bool(compaction_raw.get("enabled", True)),
            reserve_tokens=int(compaction_raw.get("reserve_tokens", compaction_raw.get("reserveTokens", 16384))),
            keep_recent_tokens=int(
                compaction_raw.get("keep_recent_tokens", compaction_raw.get("keepRecentTokens", 20000))
            ),
            compact_model=compaction_raw.get("compact_model"),
        )
        branch_summary_raw = dict(merged.get("branch_summary", {}))
        branch_summary = BranchSummarySettings(
            reserve_tokens=int(branch_summary_raw.get("reserve_tokens", branch_summary_raw.get("reserveTokens", 16384))),
            skip_prompt=bool(branch_summary_raw.get("skip_prompt", branch_summary_raw.get("skipPrompt", False))),
        )
        return CodingAgentSettings(
            default_model=merged.get("default_model", "openai:gpt-5"),
            default_thinking=merged.get("default_thinking", "medium"),
            theme=merged.get("theme", "default"),
            system_prompt_override=merged.get("system_prompt_override"),
            tool_policy=tool_policy,
            compaction=compaction,
            branch_summary=branch_summary,
        )

    def save_global(self, settings: CodingAgentSettings) -> None:
        """将设置保存回全局配置文件。"""

        self.global_settings_file.parent.mkdir(parents=True, exist_ok=True)
        self.global_settings_file.write_text(json.dumps(asdict(settings), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _load_json(path: Path | None) -> dict[str, Any]:
        """从 JSON 文件读取对象，不存在时返回空字典。"""

        if path is None or not path.exists():
            return {}
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return {}
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"Settings file must contain an object: {path}")
        return data
