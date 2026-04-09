from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_HOME_DIRNAME = ".LiuClaw"
DEFAULT_AGENT_DIRNAME = "agent"
PROJECT_DIRNAME = ".LiuClaw"


@dataclass(slots=True)
class AgentPaths:
    """定义 coding-agent 在用户目录下使用的全部路径。"""

    root: Path  # coding-agent 用户级根目录。
    settings_file: Path  # 全局设置文件路径。
    models_file: Path  # 用户自定义模型文件路径。
    sessions_dir: Path  # 会话持久化目录。
    skills_dir: Path  # 技能目录。
    prompts_dir: Path  # 提示模板目录。
    themes_dir: Path  # 主题目录。
    extensions_dir: Path  # 扩展目录。

    def ensure_exists(self) -> None:
        """确保配置目录及默认配置文件存在。"""

        self.root.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.themes_dir.mkdir(parents=True, exist_ok=True)
        self.extensions_dir.mkdir(parents=True, exist_ok=True)
        if not self.settings_file.exists():
            self.settings_file.write_text("{}\n", encoding="utf-8")
        if not self.models_file.exists():
            self.models_file.write_text("[]\n", encoding="utf-8")


def build_agent_paths(home: Path | None = None) -> AgentPaths:
    """根据给定 home 目录构造统一的配置路径集合。"""

    base_home = (home or Path.home()) / DEFAULT_HOME_DIRNAME / DEFAULT_AGENT_DIRNAME
    return AgentPaths(
        root=base_home,
        settings_file=base_home / "settings.json",
        models_file=base_home / "models.json",
        sessions_dir=base_home / "sessions",
        skills_dir=base_home / "skills",
        prompts_dir=base_home / "prompts",
        themes_dir=base_home / "themes",
        extensions_dir=base_home / "extensions",
    )


def find_project_settings_file(workspace_root: Path) -> Path:
    """返回工作区内项目级设置文件的固定位置。"""

    return workspace_root / PROJECT_DIRNAME / "settings.json"
