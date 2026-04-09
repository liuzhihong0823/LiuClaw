from __future__ import annotations

from pathlib import Path

from .agents_context_loader import load_agents_context
from .extensions_runtime import load_extension_runtime, scan_extensions
from .prompts_loader import load_prompts
from .skills_loader import load_skills
from .themes_loader import load_themes
from .types import ResourceBundle


class ResourceLoader:
    """统一编排技能、提示、主题、扩展与项目上下文的加载。"""

    def __init__(
        self,
        *,
        skills_dir: Path,
        prompts_dir: Path,
        themes_dir: Path,
        extensions_dir: Path,
        workspace_root: Path,
    ) -> None:
        """保存资源目录与工作区根目录。"""

        self.skills_dir = skills_dir  # 技能目录。
        self.prompts_dir = prompts_dir  # 提示模板目录。
        self.themes_dir = themes_dir  # 主题目录。
        self.extensions_dir = extensions_dir  # 扩展目录。
        self.workspace_root = workspace_root  # 当前工作区根目录。

    def load(self) -> ResourceBundle:
        """执行一次完整资源扫描，并返回聚合后的资源包。"""

        skills = load_skills(self.skills_dir)
        prompts = load_prompts(self.prompts_dir)
        themes = load_themes(self.themes_dir)
        agents_context = load_agents_context(self.workspace_root)
        extensions = self._scan_extensions()
        extension_runtime = load_extension_runtime(extensions)
        self._ensure_no_conflicts(skills, prompts, themes, extensions)
        return ResourceBundle(
            skills=skills,
            prompts=prompts,
            themes=themes,
            agents_context=agents_context,
            extensions=extensions,
            extension_runtime=extension_runtime,
        )

    def _scan_extensions(self) -> list[ExtensionResource]:
        """扫描扩展目录，仅返回描述信息而不执行扩展代码。"""

        if not self.extensions_dir.exists():
            return []
        return scan_extensions(self.extensions_dir)

    @staticmethod
    def _ensure_no_conflicts(skills, prompts, themes, extensions) -> None:
        """检测跨资源类型的同名冲突，避免运行时歧义。"""

        seen: dict[str, str] = {}
        for kind, collection in [
            ("skill", skills),
            ("prompt", prompts.values()),
            ("theme", themes.values()),
            ("extension", extensions),
        ]:
            for item in collection:
                if item.name in seen:
                    raise ValueError(f"Resource name conflict: '{item.name}' used by {seen[item.name]} and {kind}")
                seen[item.name] = kind
