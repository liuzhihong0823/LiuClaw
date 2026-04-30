from __future__ import annotations

import asyncio
from pathlib import Path

from .cli import parse_args
from .config.paths import build_agent_paths, find_project_settings_file
from .core import AgentSession, ModelRegistry, ResourceLoader, SessionManager, SettingsManager
from .core.multi_agent import TeamRuntime
from .modes.interactive import InteractiveApp


def main(argv: list[str] | None = None) -> int:
    """组装配置、资源、会话与交互模式，并启动程序。"""

    args = parse_args(argv)
    workspace_root = Path(args.cwd).resolve()
    paths = build_agent_paths()
    paths.ensure_exists()
    settings_manager = SettingsManager(paths.settings_file, find_project_settings_file(workspace_root))
    settings = settings_manager.load()
    if args.theme:
        settings.theme = args.theme
    registry = ModelRegistry(paths.models_file)
    model = registry.get(args.model or settings.default_model)
    session_manager = SessionManager(paths.sessions_dir)
    resource_loader = ResourceLoader(
        skills_dir=paths.skills_dir,
        prompts_dir=paths.prompts_dir,
        themes_dir=paths.themes_dir,
        extensions_dir=paths.extensions_dir,
        workspace_root=workspace_root,
    )
    session_file: str | None = None
    if args.fork:
        source = session_manager.resolve_session_file(args.fork)
        if source is None:
            raise FileNotFoundError(f"Unknown fork source: {args.fork}")
        session_manager.open(source)
        session_file = str(session_manager.create_branched_session(session_manager.get_leaf_id() or "").resolve())
    elif args.session:
        resolved = session_manager.resolve_session_file(args.session)
        if resolved is None:
            raise FileNotFoundError(f"Unknown session: {args.session}")
        session_file = str(resolved)
    elif args.resume or args.continue_session:
        recent = session_manager.list_recent_sessions(limit=1, cwd=workspace_root)
        if recent:
            session_file = str(recent[0]["session_file"])
    session = AgentSession(
        workspace_root=workspace_root,
        cwd=workspace_root,
        model=model,
        thinking=args.thinking or settings.default_thinking,
        settings=settings,
        session_manager=session_manager,
        resource_loader=resource_loader,
        model_registry=registry,
        session_id=None,
        session_file=None if args.new else session_file,
    )
    session.attach_team_runtime(
        TeamRuntime(
            owner_session=session,
            workspace_root=workspace_root,
            model_registry=registry,
        )
    )
    if session_file:
        session.resume_session()
    if args.compact:
        asyncio.run(session.compact())
        return 0
    if args.prompt:
        async def _run_prompt() -> int:
            session.send_user_message(args.prompt)
            async for event in session.run_turn():
                InteractiveApp._render_event(event)
            return 0

        return asyncio.run(_run_prompt())
    return asyncio.run(InteractiveApp(session, model_registry=registry).run())
