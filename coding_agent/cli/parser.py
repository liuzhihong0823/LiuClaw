from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """构造 coding-agent 命令行参数解析器。"""

    parser = argparse.ArgumentParser(prog="coding-agent")
    parser.add_argument("prompt", nargs="?", help="Optional one-shot prompt.")
    parser.add_argument("--model")
    parser.add_argument("--session", help="Directly open a specific session file or session id.")
    parser.add_argument("--resume", action="store_true", help="Resume the most recent session in this cwd.")
    parser.add_argument("--continue", dest="continue_session", action="store_true", help="Alias of --resume.")
    parser.add_argument("--fork", help="Fork a specific session file or session id into a new session.")
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--thinking", choices=["low", "medium", "high"])
    parser.add_argument("--new", action="store_true", help="Create a new session.")
    parser.add_argument("--compact", action="store_true", help="Compact the active session and exit.")
    parser.add_argument("--theme")
    return parser


def parse_args(argv: list[str] | None = None):
    """解析命令行参数并返回结果对象。"""

    return build_parser().parse_args(argv)


def parse_trace_args(argv: list[str]) -> argparse.Namespace:
    """解析 replay/trace 子命令参数。"""

    parser = argparse.ArgumentParser(prog="coding-agent")
    subparsers = parser.add_subparsers(dest="trace_command", required=True)
    replay = subparsers.add_parser("replay")
    replay.add_argument("trace_ref")
    trace = subparsers.add_parser("trace")
    trace_sub = trace.add_subparsers(dest="trace_action", required=True)
    show = trace_sub.add_parser("show")
    show.add_argument("trace_ref")
    return parser.parse_args(argv)
