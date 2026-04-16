from .core.agent_session import AgentSession, SessionEvent


def main(argv=None):
    from .main import main as _main

    return _main(argv)


__all__ = ["AgentSession", "SessionEvent", "main"]
