from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from ai import AssistantMessage, StreamEvent, ToolCall, ToolResultMessage


@dataclass(slots=True)
class FakeSession:
    events: list[StreamEvent]

    async def consume(self) -> AsyncIterator[StreamEvent]:
        for event in self.events:
            yield event

    async def close(self) -> None:
        return None


class EvalStreamBuilder:
    """Build deterministic stream sessions keyed by eval task id."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self._call_counts: dict[str, int] = {}

    async def __call__(self, model, context, thinking, registry=None, signal=None):
        _ = thinking, registry, signal
        latest_user = next(
            (message.content for message in reversed(context.history) if getattr(message, "role", "") == "user"),
            "",
        )
        task_id = self._extract_task_id(str(latest_user))
        self._call_counts[task_id] = self._call_counts.get(task_id, 0) + 1
        tool_results = [message for message in context.history if isinstance(message, ToolResultMessage)]
        scenario = getattr(self, f"_scenario_{task_id.replace('-', '_')}", None)
        if scenario is None:
            return self._done(model, f"unhandled eval task: {task_id}")
        return await scenario(model, str(latest_user), tool_results)

    @staticmethod
    def _extract_task_id(text: str) -> str:
        match = re.search(r"\[eval:([^\]]+)\]", text)
        return match.group(1) if match else ""

    @staticmethod
    def _has_profile_flag(text: str, key: str, value: str) -> bool:
        return f"[profile-{key}:{value}]" in text

    @staticmethod
    def _done(model, content: str, *, tool_calls: list[ToolCall] | None = None) -> FakeSession:
        return FakeSession(
            [
                StreamEvent(type="start", provider=model.provider, model=model),
                StreamEvent(
                    type="done",
                    provider=model.provider,
                    model=model,
                    assistantMessage=AssistantMessage(content=content, toolCalls=tool_calls or []),
                ),
            ]
        )

    async def _scenario_tool_single_write(self, model, latest_user: str, tool_results: list[ToolResultMessage]) -> FakeSession:
        _ = latest_user
        if not tool_results:
            return self._done(
                model,
                "writing a file",
                tool_calls=[
                    ToolCall(id="write-single", name="write", arguments={"path": "notes/alpha.txt", "content": "alpha"})
                ],
            )
        return self._done(model, "single-write complete")

    async def _scenario_tool_chain_write_read(self, model, latest_user: str, tool_results: list[ToolResultMessage]) -> FakeSession:
        _ = latest_user
        if not tool_results:
            return self._done(
                model,
                "write then read",
                tool_calls=[
                    ToolCall(id="chain-write", name="write", arguments={"path": "chain.txt", "content": "beta"}),
                    ToolCall(id="chain-read", name="read", arguments={"path": "chain.txt"}),
                ],
            )
        return self._done(model, "multi-tool chain confirmed beta")

    async def _scenario_tool_invalid_args_recovery(self, model, latest_user: str, tool_results: list[ToolResultMessage]) -> FakeSession:
        _ = latest_user
        if not tool_results:
            return self._done(
                model,
                "first attempt invalid",
                tool_calls=[ToolCall(id="invalid-write", name="write", arguments={"content": "missing-path"})],
            )
        if tool_results[-1].isError:
            return self._done(
                model,
                "retry with corrected args",
                tool_calls=[ToolCall(id="fixed-write", name="write", arguments={"path": "fixed.txt", "content": "recovered"})],
            )
        return self._done(model, "recovered after validation")

    async def _scenario_tool_runtime_error_explained(self, model, latest_user: str, tool_results: list[ToolResultMessage]) -> FakeSession:
        _ = latest_user
        if not tool_results:
            return self._done(
                model,
                "attempting missing read",
                tool_calls=[ToolCall(id="missing-read", name="read", arguments={"path": "missing.txt"})],
            )
        return self._done(model, "unable to read missing.txt because it does not exist")

    async def _scenario_continuity_resume_retains_fact(self, model, latest_user: str, tool_results: list[ToolResultMessage]) -> FakeSession:
        _ = tool_results
        if "what is my favorite_color=green" in latest_user:
            return self._done(model, "favorite_color=green")
        if "favorite_color" in latest_user:
            return self._done(model, "stored favorite_color=green")
        return self._done(model, "favorite_color not found")

    async def _scenario_continuity_branch_retains_fact(self, model, latest_user: str, tool_results: list[ToolResultMessage]) -> FakeSession:
        _ = tool_results
        if "what is branch_fact=delta?" in latest_user:
            return self._done(model, "branch_fact=delta")
        if "branch_fact" in latest_user:
            return self._done(model, "stored branch_fact=delta")
        if "mainline changes" in latest_user:
            return self._done(model, "mainline noted")
        return self._done(model, "branch_fact not found")

    async def _scenario_continuity_compaction_retains_fact(self, model, latest_user: str, tool_results: list[ToolResultMessage]) -> FakeSession:
        _ = tool_results
        if "deployment_region=ap-southeast-1" in latest_user:
            return self._done(model, "compaction preserved deployment_region=ap-southeast-1")
        return self._done(model, "deployment_region missing")

    async def _scenario_continuity_steer_follow_up(self, model, latest_user: str, tool_results: list[ToolResultMessage]) -> FakeSession:
        _ = tool_results
        text = latest_user.split("]", 1)[-1].strip()
        return self._done(model, f"continuity-steer-follow:{text}")

    async def _scenario_multi_step_plan_and_execute(self, model, latest_user: str, tool_results: list[ToolResultMessage]) -> FakeSession:
        _ = latest_user
        if not tool_results:
            return self._done(
                model,
                "phase one",
                tool_calls=[ToolCall(id="plan-write", name="write", arguments={"path": "plan.txt", "content": "phase-1 complete"})],
            )
        if len(tool_results) == 1:
            return self._done(
                model,
                "phase two",
                tool_calls=[ToolCall(id="plan-read", name="read", arguments={"path": "plan.txt"})],
            )
        return self._done(model, "phase-1 complete -> final answer")

    async def _scenario_multi_step_answer_from_tool_result(self, model, latest_user: str, tool_results: list[ToolResultMessage]) -> FakeSession:
        _ = latest_user
        if not tool_results:
            return self._done(
                model,
                "fetch the answer",
                tool_calls=[ToolCall(id="answer-read", name="read", arguments={"path": "answer.txt"})],
            )
        answer = str(tool_results[-1].content)
        return self._done(model, f"The tool result is {answer}.")

    async def _scenario_multi_step_file_artifact_written(self, model, latest_user: str, tool_results: list[ToolResultMessage]) -> FakeSession:
        _ = latest_user
        if not tool_results:
            return self._done(
                model,
                "create report",
                tool_calls=[ToolCall(id="report-write", name="write", arguments={"path": "artifacts/report.md", "content": "# Report\nstatus: ok\n"})],
            )
        return self._done(model, "artifact report created")

    async def _scenario_multi_step_team_runtime_review(self, model, latest_user: str, tool_results: list[ToolResultMessage]) -> FakeSession:
        pending_request_id = self._extract_request_id(latest_user)
        if pending_request_id:
            if any(result.toolName == "review_plan" for result in tool_results):
                return self._done(model, "plan approved")
            return self._done(
                model,
                "reviewing pending plan",
                tool_calls=[
                    ToolCall(
                        id="review-plan",
                        name="review_plan",
                        arguments={"request_id": pending_request_id, "approve": True, "feedback": "approved"},
                    )
                ],
            )
        if not tool_results:
            return self._done(
                model,
                "spawning teammate",
                tool_calls=[
                    ToolCall(
                        id="spawn-worker",
                        name="spawn_agent",
                        arguments={
                            "name": "alice",
                            "role": "coder",
                            "task_prompt": "[eval:worker_submit_plan] propose a safe update plan",
                        },
                    )
                ],
            )
        return self._done(model, "worker spawned")

    async def _scenario_worker_submit_plan(self, model, latest_user: str, tool_results: list[ToolResultMessage]) -> FakeSession:
        if "<team_inbox>" in latest_user:
            request_id = self._extract_request_id(latest_user)
            if "shutdown_request" in latest_user:
                return self._done(
                    model,
                    "processing shutdown",
                    tool_calls=[
                        ToolCall(
                            id="shutdown-worker",
                            name="respond_shutdown",
                            arguments={"request_id": request_id, "approve": True, "reason": "done"},
                        )
                    ],
                )
            if "plan_approval_response" in latest_user:
                return self._done(model, "worker saw approval")
            return self._done(model, "worker resumed from inbox")
        if not tool_results:
            return self._done(
                model,
                "submitting plan",
                tool_calls=[ToolCall(id="submit-plan", name="submit_plan", arguments={"plan": "Update docs carefully"})],
            )
        return self._done(model, "plan submitted")

    async def _scenario_memory_profile_probe(self, model, latest_user: str, tool_results: list[ToolResultMessage]) -> FakeSession:
        memory_enabled = self._has_profile_flag(latest_user, "memory_enabled", "True")
        if "what is durable_fact=violet after memory restore?" in latest_user:
            if memory_enabled:
                return self._done(model, "memory restored durable_fact=violet")
            return self._done(model, "memory unavailable")
        if "durable_fact" in latest_user:
            return self._done(model, "stored durable_fact=violet")
        return self._done(model, "memory probe idle")

    async def _scenario_overflow_recovery_profile_probe(self, model, latest_user: str, tool_results: list[ToolResultMessage]) -> FakeSession:
        if "recover from overflow" in latest_user:
            return self._done(model, "overflow recovered durable_fact=violet")
        if "durable_fact" in latest_user:
            return self._done(model, "stored durable_fact=violet")
        return self._done(model, "overflow probe idle")

    @staticmethod
    def _extract_request_id(text: str) -> str:
        match = re.search(r'"request_id":\s*"([^"]+)"', text)
        return match.group(1) if match else ""
