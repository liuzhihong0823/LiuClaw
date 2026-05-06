from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ai import Model
from coding_agent.core.agent_session import AgentSession
from coding_agent.core.model_registry import ModelRegistry
from coding_agent.core.multi_agent import TeamRuntime
from coding_agent.core.resource_loader import ResourceLoader
from coding_agent.core.session_manager import SessionManager
from coding_agent.core.types import SessionEvent

EvalCategory = Literal["tool_call_correctness", "long_context_continuity", "multi_step_completion"]
StepMode = Literal["send", "steer", "follow_up"]
EvalDomain = Literal[
    "harness_regression",
    "long_context_governance",
    "memory_benefit",
    "task_recovery_correctness",
    "tool_call_reliability",
]
MetricComparator = Literal["min", "max", "eq"]


@dataclass(slots=True)
class EvalArtifact:
    name: str
    path: str
    description: str = ""


@dataclass(slots=True)
class MetricValue:
    value: float | str | bool
    comparator: MetricComparator = "eq"
    tolerance: float = 0.0


@dataclass(slots=True)
class EvalResult:
    task_id: str
    category: EvalCategory
    passed: bool
    score: float
    reasons: list[str] = field(default_factory=list)
    artifacts: list[EvalArtifact] = field(default_factory=list)
    metrics: dict[str, float | str | bool] = field(default_factory=dict)


@dataclass(slots=True)
class EvalRuntimeProfile:
    name: str = "default"
    thinking: str | None = "medium"
    enable_compaction: bool = False
    attach_team_runtime: bool = False
    summary_text: str | None = None
    prompt_fragments: list[str] = field(default_factory=list)
    tool_execution_mode: Literal["serial", "parallel"] = "serial"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalRunStep:
    prompt: str
    mode: StepMode = "send"
    pre_hook: "EvalHook | None" = None
    post_hook: "EvalHook | None" = None


@dataclass(slots=True)
class EvalStepResult:
    prompt: str
    mode: StepMode
    events: list[SessionEvent]
    final_message: str | None


@dataclass(slots=True)
class EvalRunOutput:
    task_id: str
    category: EvalCategory
    workspace: Path
    session_file: str
    step_results: list[EvalStepResult]
    final_message: str | None
    tool_sequence: list[str]
    artifacts: list[EvalArtifact]
    trace_json_path: str = ""
    replay_markdown_path: str = ""
    trace_artifacts: dict[str, str] = field(default_factory=dict)
    run_data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalContext:
    task_id: str
    category: EvalCategory
    root: Path
    workspace: Path
    model: Model
    session: AgentSession
    session_manager: SessionManager
    resource_loader: ResourceLoader
    model_registry: ModelRegistry
    team_runtime: TeamRuntime | None
    artifacts: list[EvalArtifact] = field(default_factory=list)
    run_data: dict[str, Any] = field(default_factory=dict)
    steps: list[EvalRunStep] = field(default_factory=list)


EvalHook = Callable[[EvalContext], None | Awaitable[None]]
EvalSetup = Callable[[EvalContext], None | Awaitable[None]]
EvalScorer = Callable[["EvalTask", EvalRunOutput], EvalResult]


@dataclass(slots=True)
class EvalTask:
    id: str
    category: EvalCategory
    prompt: str
    setup: EvalSetup
    expected_outcomes: dict[str, Any]
    scorer: EvalScorer
    runtime_profile: EvalRuntimeProfile = field(default_factory=EvalRuntimeProfile)
    domains: tuple[EvalDomain, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class BenchmarkSuite:
    id: str
    task_ids: list[str]
    critical_task_ids: list[str]
    baseline_path: str
    description: str = ""


@dataclass(slots=True)
class BenchmarkBaseline:
    suite_id: str
    metrics: dict[str, MetricValue]
    description: str = ""


@dataclass(slots=True)
class BenchmarkResult:
    suite_id: str
    passed: bool
    results: list[EvalResult]
    metrics: dict[str, float]
    domain_scores: dict[str, float]
    task_reports: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    regression_failures: list[str] = field(default_factory=list)
    artifacts: list[EvalArtifact] = field(default_factory=list)


@dataclass(slots=True)
class ExperimentSpec:
    id: str
    task_ids: list[str]
    control_profile: EvalRuntimeProfile
    candidate_profile: EvalRuntimeProfile
    metrics: list[str]
    metric_directions: dict[str, Literal["higher", "lower"]] = field(default_factory=dict)
    description: str = ""


@dataclass(slots=True)
class ExperimentArmResult:
    label: str
    profile: EvalRuntimeProfile
    results: list[EvalResult]
    metrics: dict[str, float]
    artifacts: list[EvalArtifact] = field(default_factory=list)


@dataclass(slots=True)
class ExperimentComparison:
    id: str
    passed: bool
    control: ExperimentArmResult
    candidate: ExperimentArmResult
    metric_deltas: dict[str, float]
    summary: str = ""
    regressions: list[str] = field(default_factory=list)
    artifacts: list[EvalArtifact] = field(default_factory=list)


@dataclass(slots=True)
class AggregateReport:
    id: str
    metrics: dict[str, float]
    summary: str
    task_reports: list[dict[str, Any]] = field(default_factory=list)
    failure_index: list[str] = field(default_factory=list)
    artifacts: list[EvalArtifact] = field(default_factory=list)
