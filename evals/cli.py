from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .aggregation import rebuild_aggregate_from_file, write_aggregate_report
from .benchmarks import BENCHMARK_SUITES, assert_benchmark_regression, load_benchmark_baseline, run_benchmark_suite, serialize_benchmark_result
from .experiments import EXPERIMENT_SPECS, compare_experiment, serialize_experiment_comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evals.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark")
    benchmark_sub = benchmark.add_subparsers(dest="action", required=True)
    benchmark_run = benchmark_sub.add_parser("run")
    benchmark_run.add_argument("--suite", default="coding_agent_v2")
    benchmark_run.add_argument("--tmp-root", type=Path, required=True)
    benchmark_run.add_argument("--output-dir", type=Path, required=True)

    experiment = subparsers.add_parser("experiment")
    experiment_sub = experiment.add_subparsers(dest="action", required=True)
    experiment_run = experiment_sub.add_parser("run")
    experiment_run.add_argument("--spec", default="memory_on_vs_off")
    experiment_run.add_argument("--tmp-root", type=Path, required=True)
    experiment_run.add_argument("--output-dir", type=Path, required=True)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("path", type=Path)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "benchmark" and args.action == "run":
        return asyncio.run(_run_benchmark(args))
    if args.command == "experiment" and args.action == "run":
        return asyncio.run(_run_experiment(args))
    if args.command == "aggregate":
        report = rebuild_aggregate_from_file(args.path)
        paths = write_aggregate_report(report, args.output_dir)
        print(paths["aggregate_markdown"])
        return 0
    raise ValueError("Unsupported evals command")


async def _run_benchmark(args: argparse.Namespace) -> int:
    suite = BENCHMARK_SUITES[args.suite]
    result = await run_benchmark_suite(suite, tmp_path=args.tmp_root)
    baseline = load_benchmark_baseline(suite.baseline_path)
    failures = assert_benchmark_regression(result, baseline)
    paths = serialize_benchmark_result(result, args.output_dir)
    print(paths["aggregate_markdown"])
    print(f"{result.suite_id}: {'passed' if not failures else 'failed'}")
    return 0 if not failures else 1


async def _run_experiment(args: argparse.Namespace) -> int:
    spec = EXPERIMENT_SPECS[args.spec]
    comparison = await compare_experiment(spec, tmp_path=args.tmp_root)
    paths = serialize_experiment_comparison(comparison, args.output_dir)
    print(paths["experiment_markdown"])
    return 0 if comparison.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
