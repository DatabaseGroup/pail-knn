#!/usr/bin/env python3
"""Main benchmark CLI.

Builds the job list, annotates scheduler chains, creates a scheduler run, and
starts worker agents. Use `plan` to inspect commands without touching MongoDB.
"""

import argparse
import dataclasses
import functools
import shlex
import sys
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import paths
import memory
from benchmark_scheduler import SchedulerStore, WorkerLauncher, parse_servers, run_coordinator

LABEL_PREFIX = "v18-"
REMOTE_SERVERS = "cluster02,cluster04,cluster05,cluster06,cluster10"
CPU_CORES = 8
PHYSICAL_CORES = 32
EXECUTABLE = "knn_stats"
SAMPLE = 10000
QUERY_TIMEOUT_SECONDS = 2 * 60 * 60
QUERY_K_OFFSET = 1
CONCURRENCY = 1
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 1
SCAL_K = 20

PUFFINN_BYTES_PER_TOKEN = 54
PUFFINN_SCHEDULER_MEMORY_BUFFER = 1.1
PUFFINN_RECALL = "0.95"
MEMORY_DATASET_MULTIPLIER = 3

DEFAULT_DATASETS = [
    "bms-pos-dedup-raw",
    "dblp-all-modes-labels-raw",
    "friendster",
    "kosarak-dedup-raw",
    "livejournal-userswithgroups-dedup-raw",
    "lnonis1",
    "movies-all-modes-labels",
    "netflix",
    "orkut-userswithgroups-dedup-raw",
    "pmc-sentences",
    "trackers",
    "twitter",
]

DEFAULT_K_VALUES = [5, 10, 20, 40, 80]
DEFAULT_SCAL_CONCURRENCIES = [2, 4, 8, 16, 32]
DEFAULT_EXPERIMENTS = ["standard", "scalability"]
RUNNER_ARGV = [str(paths.VENV_PYTHON), str(paths.RUN_SCRIPT)]
STAGE_PRIORITY = {"precompute": 0, "standard": 1, "scalability": 2}


@dataclasses.dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    label: str
    executable_algorithm: str | None
    family: str
    mode: str = "sample"
    vector: str | None = None
    grouping: str | None = None
    min_batch_size: int | None = None
    max_batch_size: int | None = None
    force_concurrency: int | None = None
    supports_standard: bool = True
    supports_scalability: bool = True


ALGORITHMS: dict[str, AlgorithmSpec] = {
    "baseline": AlgorithmSpec("baseline", "linearscan", "baseline", "prefix", vector="0", grouping="identity"),
    "baselinepp": AlgorithmSpec(
        "baselinepp",
        "batchedlinearscanpp",
        "baselinepp",
        "prefix",
        vector="0",
        grouping="identity",
        min_batch_size=4,
        max_batch_size=32,
    ),
    "full": AlgorithmSpec("full", "full", "full", "prefix", vector="0", grouping="identity",
                          supports_scalability=False),
    "slim": AlgorithmSpec("slim", "slim", "slim", "prefix", vector="0", grouping="wsqrt"),
    "witheager": AlgorithmSpec("witheager", "eager", "slim", "prefix", vector="0", grouping="identity",
                               supports_scalability=False),
    "withgrouping": AlgorithmSpec("withgrouping", "grouping", "full", "prefix", vector="0", grouping="wsqrt",
                                  supports_scalability=False),
    "topkbaseline": AlgorithmSpec("topkbaseline", "topkbaseline", "topkbaseline", "prefix", vector="0", grouping="aio"),
    "puffinn": AlgorithmSpec(
        "puffinn",
        "puffinn",
        "puffinn",
        "prefix",
        vector="0",
        grouping="identity",
        force_concurrency=1,
        supports_scalability=False,
    ),
    "partition": AlgorithmSpec("partition", "partition", "partition", "partition"),
    "palloc": AlgorithmSpec("palloc", "palloc", "palloc", "partition"),
    "transformation": AlgorithmSpec("transformation", "transformation", "transformation", "transformation"),
    "les3": AlgorithmSpec("les3", "les3", "les3", "les3"),
    "les3-precompute": AlgorithmSpec(
        "les3-precompute",
        "les3-precompute",
        None,
        "les3-precompute",
        supports_scalability=False,
    ),
}

PRESETS: dict[str, list[str]] = {
    "all": list(ALGORITHMS),
    "query": [name for name in ALGORITHMS if name != "les3-precompute"],
    "prefix": [
        "baseline",
        "baselinepp",
        "full",
        "slim",
        "witheager",
        "withgrouping",
        "topkbaseline",
        "puffinn",
    ],
    "les3-suite": ["les3-precompute", "les3"],
}


@dataclasses.dataclass(frozen=True)
class Config:
    datasets: list[str]
    dataset_dir: Path = paths.DATASET_DIR
    cpu_cores: int = CPU_CORES


@dataclasses.dataclass(frozen=True)
class Job:
    phase: str
    algorithm: str
    label: str
    executable_algorithm: str | None
    experiment: str
    dataset: str
    k: int | None
    actual_k: int | None
    mode: str
    sample_size: int | None
    concurrency: int
    tokens: int
    dataset_size_bytes: int
    estimated_memory_bytes: int
    command_argv: list[str]
    dataset_token_count: int | None = None
    puffinn_bytes_per_token: int | None = None
    puffinn_index_memory_bytes: int | None = None
    puffinn_scheduler_memory_buffer: float | None = None


@dataclasses.dataclass(frozen=True)
class ResourceEstimate:
    dataset_size_bytes: int
    estimated_memory_bytes: int
    dataset_token_count: int | None = None
    puffinn_bytes_per_token: int | None = None
    puffinn_index_memory_bytes: int | None = None
    puffinn_scheduler_memory_buffer: float | None = None


def split_csv_words(value: str) -> list[str]:
    parts: list[str] = []
    for chunk in value.split(","):
        parts.extend(chunk.split())
    return [part for part in parts if part]


def parse_experiments(value: str | None) -> list[str]:
    experiments = list(DEFAULT_EXPERIMENTS) if value is None else split_csv_words(value)
    if "all" in experiments:
        experiments = ["standard", "scalability"]
    invalid = [experiment for experiment in experiments if experiment not in {"standard", "scalability"}]
    if invalid:
        raise ValueError(f"Unknown experiment(s): {', '.join(invalid)}")
    return list(dict.fromkeys(experiments))


def parse_selectors(values: list[str] | None) -> list[str]:
    if not values:
        raise ValueError("--algorithms is required")
    selectors: list[str] = []
    for value in values:
        selectors.extend(split_csv_words(value))
    resolved: list[str] = []
    unknown: list[str] = []
    for selector in selectors:
        if selector in PRESETS:
            resolved.extend(PRESETS[selector])
        elif selector in ALGORITHMS:
            resolved.append(selector)
        else:
            unknown.append(selector)
    if unknown:
        raise ValueError(f"Unknown algorithm or preset: {', '.join(unknown)}")
    return list(dict.fromkeys(resolved))


def build_config(args: argparse.Namespace) -> Config:
    datasets = split_csv_words(args.datasets) if args.datasets else list(DEFAULT_DATASETS)
    return Config(datasets=datasets)


def dataset_size_bytes(cfg: Config, dataset: str) -> int:
    try:
        return (cfg.dataset_dir / dataset).stat().st_size
    except FileNotFoundError:
        return 0


@functools.cache
def dataset_token_count_for_path(path: Path) -> int:
    try:
        with path.open("rb") as dataset_file:
            return sum(len(line.split()) for line in dataset_file)
    except FileNotFoundError:
        return 0


def dataset_token_count(cfg: Config, dataset: str) -> int:
    return dataset_token_count_for_path(cfg.dataset_dir / dataset)


def estimated_memory_bytes(cfg: Config, dataset: str) -> tuple[int, int]:
    size_bytes = dataset_size_bytes(cfg, dataset)
    estimate = int(size_bytes * MEMORY_DATASET_MULTIPLIER)
    return size_bytes, estimate


def puffinn_index_memory_bytes(dataset_tokens: int, dataset_size: int) -> int:
    ideal_memory = dataset_tokens * PUFFINN_BYTES_PER_TOKEN
    return min(ideal_memory,
               int((memory.maximum_memory_reservation() - dataset_size) / PUFFINN_SCHEDULER_MEMORY_BUFFER))


def puffinn_scheduler_memory_bytes(index_memory_bytes: int) -> int:
    return int(index_memory_bytes * PUFFINN_SCHEDULER_MEMORY_BUFFER)


def resource_estimate(spec: AlgorithmSpec, cfg: Config, dataset: str) -> ResourceEstimate:
    size_bytes, memory_bytes = estimated_memory_bytes(cfg, dataset)
    if spec.name == "puffinn":
        token_count = dataset_token_count(cfg, dataset)
        index_memory_bytes = puffinn_index_memory_bytes(token_count, size_bytes)
        total_memory_estimate = index_memory_bytes + size_bytes
        return ResourceEstimate(
            dataset_size_bytes=size_bytes,
            estimated_memory_bytes=puffinn_scheduler_memory_bytes(total_memory_estimate),
            dataset_token_count=token_count,
            puffinn_bytes_per_token=PUFFINN_BYTES_PER_TOKEN,
            puffinn_index_memory_bytes=index_memory_bytes,
            puffinn_scheduler_memory_buffer=PUFFINN_SCHEDULER_MEMORY_BUFFER,
        )
    elif spec.name == "baseline" or spec.name == "baselinepp":
        return ResourceEstimate(dataset_size_bytes=size_bytes, estimated_memory_bytes=size_bytes)
    else:
        return ResourceEstimate(dataset_size_bytes=size_bytes, estimated_memory_bytes=memory_bytes)


def query_command(
        spec: AlgorithmSpec,
        cfg: Config,
        dataset: str,
        k: int,
        sample: int,
        concurrency: int,
        ressources: ResourceEstimate,
) -> list[str]:
    argv = [
        str(paths.BIN_DIR / EXECUTABLE),
        "-l",
        f"{LABEL_PREFIX}{spec.label}",
        "-a",
        str(spec.executable_algorithm),
        "-c",
        str(concurrency),
        "-f",
        str(cfg.dataset_dir / dataset),
        "-k",
        str(k + QUERY_K_OFFSET),
        "-m",
        spec.mode,
        "-s",
        str(sample),
        "--timeout",
        str(QUERY_TIMEOUT_SECONDS),
    ]
    if spec.family == "prefix":
        argv.extend(
            [
                "-v",
                str(spec.vector),
                "-g",
                str(spec.grouping),
                "-y",
                str(spec.min_batch_size if spec.min_batch_size is not None else MIN_BATCH_SIZE),
                "-z",
                str(spec.max_batch_size if spec.max_batch_size is not None else MAX_BATCH_SIZE),
            ]
        )
        if spec.name == "puffinn":
            argv.extend(
                [
                    "--puffinn-memory-bytes",
                    str(ressources.puffinn_index_memory_bytes),
                    "--puffinn-recall",
                    PUFFINN_RECALL,
                ]
            )
    elif spec.family == "les3":
        argv.extend(["--les3-groups", str(paths.LES3_GROUP_DIR / f"{dataset}.groups")])
    return argv


def les3_precompute_command(cfg: Config, dataset: str) -> list[str]:
    dataset_path = cfg.dataset_dir / dataset
    output_dir = paths.LES3_GROUP_DIR
    ptr_output = output_dir / f"{dataset}.ptr"
    ptr_json = output_dir / f"{dataset}.ptr.json"
    python = paths.LES3_DIR / ".venv/bin/python"
    ptr_command = [
        str(python),
        str(paths.LES3_DIR / "ptr_embedding.py"),
        "--input",
        str(dataset_path),
        "--output",
        str(ptr_output),
    ]
    train_command = [
        str(python),
        str(paths.LES3_DIR / "train_l2p.py"),
        "--input",
        str(dataset_path),
        "--ptr",
        str(ptr_output),
        "--output-dir",
        str(output_dir),
        "--label",
        f"{LABEL_PREFIX}les3-precompute",
    ]
    shell_command = (
        f"mkdir -p {shlex.quote(str(output_dir))} && "
        f"{shlex.join(ptr_command)} > {shlex.quote(str(ptr_json))} && "
        f"{shlex.join(train_command)}"
    )
    return ["/bin/sh", "-c", shell_command]


def tokens_for(thread_count: int, cfg: Config) -> int:
    if thread_count < 1:
        return cfg.cpu_cores
    return min(thread_count, cfg.cpu_cores)


def generate_algorithm_jobs(spec: AlgorithmSpec, cfg: Config, experiments: list[str], warnings: list[str]) -> list[Job]:
    jobs: list[Job] = []
    if "standard" in experiments and spec.supports_standard:
        if spec.family == "les3-precompute":
            tokens = tokens_for(PHYSICAL_CORES, cfg)
            for dataset in cfg.datasets:
                resources = resource_estimate(spec, cfg, dataset)
                jobs.append(
                    Job(
                        phase="precompute",
                        algorithm=spec.name,
                        label=f"{LABEL_PREFIX}{spec.label}",
                        executable_algorithm=spec.name,
                        experiment="standard",
                        dataset=dataset,
                        k=None,
                        actual_k=None,
                        mode="precompute",
                        sample_size=None,
                        concurrency=PHYSICAL_CORES,
                        tokens=tokens,
                        dataset_size_bytes=resources.dataset_size_bytes,
                        estimated_memory_bytes=resources.estimated_memory_bytes,
                        command_argv=les3_precompute_command(cfg, dataset),
                    )
                )
        else:
            concurrency = spec.force_concurrency or CONCURRENCY
            tokens = tokens_for(concurrency, cfg)
            for dataset in cfg.datasets:
                resources = resource_estimate(spec, cfg, dataset)
                for k in DEFAULT_K_VALUES:
                    jobs.append(
                        Job(
                            phase="query",
                            algorithm=spec.name,
                            label=f"{LABEL_PREFIX}{spec.label}",
                            executable_algorithm=spec.executable_algorithm,
                            experiment="standard",
                            dataset=dataset,
                            k=k,
                            actual_k=k + QUERY_K_OFFSET,
                            mode=spec.mode,
                            sample_size=SAMPLE,
                            concurrency=concurrency,
                            tokens=tokens,
                            dataset_size_bytes=resources.dataset_size_bytes,
                            estimated_memory_bytes=resources.estimated_memory_bytes,
                            command_argv=query_command(
                                spec,
                                cfg,
                                dataset,
                                k,
                                SAMPLE,
                                concurrency,
                                resources,
                            ),
                            dataset_token_count=resources.dataset_token_count,
                            puffinn_bytes_per_token=resources.puffinn_bytes_per_token,
                            puffinn_index_memory_bytes=resources.puffinn_index_memory_bytes,
                            puffinn_scheduler_memory_buffer=resources.puffinn_scheduler_memory_buffer,
                        )
                    )
    elif "standard" in experiments:
        warnings.append(f"{spec.name} does not support standard experiments; skipping standard jobs")

    if "scalability" in experiments and spec.supports_scalability:
        if spec.family == "les3-precompute":
            warnings.append(f"{spec.name} does not support scalability experiments; skipping scalability jobs")
        else:
            for concurrency in DEFAULT_SCAL_CONCURRENCIES:
                sample = SAMPLE * concurrency
                tokens = tokens_for(concurrency, cfg)
                for dataset in cfg.datasets:
                    resources = resource_estimate(spec, cfg, dataset)
                    jobs.append(
                        Job(
                            phase="query",
                            algorithm=spec.name,
                            label=f"{LABEL_PREFIX}{spec.label}",
                            executable_algorithm=spec.executable_algorithm,
                            experiment="scalability",
                            dataset=dataset,
                            k=SCAL_K,
                            actual_k=SCAL_K + QUERY_K_OFFSET,
                            mode=spec.mode,
                            sample_size=sample,
                            concurrency=concurrency,
                            tokens=tokens,
                            dataset_size_bytes=resources.dataset_size_bytes,
                            estimated_memory_bytes=resources.estimated_memory_bytes,
                            command_argv=query_command(
                                spec,
                                cfg,
                                dataset,
                                SCAL_K,
                                sample,
                                concurrency,
                                resources,
                            ),
                            dataset_token_count=resources.dataset_token_count,
                            puffinn_bytes_per_token=resources.puffinn_bytes_per_token,
                            puffinn_index_memory_bytes=resources.puffinn_index_memory_bytes,
                            puffinn_scheduler_memory_buffer=resources.puffinn_scheduler_memory_buffer,
                        )
                    )
    elif "scalability" in experiments:
        warnings.append(f"{spec.name} does not support scalability experiments; skipping scalability jobs")
    return jobs


def generate_jobs(algorithms: list[str], cfg: Config, experiments: list[str]) -> tuple[list[list[Job]], list[str]]:
    jobs: list[Job] = []
    warnings: list[str] = []
    for name in algorithms:
        jobs.extend(generate_algorithm_jobs(ALGORITHMS[name], cfg, experiments, warnings))

    deduped_warnings = list(dict.fromkeys(warnings))
    precompute_jobs = [job for job in jobs if job.algorithm == "les3-precompute"]
    les3_query_jobs = [job for job in jobs if job.algorithm == "les3"]
    independent_jobs = [job for job in jobs if job.algorithm not in {"les3-precompute", "les3"}]

    if precompute_jobs and les3_query_jobs:
        return [[*precompute_jobs, *independent_jobs], les3_query_jobs], deduped_warnings
    return [jobs] if jobs else [], deduped_warnings


def job_payload(
        job: Job,
        index: int,
        *,
        run_id: str | None = None,
        job_id: str | None = None,
        stage: str | None = None,
) -> dict:
    payload = {
        "id": index,
        "run_id": run_id,
        "job_id": job_id,
        "stage": stage,
        "phase": job.phase,
        "algorithm": job.algorithm,
        "label": job.label,
        "executable_algorithm": job.executable_algorithm,
        "experiment": job.experiment,
        "dataset": job.dataset,
        "k": job.k,
        "actual_k": job.actual_k,
        "mode": job.mode,
        "sample_size": job.sample_size,
        "concurrency": job.concurrency,
        "tokens": job.tokens,
        "dataset_size_bytes": job.dataset_size_bytes,
        "estimated_memory_bytes": job.estimated_memory_bytes,
        "memory_multiplier": None if job.algorithm == "puffinn" else MEMORY_DATASET_MULTIPLIER,
        "dataset_token_count": job.dataset_token_count,
        "puffinn_bytes_per_token": job.puffinn_bytes_per_token,
        "puffinn_index_memory_bytes": job.puffinn_index_memory_bytes,
        "puffinn_scheduler_memory_buffer": job.puffinn_scheduler_memory_buffer,
        "command_argv": job.command_argv,
    }
    return {key: value for key, value in payload.items() if value is not None}


def print_warnings(warnings: Iterable[str]) -> None:
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)


def command_line(job: Job) -> str:
    return shlex.join(RUNNER_ARGV + job.command_argv)


def print_plan(phases: list[list[Job]], print_commands: bool) -> None:
    total = sum(len(phase) for phase in phases)
    print(f"Total jobs: {total}")
    for phase_index, phase in enumerate(phases, start=1):
        print(f"Phase {phase_index}: {len(phase)} jobs")
        counts: dict[tuple[str, str], int] = {}
        for job in phase:
            key = (job.algorithm, job.experiment)
            counts[key] = counts.get(key, 0) + 1
        for (algorithm, experiment), count in sorted(counts.items()):
            print(f"  {algorithm} {experiment}: {count}")
        if print_commands:
            for job in phase:
                print(command_line(job))


def job_sort_key(job: Job) -> tuple[int, str, str, int]:
    return (job.estimated_memory_bytes, job.algorithm, job.dataset, job.k or -1)


def scheduler_stage(job: Job) -> str:
    if job.algorithm == "les3-precompute":
        return "precompute"
    if job.experiment == "scalability":
        return "scalability"
    return "standard"


def build_scheduler_jobs(phases: list[list[Job]], run_id: str) -> list[dict]:
    scheduler_jobs: list[dict] = []
    precompute_datasets = {
        job.dataset for phase in phases for job in phase if scheduler_stage(job) == "precompute"
    }
    jobs = [job for phase in phases for job in phase]
    jobs = sorted(jobs, key=lambda job: (STAGE_PRIORITY[scheduler_stage(job)], *job_sort_key(job)))
    for sequence, job in enumerate(jobs, start=1):
        job_id = f"{sequence:05d}"
        memory, algorithm, dataset, k = job_sort_key(job)
        stage = scheduler_stage(job)
        ready = stage == "precompute" or not (job.algorithm == "les3" and job.dataset in precompute_datasets)
        scheduler_jobs.append(
            {
                "job_id": job_id,
                "sequence": sequence,
                "stage": stage,
                "ready": ready,
                "priority_stage": STAGE_PRIORITY[stage],
                "priority_memory": memory,
                "priority_algorithm": algorithm,
                "priority_dataset": dataset,
                "priority_k": k,
                "job_spec": job_payload(
                    job,
                    sequence,
                    run_id=run_id,
                    job_id=job_id,
                    stage=stage,
                ),
            }
        )
    return scheduler_jobs


def run_phases(phases: list[list[Job]], cfg: Config, args: argparse.Namespace) -> int:
    run_id = args.run_id or uuid4().hex
    scheduler_jobs = build_scheduler_jobs(phases, run_id)
    remote_servers = args.servers or REMOTE_SERVERS
    servers = parse_servers(remote_servers)
    metadata = {
        "label_prefix": LABEL_PREFIX,
        "datasets": cfg.datasets,
        "remote_servers": remote_servers,
        "cpu_cores": cfg.cpu_cores,
        "expected_workers": 0 if args.no_launch_workers else len(servers) * args.workers_per_host,
    }
    store = SchedulerStore()
    launcher = None
    try:
        store.create_run(run_id, scheduler_jobs, metadata)
        print(f"Created benchmark run {run_id} with {len(scheduler_jobs)} jobs", flush=True)
        if not args.no_launch_workers:
            if not servers:
                print("No worker servers configured; use --servers or --no-launch-workers", file=sys.stderr)
                return 2
            launcher = WorkerLauncher()
            launcher.start(run_id, servers, args.workers_per_host)
        return run_coordinator(run_id=run_id, store=store, launcher=launcher, poll_seconds=args.poll_seconds)
    finally:
        store.close()


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--algorithms", action="append", help="Comma or space separated algorithms/presets to include")
    parser.add_argument("--experiments", help="standard, scalability, all, or a comma separated combination")
    parser.add_argument("--datasets", help="Comma or space separated dataset names for focused runs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and run unified kNN benchmark job queues.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List algorithms and presets")

    plan_parser = subparsers.add_parser("plan", help="Print the generated benchmark plan")
    add_common_options(plan_parser)
    plan_parser.add_argument("--print-commands", action="store_true", help="Print full wrapped benchmark commands")

    run_parser = subparsers.add_parser("run", help="Run the generated benchmark plan")
    add_common_options(run_parser)
    run_parser.add_argument("--print-plan", action="store_true", help="Print the plan before executing")
    run_parser.add_argument("--servers", help="Comma or space separated worker hosts; defaults to REMOTE_SERVERS")
    run_parser.add_argument(
        "--workers-per-host",
        type=int,
        default=1,
        help="Worker agent processes to start per host; only 1 is supported",
    )
    run_parser.add_argument("--no-launch-workers", action="store_true",
                            help="Create and monitor the run without SSH workers")
    run_parser.add_argument("--run-id", help="Explicit run id for the scheduler documents")
    run_parser.add_argument("--poll-seconds", type=float, default=5.0, help="Coordinator polling interval")

    worker_parser = subparsers.add_parser("worker", help="Run a local worker agent for an existing scheduler run")
    worker_parser.add_argument("--run-id", required=True)
    worker_parser.add_argument("--worker-id")

    status_parser = subparsers.add_parser("status", help="Print scheduler status for an existing run")
    status_parser.add_argument("--run-id", required=True)

    return parser


def list_items() -> int:
    print("Algorithms:")
    for name, spec in ALGORITHMS.items():
        scalability = "yes" if spec.supports_scalability else "no"
        print(f"  {name}: label={spec.label}, family={spec.family}, scalability={scalability}")
    print("Presets:")
    for name, algorithms in PRESETS.items():
        print(f"  {name}: {', '.join(algorithms)}")
    return 0


def command_plan(args: argparse.Namespace) -> int:
    try:
        algorithms = parse_selectors(args.algorithms)
        experiments = parse_experiments(args.experiments)
        cfg = build_config(args)
        phases, warnings = generate_jobs(algorithms, cfg, experiments)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print_warnings(warnings)
    print_plan(phases, args.print_commands)
    return 0


def command_run(args: argparse.Namespace) -> int:
    try:
        algorithms = parse_selectors(args.algorithms)
        experiments = parse_experiments(args.experiments)
        cfg = build_config(args)
        phases, warnings = generate_jobs(algorithms, cfg, experiments)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print_warnings(warnings)
    if args.workers_per_host != 1:
        print("--workers-per-host must be 1; concurrency is controlled by benchmark jobs", file=sys.stderr)
        return 2
    if args.poll_seconds <= 0:
        print("--poll-seconds must be positive", file=sys.stderr)
        return 2
    if args.print_plan:
        print_plan(phases, print_commands=False)
    if not phases:
        print("No jobs generated", file=sys.stderr)
        return 1
    return run_phases(phases, cfg, args)


def command_worker(args: argparse.Namespace) -> int:
    from benchmark_scheduler import WorkerAgent

    worker = WorkerAgent(args.run_id, worker_id=args.worker_id)
    return worker.run()


def command_status(args: argparse.Namespace) -> int:
    from benchmark_scheduler import print_status

    return print_status(args.run_id)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command in {"run", "worker", "status"}:
        rc = paths.ensure_project_python()
        if rc is not None:
            return rc
    try:
        if args.command == "list":
            return list_items()
        if args.command == "plan":
            return command_plan(args)
        if args.command == "run":
            return command_run(args)
        if args.command == "worker":
            return command_worker(args)
        if args.command == "status":
            return command_status(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
