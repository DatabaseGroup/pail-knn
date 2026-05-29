#!/usr/bin/env python3
"""Worker process entry point.

One worker polls MongoDB for jobs in a run. Claimed jobs are executed locally by
WorkerAgent, which also tracks local CPU and memory.
"""

import argparse

import paths
from benchmark_scheduler import WorkerAgent


def main() -> int:
    parser = argparse.ArgumentParser(description="Poll and execute jobs for one benchmark run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--worker-id")
    args = parser.parse_args()

    rc = paths.ensure_project_python()
    if rc is not None:
        return rc

    return WorkerAgent(args.run_id, worker_id=args.worker_id).run()


if __name__ == "__main__":
    raise SystemExit(main())
