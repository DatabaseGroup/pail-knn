#!/bin/env python3
"""Run one benchmark command and record its result.

The worker starts this wrapper around knn_stats or LES3 precompute, parses JSON
stdout, and stores success or failure in MongoDB.
"""

import json
import json.decoder
import logging
import sys
import subprocess
import signal

from result_records import augment_success, load_job_from_env, write_failure, write_record

try:
    from bson import json_util
except ModuleNotFoundError:
    json_util = None

logging.basicConfig()
logger = logging.getLogger("benchmark")
logger.setLevel(logging.INFO)


def normalized_exit_code(returncode: int) -> int:
    if 0 < returncode < 256:
        return returncode
    if returncode < 0:
        return min(255, 128 + abs(returncode))
    return 1


def failure_reason(returncode: int) -> tuple[str, str | None]:
    if returncode < 0:
        signal_number = abs(returncode)
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = f"SIG{signal_number}"
        if signal_number == signal.SIGKILL:
            return "signal", f"Process was killed by {signal_name}; this may indicate an out-of-memory kill"
        return "signal", f"Process was terminated by {signal_name}"
    return "exit_code", None


def run_command(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    argv = sys.argv[1:]
    job = load_job_from_env()
    results = run_command(argv)

    if results.returncode != 0:
        logger.warning("Executable failed with exit code {} for {}".format(results.returncode, argv))
        reason, message = failure_reason(results.returncode)
        write_failure(
            job,
            argv,
            reason,
            exit_code=results.returncode,
            stdout=results.stdout,
            stderr=results.stderr,
            message=message,
        )
        return normalized_exit_code(results.returncode)

    try:
        result = json_util.loads(results.stdout) if json_util is not None else json.loads(results.stdout)
    except json.decoder.JSONDecodeError:
        logger.warning("Could not decode executable output {}".format(argv))
        logger.warning("{}".format(results.stdout))
        write_failure(job, argv, "invalid_json", exit_code=1, stdout=results.stdout, stderr=results.stderr)
        return 1

    write_record(augment_success(result, job, argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
