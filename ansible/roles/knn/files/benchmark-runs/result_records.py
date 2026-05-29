"""Result document helpers.

run.py calls this after a command exits. Scheduler metadata is copied into the
result document so plots can ignore failures but still inspect them later.
"""

import json
import os
import shlex
from datetime import datetime, timezone
from typing import Any

import config
from db import connect_to_db, write_to_db


JOB_ENV = "BENCHMARK_JOB_SPEC"
SNIPPET_LIMIT = 4000


def load_job_from_env() -> dict[str, Any] | None:
    raw = os.environ.get(JOB_ENV)
    if not raw:
        return None
    return json.loads(raw)


def command_string(argv: list[str]) -> str:
    return shlex.join([str(part) for part in argv])


def scheduler_metadata(job: dict[str, Any] | None) -> dict[str, Any]:
    if job is None:
        return {}
    return {
        "run_id": job.get("run_id"),
        "job_id": job.get("job_id"),
        "attempt": job.get("attempt"),
        "worker_id": job.get("worker_id"),
        "host": job.get("host"),
        "stage": job.get("stage"),
        "phase": job.get("phase"),
        "experiment": job.get("experiment"),
        "selected_algorithm": job.get("algorithm"),
        "requested_k": job.get("k"),
        "actual_k": job.get("actual_k"),
        "dataset_size_bytes": job.get("dataset_size_bytes"),
        "estimated_memory_bytes": job.get("estimated_memory_bytes"),
        "tokens": job.get("tokens"),
        "memory_multiplier": job.get("memory_multiplier"),
        "dataset_token_count": job.get("dataset_token_count"),
        "puffinn_bytes_per_token": job.get("puffinn_bytes_per_token"),
        "puffinn_index_memory_bytes": job.get("puffinn_index_memory_bytes"),
        "puffinn_scheduler_memory_buffer": job.get("puffinn_scheduler_memory_buffer"),
        "command": command_string(job.get("command_argv", [])),
    }


def metadata_from_job(job: dict[str, Any] | None, argv: list[str]) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "date": datetime.now(timezone.utc),
        "command": command_string(argv),
    }
    if job is None:
        return meta

    meta.update(
        {
            "label": job.get("label"),
            "algorithm": job.get("executable_algorithm") or job.get("algorithm"),
            "selected_algorithm": job.get("algorithm"),
            "dataset": job.get("dataset"),
            "k": job.get("actual_k"),
            "requested_k": job.get("k"),
            "mode": job.get("mode"),
            "sample_size": job.get("sample_size"),
            "concurrency": job.get("concurrency"),
            "experiment": job.get("experiment"),
            "dataset_size_bytes": job.get("dataset_size_bytes"),
            "estimated_memory_bytes": job.get("estimated_memory_bytes"),
            "dataset_token_count": job.get("dataset_token_count"),
            "puffinn_bytes_per_token": job.get("puffinn_bytes_per_token"),
            "puffinn_index_memory_bytes": job.get("puffinn_index_memory_bytes"),
            "puffinn_scheduler_memory_buffer": job.get("puffinn_scheduler_memory_buffer"),
            "run_id": job.get("run_id"),
            "job_id": job.get("job_id"),
        }
    )
    return {key: value for key, value in meta.items() if value is not None}


def _snippet(value: str | bytes | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if len(value) <= SNIPPET_LIMIT:
        return value
    return value[:SNIPPET_LIMIT] + "\n...[truncated]"


def failure_document(
    job: dict[str, Any] | None,
    argv: list[str],
    reason: str,
    *,
    state: str = "failed",
    exit_code: int | None = None,
    stdout: str | bytes | None = None,
    stderr: str | bytes | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    failure: dict[str, Any] = {"reason": reason}
    if exit_code is not None:
        failure["exit_code"] = exit_code
    if message is not None:
        failure["message"] = message
    stdout_snippet = _snippet(stdout)
    stderr_snippet = _snippet(stderr)
    if stdout_snippet is not None:
        failure["stdout"] = stdout_snippet
    if stderr_snippet is not None:
        failure["stderr"] = stderr_snippet

    return {
        "state": state,
        "meta": metadata_from_job(job, argv),
        "scheduler": scheduler_metadata(job),
        "failure": failure,
    }


def augment_success(result: dict[str, Any], job: dict[str, Any] | None, argv: list[str]) -> dict[str, Any]:
    result["state"] = "success"
    result["scheduler"] = scheduler_metadata(job)
    return result


def write_record(document: dict[str, Any]) -> None:
    client, _, collection = connect_to_db(config.db_config)
    try:
        write_to_db(collection, document)
    finally:
        client.close()


def write_failure(
    job: dict[str, Any] | None,
    argv: list[str],
    reason: str,
    *,
    state: str = "failed",
    exit_code: int | None = None,
    stdout: str | bytes | None = None,
    stderr: str | bytes | None = None,
    message: str | None = None,
) -> None:
    write_record(
        failure_document(
            job,
            argv,
            reason,
            state=state,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            message=message,
        )
    )
