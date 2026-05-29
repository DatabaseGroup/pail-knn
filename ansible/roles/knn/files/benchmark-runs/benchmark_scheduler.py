#!/usr/bin/env python3
"""Mongo-backed benchmark scheduler.

The coordinator refreshes job readiness and records skips. Workers poll for
ready jobs that fit their free CPU and memory, run them, renew leases, and write
final state.
"""

import json
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable
from uuid import uuid4

import config
import paths
from memory import memory_info, memory_reserve, usable_memory
from db import connect_to_db
from result_records import write_failure


TERMINAL_STATES = {"success", "failed", "skipped"}
LEASE_SECONDS = 120
LEASE_RENEW_SECONDS = 30
WORKER_POLL_SECONDS = 2
COORDINATOR_POLL_SECONDS = 5
MAX_LEASE_ATTEMPTS = 2
EXIT_MEMORY_UNFIT = 75


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def command_string(argv: Iterable[Any]) -> str:
    return shlex.join([str(part) for part in argv])


def skip_reason_for_job(
    job: dict[str, Any],
    failed_precompute: set[str],
) -> tuple[str, str] | None:
    if job.get("algorithm") == "les3" and job.get("dataset") in failed_precompute:
        return "dependency_failed", "LES3 precompute failed for this dataset"
    return None


def scheduler_collection_name() -> str:
    return str(config.db_config.get("scheduler_collection") or f"{config.db_config['collection']}_scheduler")


class SchedulerStore:
    def __init__(self) -> None:
        from pymongo import ReturnDocument

        self.return_document_after = ReturnDocument.AFTER
        self.client, self.database, self.result_collection = connect_to_db(config.db_config)
        self.collection = self.database.get_collection(scheduler_collection_name())
        self._ensure_indexes()

    def close(self) -> None:
        self.client.close()

    def _ensure_indexes(self) -> None:
        from pymongo.errors import OperationFailure

        try:
            self.collection.create_index(
                [("type", 1), ("run_id", 1)],
                unique=True,
                partialFilterExpression={"type": "benchmark_run"},
                name="unique_benchmark_run",
            )
        except OperationFailure as exc:
            raise RuntimeError(
                "Could not create unique benchmark run index. Remove duplicate benchmark_run documents "
                "with the same run_id, or drop a conflicting index definition."
            ) from exc
        self.collection.create_index(
            [("type", 1), ("run_id", 1), ("job_id", 1)],
            unique=True,
            partialFilterExpression={"type": "benchmark_job"},
        )
        self.collection.create_index([("type", 1), ("run_id", 1), ("state", 1), ("ready", 1)])
        self.collection.create_index([("type", 1), ("run_id", 1), ("lease_expires_at", 1)])
        self.collection.create_index([("type", 1), ("run_id", 1), ("worker_id", 1)])
        self.result_collection.create_index(
            [("scheduler.run_id", 1), ("scheduler.job_id", 1)],
            unique=True,
            partialFilterExpression={"scheduler.run_id": {"$exists": True}, "scheduler.job_id": {"$exists": True}},
        )

    def create_run(self, run_id: str, jobs: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
        now = utcnow()
        run_doc = {
            "type": "benchmark_run",
            "run_id": run_id,
            "state": "running",
            "created_at": now,
            "updated_at": now,
            "metadata": metadata,
        }
        self.collection.insert_one(run_doc)
        if not jobs:
            self.collection.update_one(
                {"type": "benchmark_run", "run_id": run_id},
                {"$set": {"state": "completed", "finished_at": now, "updated_at": now}},
            )
            return
        self.collection.insert_many(
            [
                {
                    "type": "benchmark_job",
                    "run_id": run_id,
                    "job_id": job["job_id"],
                    "sequence": job["sequence"],
                    "stage": job["stage"],
                    "ready": job["ready"],
                    "state": "pending",
                    "priority_stage": job["priority_stage"],
                    "priority_memory": job["priority_memory"],
                    "priority_algorithm": job["priority_algorithm"],
                    "priority_dataset": job["priority_dataset"],
                    "priority_k": job["priority_k"],
                    "job_spec": job["job_spec"],
                    "attempts": 0,
                    "created_at": now,
                    "updated_at": now,
                }
                for job in jobs
            ]
        )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self.collection.find_one({"type": "benchmark_run", "run_id": run_id})

    def set_run_state(self, run_id: str, state: str, **fields: Any) -> None:
        fields.update({"state": state, "updated_at": utcnow()})
        if state in {"completed", "failed", "stopping"}:
            fields.setdefault("finished_at", utcnow())
        self.collection.update_one({"type": "benchmark_run", "run_id": run_id}, {"$set": fields})

    def heartbeat_worker(
        self,
        run_id: str,
        worker_id: str,
        host: str,
        cpu_capacity: int,
        cpu_free: int,
        memory_total: int,
        memory_usable: int,
        memory_free: int,
        active_jobs: int,
    ) -> None:
        now = utcnow()
        self.collection.update_one(
            {"type": "benchmark_worker", "run_id": run_id, "worker_id": worker_id},
            {
                "$set": {
                    "host": host,
                    "cpu_capacity": cpu_capacity,
                    "cpu_free": cpu_free,
                    "memory_total": memory_total,
                    "memory_usable": memory_usable,
                    "memory_free": memory_free,
                    "active_jobs": active_jobs,
                    "last_seen": now,
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )

    def active_worker_capacities(self, run_id: str, max_age_seconds: int = 180) -> list[dict[str, Any]]:
        cutoff = utcnow() - timedelta(seconds=max_age_seconds)
        return list(
            self.collection.find(
                {
                    "type": "benchmark_worker",
                    "run_id": run_id,
                    "last_seen": {"$gte": cutoff},
                }
            )
        )

    def claim_next_job(
        self,
        run_id: str,
        worker_id: str,
        host: str,
        cpu_free: int,
        memory_free: int,
        lease_seconds: int = LEASE_SECONDS,
    ) -> dict[str, Any] | None:
        run_doc = self.get_run(run_id)
        if not run_doc or run_doc.get("state") != "running":
            return None
        now = utcnow()
        lease_expires_at = now + timedelta(seconds=lease_seconds)
        doc = self.collection.find_one_and_update(
            {
                "type": "benchmark_job",
                "run_id": run_id,
                "state": "pending",
                "ready": True,
                "job_spec.tokens": {"$lte": cpu_free},
                "job_spec.estimated_memory_bytes": {"$lte": memory_free},
            },
            {
                "$set": {
                    "state": "running",
                    "worker_id": worker_id,
                    "host": host,
                    "lease_owner": worker_id,
                    "lease_expires_at": lease_expires_at,
                    "started_at": now,
                    "updated_at": now,
                },
                "$inc": {"attempts": 1},
            },
            sort=[
                ("priority_stage", 1),
                ("priority_memory", 1),
                ("priority_algorithm", 1),
                ("priority_dataset", 1),
                ("priority_k", 1),
                ("sequence", 1),
            ],
            return_document=self.return_document_after,
        )
        if doc is not None:
            attempt = int(doc.get("attempts", 1))
            self.collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"job_spec.attempt": attempt, "job_spec.worker_id": worker_id, "job_spec.host": host}},
            )
            doc["job_spec"]["attempt"] = attempt
            doc["job_spec"]["worker_id"] = worker_id
            doc["job_spec"]["host"] = host
        return doc

    def renew_lease(self, run_id: str, job_id: str, lease_owner: str, lease_seconds: int = LEASE_SECONDS) -> bool:
        result = self.collection.update_one(
            {
                "type": "benchmark_job",
                "run_id": run_id,
                "job_id": job_id,
                "state": "running",
                "lease_owner": lease_owner,
            },
            {"$set": {"lease_expires_at": utcnow() + timedelta(seconds=lease_seconds), "updated_at": utcnow()}},
        )
        return result.modified_count == 1

    def complete_job(
        self,
        run_id: str,
        job_id: str,
        lease_owner: str,
        state: str,
        *,
        failure: dict[str, Any] | None = None,
    ) -> bool:
        fields: dict[str, Any] = {
            "state": state,
            "finished_at": utcnow(),
            "updated_at": utcnow(),
        }
        if failure is not None:
            fields["failure"] = failure
        result = self.collection.update_one(
            {
                "type": "benchmark_job",
                "run_id": run_id,
                "job_id": job_id,
                "state": "running",
                "lease_owner": lease_owner,
            },
            {
                "$set": fields,
                "$unset": {"lease_expires_at": "", "lease_owner": ""},
                "$push": {
                    "attempt_log": {
                        "state": state,
                        "finished_at": fields["finished_at"],
                        "failure": failure,
                        "lease_owner": lease_owner,
                    }
                },
            },
        )
        return result.modified_count == 1

    def expire_leases(self, run_id: str, max_attempts: int = MAX_LEASE_ATTEMPTS) -> list[dict[str, Any]]:
        now = utcnow()
        expired = list(
            self.collection.find(
                {
                    "type": "benchmark_job",
                    "run_id": run_id,
                    "state": "running",
                    "lease_expires_at": {"$lt": now},
                }
            )
        )
        terminal_failures: list[dict[str, Any]] = []
        for job in expired:
            attempts = int(job.get("attempts", 0))
            if attempts >= max_attempts:
                failure = {
                    "reason": "lease_expired",
                    "exit_code": None,
                    "message": f"Worker lease expired after {attempts} attempt(s)",
                }
                result = self.collection.update_one(
                    {
                        "_id": job["_id"],
                        "state": "running",
                        "lease_owner": job.get("lease_owner"),
                    },
                    {
                        "$set": {
                            "state": "failed",
                            "failure": failure,
                            "finished_at": now,
                            "updated_at": now,
                        },
                        "$unset": {"lease_expires_at": "", "lease_owner": ""},
                        "$push": {"attempt_log": {"state": "failed", "finished_at": now, "failure": failure}},
                    },
                )
                if result.modified_count == 1:
                    job["failure"] = failure
                    terminal_failures.append(job)
                continue
            result = self.collection.update_one(
                {"_id": job["_id"], "state": "running", "lease_owner": job.get("lease_owner")},
                {
                    "$set": {"state": "pending", "updated_at": now},
                    "$unset": {"lease_expires_at": "", "lease_owner": "", "worker_id": "", "host": ""},
                    "$push": {
                        "attempt_log": {
                            "state": "requeued",
                            "finished_at": now,
                            "failure": {"reason": "lease_expired", "message": "Worker lease expired; requeued"},
                        }
                    },
                },
            )
            if result.modified_count == 1:
                print(f"Requeued {job['job_id']} after worker lease expired", file=sys.stderr, flush=True)
        return terminal_failures

    def failed_precompute_datasets(self, run_id: str) -> set[str]:
        failed_precompute: set[str] = set()
        cursor = self.collection.find(
            {
                "type": "benchmark_job",
                "run_id": run_id,
                "state": "failed",
            },
            {"job_spec": 1, "failure": 1},
        )
        for doc in cursor:
            job = doc.get("job_spec", {})
            if job.get("algorithm") == "les3-precompute":
                failed_precompute.add(str(job.get("dataset")))
        return failed_precompute

    def apply_skips(self, run_id: str) -> int:
        failed_precompute = self.failed_precompute_datasets(run_id)
        skipped = 0
        pending = list(self.collection.find({"type": "benchmark_job", "run_id": run_id, "state": "pending"}))
        for doc in pending:
            job = doc.get("job_spec", {})
            skip = skip_reason_for_job(job, failed_precompute)
            if skip is not None and self.mark_skipped(doc, skip[0], skip[1]):
                skipped += 1
        return skipped

    def mark_skipped(self, doc: dict[str, Any], reason: str, message: str) -> bool:
        now = utcnow()
        result = self.collection.update_one(
            {"_id": doc["_id"], "state": "pending"},
            {
                "$set": {
                    "state": "skipped",
                    "failure": {"reason": reason, "message": message},
                    "finished_at": now,
                    "updated_at": now,
                }
            },
        )
        if result.modified_count != 1:
            return False
        job = doc["job_spec"]
        value = f"cores={job.get('concurrency')}" if job.get("stage") == "scalability" else f"k={job.get('k')}"
        print(f"Skipping {job.get('algorithm')} {job.get('dataset')} {value}: {message}", file=sys.stderr)
        write_failure(
            job,
            [str(part) for part in job.get("command_argv", [])],
            reason,
            state="skipped",
            message=message,
        )
        return True

    def mark_memory_unfit_jobs(self, run_id: str, *, min_active_workers: int = 1) -> int:
        workers = self.active_worker_capacities(run_id)
        if len(workers) < min_active_workers:
            return 0
        max_cpu = max(int(worker.get("cpu_capacity", 0)) for worker in workers)
        max_memory = max(int(worker.get("memory_usable", 0)) for worker in workers)
        unfit = list(
            self.collection.find(
                {
                    "type": "benchmark_job",
                    "run_id": run_id,
                    "state": "pending",
                    "$or": [
                        {"job_spec.tokens": {"$gt": max_cpu}},
                        {"job_spec.estimated_memory_bytes": {"$gt": max_memory}},
                    ],
                }
            )
        )
        marked = 0
        for doc in unfit:
            job = doc["job_spec"]
            reason = "memory_unfit"
            message = (
                f"Estimated resources exceed all active workers: tokens={job.get('tokens')} "
                f"memory={job.get('estimated_memory_bytes')}"
            )
            now = utcnow()
            result = self.collection.update_one(
                {"_id": doc["_id"], "state": "pending"},
                {
                    "$set": {
                        "state": "failed",
                        "failure": {"reason": reason, "exit_code": EXIT_MEMORY_UNFIT, "message": message},
                        "finished_at": now,
                        "updated_at": now,
                    }
                },
            )
            if result.modified_count == 1:
                write_failure(
                    job,
                    [str(part) for part in job.get("command_argv", [])],
                    reason,
                    exit_code=EXIT_MEMORY_UNFIT,
                    message=message,
                )
                marked += 1
        return marked

    def refresh_ready_jobs(self, run_id: str) -> int:
        run_doc = self.get_run(run_id)
        if not run_doc or run_doc.get("state") != "running":
            return 0
        precompute_docs = list(
            self.collection.find(
                {
                    "type": "benchmark_job",
                    "run_id": run_id,
                    "stage": "precompute",
                },
                {"job_spec.dataset": 1, "state": 1},
            )
        )
        precompute_datasets = {str(doc.get("job_spec", {}).get("dataset")) for doc in precompute_docs}
        successful_precompute = {
            str(doc.get("job_spec", {}).get("dataset")) for doc in precompute_docs if doc.get("state") == "success"
        }
        newly_ready: list[Any] = []
        blocked = list(
            self.collection.find(
                {
                    "type": "benchmark_job",
                    "run_id": run_id,
                    "state": "pending",
                    "ready": {"$ne": True},
                }
            )
        )
        for doc in blocked:
            if self._job_is_ready(doc, precompute_datasets, successful_precompute):
                newly_ready.append(doc["_id"])
        if not newly_ready:
            return 0
        result = self.collection.update_many(
            {"_id": {"$in": newly_ready}, "state": "pending"},
            {"$set": {"ready": True, "updated_at": utcnow()}},
        )
        return int(result.modified_count)

    @staticmethod
    def _job_is_ready(
        doc: dict[str, Any],
        precompute_datasets: set[str],
        successful_precompute: set[str],
    ) -> bool:
        stage = str(doc.get("stage"))
        job = doc.get("job_spec", {})
        if stage == "precompute":
            return True
        dataset = str(job.get("dataset"))
        if job.get("algorithm") == "les3" and dataset in precompute_datasets:
            return dataset in successful_precompute
        return True

    def complete_run_if_done(self, run_id: str) -> bool:
        run_doc = self.get_run(run_id)
        if not run_doc or run_doc.get("state") != "running":
            return False
        active_count = self.collection.count_documents(
            {
                "type": "benchmark_job",
                "run_id": run_id,
                "state": {"$nin": list(TERMINAL_STATES)},
            }
        )
        if active_count == 0:
            self.set_run_state(run_id, "completed")
            return True
        return False

    def state_counts(self, run_id: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.collection.aggregate(
            [
                {"$match": {"type": "benchmark_job", "run_id": run_id}},
                {"$group": {"_id": "$state", "count": {"$sum": 1}}},
            ]
        ):
            counts[str(item["_id"])] = int(item["count"])
        return counts

    def readiness_counts(self, run_id: str) -> dict[str, int]:
        return {
            "pending_ready": self.collection.count_documents(
                {"type": "benchmark_job", "run_id": run_id, "state": "pending", "ready": True}
            ),
            "pending_blocked": self.collection.count_documents(
                {"type": "benchmark_job", "run_id": run_id, "state": "pending", "ready": {"$ne": True}}
            ),
            "running": self.collection.count_documents({"type": "benchmark_job", "run_id": run_id, "state": "running"}),
        }

    def terminal_counts(self, run_id: str) -> tuple[int, int]:
        failed = self.collection.count_documents({"type": "benchmark_job", "run_id": run_id, "state": "failed"})
        skipped = self.collection.count_documents({"type": "benchmark_job", "run_id": run_id, "state": "skipped"})
        return failed, skipped


@dataclass
class ResourcePool:
    cpu_capacity: int
    cpu_used: int = 0
    memory_reserved: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> tuple[int, int, int, int]:
        total, available = memory_info()
        reserve = memory_reserve(total)
        usable = usable_memory(total)
        with self.lock:
            free_cpu = max(0, self.cpu_capacity - self.cpu_used)
            free_by_estimate = max(0, usable - self.memory_reserved)
        free_by_current = max(0, available - reserve)
        return free_cpu, total, usable, min(free_by_estimate, free_by_current)

    def reserve(self, job: dict[str, Any]) -> None:
        with self.lock:
            self.cpu_used += int(job.get("tokens", 1))
            self.memory_reserved += int(job.get("estimated_memory_bytes", 0))

    def release(self, job: dict[str, Any]) -> None:
        with self.lock:
            self.cpu_used = max(0, self.cpu_used - int(job.get("tokens", 1)))
            self.memory_reserved = max(0, self.memory_reserved - int(job.get("estimated_memory_bytes", 0)))


class WorkerAgent:
    def __init__(
        self,
        run_id: str,
        *,
        worker_id: str | None = None,
        poll_seconds: float = WORKER_POLL_SECONDS,
        lease_seconds: int = LEASE_SECONDS,
    ) -> None:
        self.run_id = run_id
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}-{uuid4().hex[:8]}"
        self.host = socket.gethostname()
        self.poll_seconds = poll_seconds
        self.lease_seconds = lease_seconds
        self.store = SchedulerStore()
        run_doc = self.store.get_run(run_id)
        metadata = run_doc.get("metadata", {}) if run_doc else {}
        self.resources = ResourcePool(cpu_capacity=max(1, int(metadata.get("cpu_cores", os.cpu_count() or 1))))
        self.active: list[threading.Thread] = []
        self.active_lock = threading.Lock()

    def close(self) -> None:
        self.store.close()

    def run(self) -> int:
        print(f"Worker {self.worker_id} polling run {self.run_id}", flush=True)
        try:
            while True:
                self._reap_threads()
                free_cpu, total_memory, usable, free_memory = self.resources.snapshot()
                self.store.heartbeat_worker(
                    self.run_id,
                    self.worker_id,
                    self.host,
                    self.resources.cpu_capacity,
                    free_cpu,
                    total_memory,
                    usable,
                    free_memory,
                    len(self.active),
                )
                claimed_any = False
                while free_cpu > 0 and free_memory > 0:
                    doc = self.store.claim_next_job(
                        self.run_id,
                        self.worker_id,
                        self.host,
                        free_cpu,
                        free_memory,
                        self.lease_seconds,
                    )
                    if doc is None:
                        break
                    claimed_any = True
                    job = doc["job_spec"]
                    self.resources.reserve(job)
                    thread = threading.Thread(target=self._execute_claimed_job, args=(doc,), daemon=False)
                    thread.start()
                    with self.active_lock:
                        self.active.append(thread)
                    free_cpu, total_memory, usable, free_memory = self.resources.snapshot()
                run_doc = self.store.get_run(self.run_id)
                if run_doc is None or run_doc.get("state") in TERMINAL_STATES | {"completed", "failed", "stopping"}:
                    if not self.active:
                        return 0
                if not claimed_any:
                    time.sleep(self.poll_seconds)
        finally:
            self.close()

    def _reap_threads(self) -> None:
        with self.active_lock:
            self.active = [thread for thread in self.active if thread.is_alive()]

    def _execute_claimed_job(self, doc: dict[str, Any]) -> None:
        job = doc["job_spec"]
        run_id = str(doc["run_id"])
        job_id = str(doc["job_id"])
        lease_owner = str(doc["lease_owner"])
        stop_renewal = threading.Event()
        renewer = threading.Thread(
            target=self._renew_lease_until_stopped,
            args=(job_id, lease_owner, stop_renewal),
            daemon=True,
        )
        renewer.start()
        try:
            argv = [str(paths.VENV_PYTHON), str(paths.RUN_SCRIPT)] + [str(part) for part in job["command_argv"]]
            env = os.environ.copy()
            env["BENCHMARK_JOB_SPEC"] = json.dumps(job, separators=(",", ":"))
            print(f"Executing {command_string(argv)}", flush=True)
            result = subprocess.run(argv, check=False, env=env)
            if result.returncode == 0:
                self.store.complete_job(run_id, job_id, lease_owner, "success")
                return
            failure = self._failure_from_exit_code(result.returncode)
            self.store.complete_job(run_id, job_id, lease_owner, "failed", failure=failure)
        except Exception as exc:
            failure = {"reason": "worker_exception", "exit_code": 1, "message": str(exc)}
            self.store.complete_job(run_id, job_id, lease_owner, "failed", failure=failure)
            write_failure(
                job,
                [str(part) for part in job.get("command_argv", [])],
                "worker_exception",
                exit_code=1,
                message=str(exc),
            )
        finally:
            stop_renewal.set()
            renewer.join(timeout=1)
            self.resources.release(job)

    def _renew_lease_until_stopped(self, job_id: str, lease_owner: str, stop_renewal: threading.Event) -> None:
        while not stop_renewal.wait(LEASE_RENEW_SECONDS):
            self.store.renew_lease(self.run_id, job_id, lease_owner, self.lease_seconds)

    @staticmethod
    def _failure_from_exit_code(exit_code: int) -> dict[str, Any]:
        if exit_code in {137, 143}:
            return {"reason": "signal", "exit_code": exit_code, "message": "Benchmark command was terminated by signal"}
        return {"reason": "exit_code", "exit_code": exit_code}


@dataclass(frozen=True)
class WorkerProcess:
    server: str
    worker_id: str
    process: subprocess.Popen


class WorkerLauncher:
    def __init__(self) -> None:
        self.processes: list[WorkerProcess] = []

    def start(self, run_id: str, servers: Iterable[str], workers_per_host: int) -> None:
        for server in servers:
            for index in range(workers_per_host):
                self.processes.append(self._start_one(run_id, server, index))

    def failed_process(self) -> WorkerProcess | None:
        for launched in self.processes:
            returncode = launched.process.poll()
            if returncode is not None and returncode != 0:
                return launched
        return None

    def terminate(self) -> None:
        for launched in self.processes:
            if launched.process.poll() is None:
                launched.process.terminate()
        deadline = time.monotonic() + 10
        for launched in self.processes:
            process = launched.process
            if process.poll() is not None:
                continue
            remaining = max(0.1, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.kill()

    def _start_one(self, run_id: str, server: str, index: int) -> WorkerProcess:
        worker_id = f"{server}-{index}-{uuid4().hex[:8]}"
        command = [
            str(paths.VENV_PYTHON),
            str(paths.WORKER_SCRIPT),
            "--run-id",
            run_id,
            "--worker-id",
            worker_id,
        ]
        if server in {"", "local", "localhost", "127.0.0.1"}:
            return WorkerProcess(server, worker_id, subprocess.Popen(command, cwd=str(paths.ROOT)))
        remote_command = self._remote_command(command)
        print(f"Starting worker on {server}: {remote_command}", flush=True)
        return WorkerProcess(server, worker_id, subprocess.Popen(["ssh", server, remote_command]))

    def _remote_command(self, command: list[str]) -> str:
        return f"cd {shlex.quote(str(paths.ROOT))} && exec {shlex.join(command)}"


def parse_servers(value: str) -> list[str]:
    servers: list[str] = []
    for chunk in value.split(","):
        servers.extend(part for part in chunk.split() if part)
    return servers


def run_coordinator(
    *,
    run_id: str,
    store: SchedulerStore,
    launcher: WorkerLauncher | None,
    poll_seconds: float = COORDINATOR_POLL_SECONDS,
) -> int:
    last_state = 0.0
    try:
        while True:
            for expired in store.expire_leases(run_id):
                job = expired["job_spec"]
                failure = expired.get("failure", {})
                write_failure(
                    job,
                    [str(part) for part in job.get("command_argv", [])],
                    failure.get("reason", "lease_expired"),
                    exit_code=failure.get("exit_code"),
                    message=failure.get("message"),
                )
            store.apply_skips(run_id)
            run_doc = store.get_run(run_id)
            if launcher is not None and run_doc is not None and run_doc.get("state") == "running":
                failed_worker = launcher.failed_process()
                if failed_worker is not None:
                    returncode = failed_worker.process.returncode
                    message = (
                        f"Worker {failed_worker.worker_id} on {failed_worker.server} "
                        f"exited with code {returncode}"
                    )
                    print(message, file=sys.stderr, flush=True)
                    store.set_run_state(
                        run_id,
                        "failed",
                        failure={
                            "reason": "worker_exit",
                            "exit_code": returncode,
                            "worker_id": failed_worker.worker_id,
                            "host": failed_worker.server,
                            "message": message,
                        },
                    )
                    return 1
            expected_workers = int((run_doc or {}).get("metadata", {}).get("expected_workers", 0) or 0)
            if expected_workers > 0:
                store.mark_memory_unfit_jobs(run_id, min_active_workers=expected_workers)
                store.apply_skips(run_id)
            store.refresh_ready_jobs(run_id)
            store.complete_run_if_done(run_id)
            run_doc = store.get_run(run_id)
            counts = store.state_counts(run_id)
            now = time.monotonic()
            if now - last_state >= 30:
                readiness = store.readiness_counts(run_id)
                print(f"Run {run_id} state: {counts}; readiness: {readiness}", flush=True)
                last_state = now
            if run_doc is None or run_doc.get("state") in {"completed", "failed", "stopping"}:
                if run_doc is not None and run_doc.get("state") == "failed":
                    return 1
                failed, skipped = store.terminal_counts(run_id)
                return 1 if failed or skipped else 0
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        store.set_run_state(run_id, "stopping")
        raise
    finally:
        if launcher is not None:
            launcher.terminate()


def print_status(run_id: str) -> int:
    store = SchedulerStore()
    try:
        run_doc = store.get_run(run_id)
        if run_doc is None:
            print(f"Run {run_id} not found", file=sys.stderr)
            return 1
        print(f"Run: {run_id}")
        print(f"state: {run_doc.get('state')}")
        print(f"Jobs: {store.state_counts(run_id)}")
        print(f"Readiness: {store.readiness_counts(run_id)}")
        workers = store.active_worker_capacities(run_id)
        if workers:
            print("Active workers:")
            for worker in sorted(workers, key=lambda item: str(item.get("worker_id"))):
                print(
                    "  "
                    f"{worker.get('worker_id')} host={worker.get('host')} "
                    f"cpu_free={worker.get('cpu_free')}/{worker.get('cpu_capacity')} "
                    f"memory_free={worker.get('memory_free')}"
                )
        return 0
    finally:
        store.close()
