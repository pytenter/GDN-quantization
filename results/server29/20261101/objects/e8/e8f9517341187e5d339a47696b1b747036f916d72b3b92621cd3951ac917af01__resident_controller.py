#!/usr/bin/env python3
"""One-shot, gold-blind resident qualification -> conditional formal controller.

Only standard-library imports. No GPU work is performed in this process, and no
failed/started unit is ever reset, resumed, retried, or dispatched a second time.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
import traceback

WATCHDOG = 1800.0
RELEASE_OBSERVATION = 120.0
POLL_SECONDS = 2.0
HEALTH_SECONDS = 30.0
PROGRESS_KEYS = ("phase", "condition", "stage", "token", "tokens", "step",
                 "input_index", "input_forwards", "generated_tokens",
                 "generated_token_count", "count")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


EXECUTION_DIRECTORIES = ("qualification", "formal", "logs", "logs/qualification", "logs/formal",
                         "operations", "dispatch", "dispatch/qualification", "monitoring")


def prepare_execution_directories(root):
    """Prepare empty execution parents, never a unit target or prior receipt.

    Check every existing component before mkdir so files, ordinary/dangling
    symlinks, FIFOs and other non-directory paths fail without being replaced.
    The separate unit target guard must still reject every existing leaf.
    """
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "Execution root must be an existing ordinary directory")
    root = root.resolve()

    def inspect(name):
        current = root
        for part in Path(name).parts:
            current = current / part
            if os.path.lexists(current):
                mode = current.lstat().st_mode
                require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode),
                        "Execution parent is not an ordinary directory: " + str(current.relative_to(root)))
        require(current.resolve().is_relative_to(root), "Execution parent escaped its root")
        return current

    # Validate the whole set first; a known bad path never causes partial mkdir.
    for name in EXECUTION_DIRECTORIES:
        inspect(name)
    for name in EXECUTION_DIRECTORIES:
        path = inspect(name)
        path.mkdir(parents=True, exist_ok=True)
        inspect(name)
    return dict(status="PASS_ORDINARY_EXECUTION_PARENT_DIRECTORIES", directories=list(EXECUTION_DIRECTORIES),
                unit_targets_created=False)


class Controller:
    def __init__(self, root, control):
        self.root, self.C = root, control
        self.identity = control.process_identity(os.getpid())
        require(isinstance(self.identity, dict), "Controller exact process identity is unresolved")
        self.active, self.completed = {}, {"qualification": {}, "formal": {}}
        self.intervals, self.attempted, self.stop_request = [], set(), None
        self.unresolved_launches = []
        self.reference_pair, self.pair_handle = None, None
        self.last_health = 0.0
        self.first_failure = None
        self.ledger = root / "GPU_RESOURCE_LEDGER.jsonl"
        self.phase = "INITIAL_ADMISSION"

    def event(self, kind, **fields):
        self.C.event(self.ledger, kind, controller_identity=self.identity, **fields)

    def target(self, phase, unit):
        return self.root / ("qualification" if phase == "qualification" else "formal") / unit["id"]

    def fail(self, error, **scope):
        if self.first_failure is not None:
            return
        self.first_failure = dict(status="FAIL_CLOSED_STOP", error=str(error),
            phase=self.phase, controller_identity=self.identity, monotonic=time.monotonic(),
            qualification_completed=len(self.completed["qualification"]),
            formal_completed=len(self.completed["formal"]), no_retry=True, **scope)
        path = self.root / "FIRST_FAILURE.json"
        if path.exists():
            self.first_failure = self.C.read(path)
        else:
            self.C.atomic_new(path, self.first_failure)
        self.event("FIRST_FAILURE", failure=self.first_failure)

    def own_nvml_present(self, identity, uuid):
        observation = self.C.query_gpu()
        matching = any(int(app["pid"]) == int(identity["pid"]) and app["uuid"] == uuid
                       for app in observation["apps"])
        if not matching:
            return False
        current = self.C.process_identity(identity["pid"])
        # A positively identified different starttick/argv is a foreign PID
        # reuse, never an own worker to signal or retain. A stale NVML row with
        # no readable current process remains unresolved until it disappears.
        return current is None or current == identity

    def selected_gpu_empty(self, gpu):
        observation = self.C.query_gpu()
        uuid = self.C.UUIDS[gpu]
        require(not any(app["uuid"] == uuid for app in observation["apps"]),
                "Selected UUID has an owner; never preempt a foreign job: " + uuid)
        return observation

    def reserve_reference_pair(self, unit):
        if self.reference_pair is not None:
            return self.reference_pair["gpu"] == unit["gpu"] and self.reference_pair["reference_unit"] == unit["reference_unit"]
        require(unit.get("repeat") == 0, "R1 cannot be dispatched without this controller's R0 reservation")
        handle = (self.root / "operations/REFERENCE_PAIR_SCHEDULER.lock").open("a")
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ram = Path("/dev/shm") / self.root.name
        ram.mkdir(parents=True, exist_ok=True)
        require(not (ram / "PAIR_OWNER.json").exists() and not (ram / unit["reference_unit"]).exists(),
                "Stale/unknown engineering reference exists; retain, do not overwrite or resume")
        stat = os.statvfs(ram)
        available = stat.f_bavail * stat.f_frsize
        require(available >= 100 * 1024 ** 3, "Reference pair requires80GiB budget plus20GiB free margin before worker/model creation")
        self.reference_pair = dict(gpu=unit["gpu"], reference_unit=unit["reference_unit"],
            source_freeze_sha256=self.C.sha(self.root / "SOURCE_FREEZE.json"),
            free_bytes_before_worker=available, reservation_monotonic=time.monotonic(),
            R0_validated=False, RAM_root=str(ram))
        self.pair_handle = handle
        self.event("REFERENCE_PAIR_RESERVED_BEFORE_WORKER", reservation=self.reference_pair)
        return True

    def finish_reference_unit(self, unit):
        if not unit.get("reference_unit"):
            return
        require(self.reference_pair is not None and self.reference_pair["reference_unit"] == unit["reference_unit"]
                and self.reference_pair["gpu"] == unit["gpu"], "Reference reservation identity drift")
        if unit["repeat"] == 0:
            require(not self.reference_pair["R0_validated"], "Duplicate R0 completion")
            self.reference_pair["R0_validated"] = True
            self.event("REFERENCE_R0_PASS_RELEASED_R1_STILL_RESERVED", reference_unit=unit["reference_unit"])
            return
        require(unit["repeat"] == 1 and self.reference_pair["R0_validated"], "R1 lacks a legal completed R0")
        receipt = self.C.read(self.target("qualification", unit) / "REFERENCE_RELEASED.json")
        require(receipt["status"] == "PASS" and receipt["paired_actual_bytes_PASS"] is True,
                "R1 must durably PASS and release only its exact engineering reference")
        ram = Path(self.reference_pair["RAM_root"])
        require(not (ram / "PAIR_OWNER.json").exists() and not (ram / unit["reference_unit"]).exists(),
                "R1 reference/owner still present after claimed cleanup")
        self.event("REFERENCE_PAIR_COMPLETE_PASS_RELEASED", reservation=self.reference_pair,
                   release_receipt_sha256=self.C.sha(self.target("qualification", unit) / "REFERENCE_RELEASED.json"))
        self.reference_pair = None
        fcntl.flock(self.pair_handle, fcntl.LOCK_UN)
        self.pair_handle.close()
        self.pair_handle = None

    def dispatch(self, phase, unit):
        key = (phase, unit["id"])
        require(key not in self.attempted, "Already attempted; no redispatch")
        require(self.first_failure is None, "Global failure prevents all future dispatch")
        gpu, uuid = unit["gpu"], self.C.UUIDS[unit["gpu"]]
        require(unit["gpu_uuid"] == uuid and gpu not in self.active, "GPU/UUID queue identity differs")
        prepare_execution_directories(self.root)
        self.C.verify_sources(self.root)
        self.selected_gpu_empty(gpu)
        target = self.target(phase, unit)
        if phase == "qualification":
            require(not os.path.lexists(target), "Qualification target already exists; never repeat any attempted unit")
            dispatch_path = self.root / "dispatch/qualification" / (unit["id"] + ".json")
            permit_path = self.root / "GPU_PERMIT.json"
            plan = self.root / "QUALIFICATION_JOB_MANIFEST.json"
            script = self.root / "scripts/qualify_gpu.py"
        else:
            require(not os.path.lexists(target), "Formal sample target already exists; never restart a sample")
            target.mkdir(parents=True, exist_ok=False)
            dispatch_path = target / "DISPATCH.json"
            permit_path = self.root / "FORMAL_GPU_PERMIT.json"
            plan = self.root / "FORMAL_JOB_MANIFEST.json"
            script = self.root / "scripts/resident_formal_worker.py"
        now = time.monotonic()
        dispatch = dict(task_id=self.C.TASK, execution_attempt=self.root.name, phase=phase,
            unit=unit["id"], unit_record=unit, gpu=gpu, gpu_uuid=uuid,
            controller_identity=self.identity, source_freeze_sha256=self.C.sha(self.root / "SOURCE_FREEZE.json"),
            permit_sha256=self.C.sha(permit_path), base_gpu_permit_sha256=self.C.sha(self.root / "GPU_PERMIT.json"),
            plan_sha256=self.C.sha(plan), dispatch_monotonic=now)
        if phase == "formal":
            dispatch["formal_authorization_sha256"] = self.C.sha(self.root / "AUTHORIZATION_FORMAL.json")
        self.C.atomic_new(dispatch_path, dispatch)
        self.attempted.add(key)
        logs = self.root / "logs" / phase
        logs.mkdir(parents=True, exist_ok=True)
        log = (logs / (unit["id"] + ".log")).open("xb")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, PYTHONUNBUFFERED="1",
                   TRANSFORMERS_OFFLINE="1", HF_HUB_OFFLINE="1")
        command = [str(self.C.PYTHON), "-u", "-B", str(script), "--root", str(self.root),
                   "--unit", unit["id"], "--gpu", str(gpu)]
        process = None
        try:
            process = subprocess.Popen(command, cwd=self.root, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            identity = self.C.process_identity(process.pid)
            require(isinstance(identity, dict), "Child PID/starttick/argv identity is unresolved after launch")
        except BaseException:
            log.close()
            if process is not None:
                unresolved = dict(unit=unit["id"], phase=phase, pid=process.pid, command=command,
                    exit_returncode=process.poll(), identity="UNRESOLVED_NO_PID_ONLY_SIGNAL_ALLOWED",
                    monotonic=time.monotonic())
                self.unresolved_launches.append(unresolved)
                self.event("WORKER_LAUNCH_IDENTITY_UNRESOLVED", launch=unresolved)
            raise
        record = dict(phase=phase, unit=unit, gpu=gpu, uuid=uuid, target=target,
            process=process, identity=identity, log=log, dispatch=dispatch,
            identity_observed_monotonic=time.monotonic(), last_progress_monotonic=time.monotonic(),
            progress_signature=None, latest_progress=None, exit_observed=None, exit_returncode=None)
        self.active[gpu] = record
        self.C.atomic_new(self.root / "operations" / (phase + "_" + unit["id"] + "_WORKER_IDENTITY.json"),
                          dict(identity=identity, command=command, dispatch=dispatch))
        self.event("DISPATCH", phase=phase, unit=unit["id"], gpu=gpu, gpu_uuid=uuid,
            identity=identity, dispatch_monotonic_lower=now,
            identity_monotonic_upper=record["identity_observed_monotonic"],
            source_freeze_sha256=dispatch["source_freeze_sha256"], permit_sha256=dispatch["permit_sha256"])

    def complete_released(self, record, release, *, validate=True):
        unit, target = record["unit"], record["target"]
        target.mkdir(parents=True, exist_ok=True)
        observed = time.monotonic()
        interval = dict(unit=unit["id"], phase=record["phase"], gpu=record["gpu"], gpu_uuid=record["uuid"],
            identity=record["identity"], controller_identity=self.identity,
            dispatch_monotonic_lower=record["dispatch"]["dispatch_monotonic"],
            identity_monotonic_upper=record["identity_observed_monotonic"],
            exit_first_observed_monotonic=record["exit_observed"], exit_returncode=record["exit_returncode"],
            release_first_observed_monotonic=observed, release=release,
            EXACT_physical_GPUh="UNRESOLVED", source_freeze_sha256=record["dispatch"]["source_freeze_sha256"],
            permit_sha256=record["dispatch"]["permit_sha256"], monotonic_same_host=True)
        self.C.atomic_new(target / "RESOURCE_INTERVAL.json", interval)
        self.intervals.append(interval)
        record["log"].close()
        del self.active[record["gpu"]]
        self.event("RELEASE_OBSERVED", interval=interval)
        require(release == "PASS", "Exact own process/NVML release remains unresolved after120s")
        require(record["exit_returncode"] == 0, "Worker exited nonzero: " + unit["id"])
        require(not (target / "FIRST_FAILURE.json").exists(), "Worker persisted FIRST_FAILURE: " + unit["id"])
        if validate:
            function = (self.C.validate_qualification_result if record["phase"] == "qualification"
                        else self.C.validate_formal_result)
            function(self.root, unit)
            receipt = dict(status="PASS_VALIDATED_RESULT_EXIT_AND_OWN_RELEASE", unit=unit["id"],
                gpu=record["gpu"], gpu_uuid=record["uuid"], result_sha256=self.C.sha(target / "RESULT.json"),
                resource_interval_sha256=self.C.sha(target / "RESOURCE_INTERVAL.json"),
                source_freeze_sha256=record["dispatch"]["source_freeze_sha256"], monotonic=time.monotonic())
            self.C.atomic_new(target / "CONTROLLER_VALIDATION.json", receipt)
            receipt["validation_sha256"] = self.C.sha(target / "CONTROLLER_VALIDATION.json")
            if record["phase"] == "qualification":
                self.finish_reference_unit(unit)
            self.completed[record["phase"]][unit["id"]] = receipt

    def poll(self):
        now = time.monotonic()
        for record in list(self.active.values()):
            code = record["process"].poll()
            if code is None and not self.C.same_process(record["identity"]):
                # Normal exit can occur between waitpid(WNOHANG) and /proc
                # observation. Reap the same Popen once more before calling a
                # still-live identity change; never invent an exited identity.
                code = record["process"].poll()
                require(code is not None, "Live worker PID/starttick/argv identity changed")
            if code is not None:
                if record["exit_observed"] is None:
                    record["exit_observed"], record["exit_returncode"] = now, code
                    self.event("EXIT_FIRST_OBSERVED", phase=record["phase"], unit=record["unit"]["id"],
                        identity=record["identity"], exit_returncode=code, monotonic=now)
                released = (not self.C.same_process(record["identity"]) and
                            not self.own_nvml_present(record["identity"], record["uuid"]))
                if released:
                    self.complete_released(record, "PASS")
                elif now - record["exit_observed"] >= RELEASE_OBSERVATION:
                    self.complete_released(record, "UNRESOLVED_AFTER_120S")
                continue
            failure = record["target"] / "FIRST_FAILURE.json"
            require(not failure.exists(), "Worker persisted FIRST_FAILURE before exit: " + record["unit"]["id"])
            path = record["target"] / "PROGRESS.json"
            if path.exists():
                progress = self.C.read(path)
                signature = json.dumps({key: progress.get(key) for key in PROGRESS_KEYS}, sort_keys=True)
                if signature != record["progress_signature"]:
                    record["progress_signature"] = signature
                    record["last_progress_monotonic"] = now
                record["latest_progress"] = {key: progress[key] for key in PROGRESS_KEYS if key in progress}
            require(now - record["last_progress_monotonic"] <= WATCHDOG,
                    "1800s without actual phase/token progress: " + record["unit"]["id"])

    def health(self):
        now = time.monotonic()
        if now - self.last_health < HEALTH_SECONDS:
            return
        self.last_health = now
        self.C.status(self.root, self.phase,
            qualification_completed=len(self.completed["qualification"]), formal_completed=len(self.completed["formal"]),
            reference_pair=self.reference_pair, active=[dict(unit=r["unit"]["id"], gpu=r["gpu"],
                gpu_uuid=r["uuid"], identity=r["identity"], progress=r["latest_progress"])
                for r in self.active.values()], monotonic=now, gold_blind=True)

    @staticmethod
    def is_performance(unit):
        return unit.get("kind", "").startswith(("perf_", "profile_"))

    def drive(self, phase, queues):
        self.phase = "QUALIFICATION_RUNNING" if phase == "qualification" else "FORMAL_RUNNING"
        while any(queues.values()) or self.active:
            if self.stop_request:
                raise RuntimeError("External controller signal requested STOP: " + self.stop_request)
            self.poll()
            for gpu in sorted(queues):
                if gpu in self.active or not queues[gpu]:
                    continue
                unit = queues[gpu][0]
                if phase == "qualification":
                    if self.is_performance(unit):
                        nonperformance_pending = any(not self.is_performance(x) for q in queues.values() for x in q)
                        if nonperformance_pending or self.active or self.reference_pair is not None:
                            continue
                    if unit.get("reference_unit"):
                        if self.reference_pair is not None and (self.reference_pair["gpu"] != gpu or
                                self.reference_pair["reference_unit"] != unit["reference_unit"]):
                            continue
                        self.reserve_reference_pair(unit)
                        if unit["repeat"] == 1:
                            require(self.reference_pair["R0_validated"], "Cannot create R1 before R0 PASS+exit+release")
                    elif self.reference_pair is not None and self.reference_pair["gpu"] == gpu:
                        raise RuntimeError("Reserved R0/R1 pair must complete contiguously on its UUID")
                self.dispatch(phase, unit)
                queues[gpu].pop(0)
            self.health()
            time.sleep(POLL_SECONDS)
        require(self.reference_pair is None, "A reference pair remains incomplete")

    def stop_all(self):
        for record in list(self.active.values()):
            try:
                receipt = self.C.stop_own(record["identity"], self.root, "Resident global first failure")
                self.event("PRECISE_OWN_STOP", unit=record["unit"]["id"], receipt=receipt)
            except BaseException as error:
                self.event("PRECISE_OWN_STOP_ERROR", unit=record["unit"]["id"], error=repr(error))
        deadline = time.monotonic() + RELEASE_OBSERVATION
        while self.active and time.monotonic() < deadline:
            for record in list(self.active.values()):
                code = record["process"].poll()
                if code is None:
                    continue
                if record["exit_observed"] is None:
                    record["exit_observed"], record["exit_returncode"] = time.monotonic(), code
                    self.event("EXIT_FIRST_OBSERVED", phase=record["phase"], unit=record["unit"]["id"],
                               identity=record["identity"], exit_returncode=code)
                try:
                    if not self.C.same_process(record["identity"]) and not self.own_nvml_present(record["identity"], record["uuid"]):
                        self.complete_released(record, "PASS", validate=(code == 0))
                except BaseException as error:
                    self.event("STOPPED_WORKER_VALIDATION_NOTE", unit=record["unit"]["id"], error=repr(error))
            if self.active:
                time.sleep(POLL_SECONDS)
        for record in list(self.active.values()):
            if record["exit_observed"] is None:
                record["exit_observed"] = time.monotonic()
                record["exit_returncode"] = record["process"].poll()
            try:
                self.complete_released(record, "UNRESOLVED_AFTER_120S", validate=False)
            except BaseException as error:
                self.event("OWN_RELEASE_UNRESOLVED", unit=record["unit"]["id"], error=repr(error))
        # Failed RAM references and PAIR_OWNER remain untouched for preservation.
        if self.pair_handle is not None:
            fcntl.flock(self.pair_handle, fcntl.LOCK_UN)
            self.pair_handle.close()
            self.pair_handle = None

    def run(self):
        prepare_execution_directories(self.root)
        started = dict(task_id=self.C.TASK, execution_attempt=self.root.name,
            identity=self.identity, monotonic=time.monotonic(), no_gpu_time_cap=True,
            watchdog_only_without_actual_progress_seconds=WATCHDOG, cleanup_observation_seconds=RELEASE_OBSERVATION)
        self.C.atomic_new(self.root / "CONTROLLER_STARTED.json", started)
        try:
            self.C.verify_sources(self.root)
            self.C.verify_model_and_external(self.root)
            self.C.validate_prior_p1_stop(self.root)
            qual = self.C.read(self.root / "QUALIFICATION_JOB_MANIFEST.json")["units"]
            formal = self.C.read(self.root / "FORMAL_JOB_MANIFEST.json")["jobs"]
            require(len(qual) == 58 and len({x["id"] for x in qual}) == 58, "Qualification roster must be exactly58 unique units")
            require(len(formal) == 60 and len({x["id"] for x in formal}) == 60, "Formal roster must be exactly60 unique samples")
            codec = [x for x in qual if x["id"] == "GPU1_CODEC"]
            fp = [x for x in qual if x["id"] == "GPU1_FP128"]
            require(len(codec) == len(fp) == 1 and codec[0]["gpu"] == fp[0]["gpu"] == 1, "Initial GPU1 dependency units differ")
            self.drive("qualification", {1: codec.copy()})
            self.drive("qualification", {1: fp.copy()})
            remaining = [x for x in qual if x["id"] not in ("GPU1_CODEC", "GPU1_FP128")]
            self.drive("qualification", {gpu: [x for x in remaining if x["gpu"] == gpu] for gpu in self.C.UUIDS})
            require(len(self.completed["qualification"]) == 58 and not self.active,
                    "All58 actual RESULT+exit+release gates must PASS")
            self.C.status(self.root, "QUALIFICATION_ALL58_PASS_RELEASED", qualification_completed=58,
                          formal_completed=0, active=[], monotonic=time.monotonic())
            self.C.authorize_formal(self.root, list(self.completed["qualification"].values()))
            require((self.root / "AUTHORIZATION_FORMAL.json").is_file() and
                    (self.root / "FORMAL_GPU_PERMIT.json").is_file(), "Unique formal authorization/permit missing")
            self.drive("formal", {gpu: [x for x in formal if x["gpu"] == gpu] for gpu in self.C.UUIDS})
            require(len(self.completed["formal"]) == 60 and not self.active,
                    "Formal COMPLETE requires60 legal independently validated samples and release")
            status = "RESIDENT_ROTATION_QUALIFIED_FORMAL_COMPLETE_60_60_AND_STOP"
        except BaseException as error:
            self.fail(repr(error), traceback=traceback.format_exc())
            self.stop_all()
            status = "FAIL_CLOSED_STOP"
        own_released = (not self.active and not self.unresolved_launches and
                        all(x["release"] == "PASS" for x in self.intervals))
        formal_receipts = list(self.completed["formal"].values())
        generations = dict(status=("COMPLETE_60_60" if len(formal_receipts) == 60 and
                own_released and status == "RESIDENT_ROTATION_QUALIFIED_FORMAL_COMPLETE_60_60_AND_STOP"
                else "PARTIAL_FAIL_CLOSED_STOP"),
            task_id=self.C.TASK, execution_attempt=self.root.name, count=len(formal_receipts),
            planned=60, formal_receipts=formal_receipts, all_own_processes_released=own_released,
            source_freeze_sha256=self.C.sha(self.root / "SOURCE_FREEZE.json"),
            monotonic=time.monotonic(), gold_blind=True, no_accuracy_claim=True)
        self.C.atomic_new(self.root / "FORMAL_GENERATIONS_COMPLETE.json", generations)
        session = dict(status=status, task_id=self.C.TASK, execution_attempt=self.root.name,
            qualification_completed=len(self.completed["qualification"]), formal_completed=len(self.completed["formal"]),
            qualification_receipts=list(self.completed["qualification"].values()),
            formal_receipts=formal_receipts,
            attempted_units=len(self.attempted), resource_intervals=len(self.intervals),
            own_release_all_PASS=own_released, all_own_processes_released=own_released,
            unresolved_launches=self.unresolved_launches,
            first_failure=self.first_failure, reference_pair_retained=self.reference_pair,
            controller_identity=self.identity, monotonic=time.monotonic(),
            EXACT_physical_GPUh="UNRESOLVED", no_retry=True, gold_blind=True)
        self.C.atomic_new(self.root / "SESSION_RESULT.json", session)
        self.C.status(self.root, status, qualification_completed=session["qualification_completed"],
                      formal_completed=session["formal_completed"], own_release_all_PASS=session["own_release_all_PASS"], active=[])
        return 0 if status == "RESIDENT_ROTATION_QUALIFIED_FORMAL_COMPLETE_60_60_AND_STOP" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    require(args.execute, "Explicit --execute is required; this entry point never auto-starts")
    root = args.root.absolute()
    require(not os.path.lexists(root / "CONTROLLER_STARTED.json"), "Controller already attempted; no restart/reset")
    prepare_execution_directories(root)
    root = root.resolve()
    sys.path.insert(0, str(root / "src"))
    import resident_execution_control as control
    handle = (root / "operations/CONTROLLER.lock").open("a")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not (root / "CONTROLLER_STARTED.json").exists(), "Controller already attempted; no restart/reset")
    controller = Controller(root, control)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda number, _frame: setattr(controller, "stop_request", signal.Signals(number).name))
    return controller.run()


if __name__ == "__main__":
    raise SystemExit(main())
