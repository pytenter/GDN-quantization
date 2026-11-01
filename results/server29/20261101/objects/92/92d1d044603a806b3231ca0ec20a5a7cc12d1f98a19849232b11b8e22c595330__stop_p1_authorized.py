#!/usr/bin/env python3
"""Explicit human-authorized exact P1 cancellation; default is read-only dry-run.

Only --execute with the exact authorization hash may signal anything. All
signals prefer Linux pidfds bound to freshly checked PID/starttick/cmdline. On
the actual Linux3.10 host without pidfds, an explicit audited exact-identity
check immediately precedes each individual os.kill; no process-group signal.
The
frozen P1 sources, existing receipts, trajectories and RAM references are never
rewritten or removed. This tool does not start a resident or a P1 GPU process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


ATTEMPT = "QWEN_GDN_OSCAR_RESIDENT_ROTATION_EFFICIENT_QUALIFICATION_V1_GPU_CONTINUATION_20261022"
REMOTE_ROOT = Path("/data/zypan/analyses/qwen_gdn_oscar_resident_rotation_efficient_qualification_v1_gpu_continuation_20261022")
P1 = Path("/data/zypan/analyses/qwen_gdn_oscar_sym32_p1_c1_s3_unified_recovery_aime26_v2_rerun_20261022")
PYTHON = "/data/ydai/miniconda3/envs/bitdecode/bin/python"
UUIDS = {1: "GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3",
         2: "GPU-b7ed673d-18ba-39ca-d7bc-c846af4f6b31",
         3: "GPU-30b07aba-a728-7649-4899-e09d33f8bc75"}
PINS = {
    "SOURCE_FREEZE.json": "3955f03f2eaa56f83245a85ccad1c56d7d98b70eaa046154a1d95f5b79a7fa14",
    "GPU_PERMIT.json": "bfd67ac400600efe3680ee8cfc6dafa5059442c52273f1940f3d2d6892b6b604",
    "FORMAL_JOB_MANIFEST.json": "a75e71a560456ad01d36f49d4c936be83b1947ad1699b155262a330646427362",
    "P1_EXECUTABLE_IDENTITY.json": "d19032d35cce14624b9ab0e2625f324c0f3c6d8dc55a482a8cf460771b6a531e",
    "scripts/run_offline.sh": "22bad2e0812c134776ed47ebf9d92e02e293087c6c5b328e17c872f18b0ddcc1",
    "scripts/gpu_controller.py": "90ad5f2f2318a2fdc7ad53dab282ecb1023a4ee65fbd6e902aa158e1d375a2e9",
    "scripts/finalize_offline.py": "13fe343dc9d6b6b31b12c82d19dd953f36318573f15c9118cd73c236bf77e379",
}
RAW_SUFFIXES = (".bin", ".npz", ".npy", ".f32", ".pt", ".pth", ".safetensors")
NO_ACTION_SOURCE = "No process signalling, GPU launch, restart, Tensor deletion or original receipt modification"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical(value))
        handle.flush()
        os.fsync(handle.fileno())


def validate_authorization(path, expected_hash, execute):
    path = Path(path)
    actual = sha(path)
    if execute:
        require(expected_hash is not None and expected_hash == actual,
                "EXECUTE requires the exact independently reviewed authorization SHA256")
    elif expected_hash is not None:
        require(expected_hash == actual, "Authorization SHA256 differs")
    document = load(path)
    require(document["execution_attempt"] == ATTEMPT and document["P1_stop_authorized"] is True
            and document["P1_restart_authorized"] is False and document["GPU0_authorized"] is False,
            "Human authorization scope does not allow this P1-only stop")
    require(document["prior_science_source_sha256"] ==
            "faa812e121ffd2987ded1d3b88460e4ece274939c7161c6e34848fdff83b0c08",
            "Prior resident scientific identity differs; do not rewrite old blocked terminal")
    return actual, document


def proc_info(pid):
    try:
        path = Path("/proc") / str(pid)
        raw = (path / "stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        args = (path / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
        # A zombie loses argv, so identity comparison uses the recorded starttick
        # to identify the already-exited child; no signal is sent to a zombie.
        return dict(identity=dict(pid=int(pid), starttick=int(fields[19]), cmdline=args),
                    state=fields[0], ppid=int(fields[1]),
                    kernel_exit_wait_status=int(fields[49]) if len(fields) > 49 else None)
    except (FileNotFoundError, ProcessLookupError):
        return None


def exact_live(expected):
    actual = proc_info(expected["pid"])
    return actual is not None and actual["identity"] == expected and actual["state"] not in ("Z", "X", "x")


def exited_observation(expected):
    actual = proc_info(expected["pid"])
    if actual is None:
        return dict(exact_identity_exited=True, observation="PID_ABSENT", expected=expected,
                    exit_returncode="UNRESOLVED_NOT_PARENT_WAIT")
    if actual["identity"]["starttick"] != expected["starttick"]:
        return dict(exact_identity_exited=True, observation="PID_REUSED_DIFFERENT_STARTTICK", expected=expected,
                    current=actual, exit_returncode="UNRESOLVED_NOT_PARENT_WAIT")
    if actual["state"] in ("Z", "X", "x"):
        return dict(exact_identity_exited=True, observation="EXACT_STARTTICK_ZOMBIE_OR_EXITED", expected=expected,
                    current=actual, kernel_exit_wait_status=actual["kernel_exit_wait_status"],
                    exit_returncode="UNRESOLVED_NOT_PARENT_WAIT")
    require(actual["identity"] == expected, "Same-starttick process changed argv unexpectedly; no signal or false exit claim")
    return dict(exact_identity_exited=False, observation="EXACT_PROCESS_LIVE", expected=expected, current=actual)


def query_nvml():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    require(result.returncode == 0, "Read-only NVML process query failed:" + result.stderr)
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split(",")]
        require(len(parts) == 3, "Unexpected NVML process CSV")
        rows.append(dict(uuid=parts[0], pid=int(parts[1]), memory_MiB=parts[2],
                         actual_process=proc_info(int(parts[1]))))
    return rows


def verify_frozen_p1():
    for relative, expected in PINS.items():
        require(sha(P1 / relative) == expected, "Frozen P1 identity differs:" + relative)
    freeze = load(P1 / "SOURCE_FREEZE.json")
    require(len(freeze["files"]) == 622, "Expected frozen622 P1 inputs/sources")
    for row in freeze["files"]:
        path = (P1 / row["path"]).resolve()
        require(path.is_relative_to(P1) and not (P1 / row["path"]).is_symlink(), "Frozen P1 path escapes root")
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "Frozen P1 file changed:" + row["path"])
    launch = load(P1 / "operations/LAUNCH_RECEIPT.json")
    require(launch["command"] == "bash " + str(P1 / "scripts/run_offline.sh")
            and launch["source_freeze_sha256"] == PINS["SOURCE_FREEZE.json"]
            and launch["permit_sha256"] == PINS["GPU_PERMIT.json"], "Original once-only shell launch identity differs")
    coordinator = load(P1 / "GPU_RUN_STARTED.json")["identity"]
    expected = [PYTHON, "-u", "-B", str(P1 / "scripts/gpu_controller.py"), "--root", str(P1), "--execute"]
    require(coordinator["cmdline"] == expected, "Original coordinator command differs")
    info = proc_info(coordinator["pid"])
    require(info is not None and info["identity"] == coordinator and info["state"] not in ("Z", "X", "x"),
            "Coordinator is no longer exact/live; inspect actual completion instead of stopping a historical PID")
    parent = proc_info(info["ppid"])
    require(parent is not None and parent["state"] not in ("Z", "X", "x"), "Coordinator launcher shell missing")
    args = parent["identity"]["cmdline"]
    require(len(args) == 2 and Path(args[0]).name == "bash" and args[1] == str(P1 / "scripts/run_offline.sh"),
            "Direct coordinator parent is not the exact frozen run_offline bash")
    require(not (P1 / "FINALIZER_STARTED.json").exists(),
            "Original finalizer already attempted; inspect it before any cancellation signals")
    return coordinator, parent["identity"], launch


def jobs_by_id():
    jobs = load(P1 / "FORMAL_JOB_MANIFEST.json")["jobs"]
    return {row["id"]: row for row in jobs}


def validate_worker(info, coordinator, jobs, allow_reparented=False):
    identity = info["identity"]
    args = identity["cmdline"]
    expected_prefix = [PYTHON, "-u", "-B", str(P1 / "scripts/gpu_unit_worker.py"), "--root", str(P1), "--unit"]
    require(len(args) == 9 and args[:7] == expected_prefix and args[8] == "--formal",
            "Worker does not match the exact current P1 formal command")
    unit_id = args[7]
    require(unit_id in jobs, "Worker job not in original frozen roster")
    job = jobs[unit_id]
    require(job["condition"] == "P1" and job["gpu"] in UUIDS and job["gpu_uuid"] == UUIDS[job["gpu"]],
            "Only original P1 jobs on authorizedUUID1/2/3 may be stopped")
    target = P1 / "formal" / unit_id
    dispatch = load(target / "DISPATCH.json")
    require(dispatch["controller_identity"] == coordinator and dispatch["unit"] == unit_id
            and dispatch["permit_sha256"] == PINS["GPU_PERMIT.json"]
            and dispatch["gpu"] == job["gpu"] and dispatch["gpu_uuid"] == job["gpu_uuid"]
            and dispatch["formal"] is True, "Original worker dispatch ownership differs")
    if (target / "STARTED.json").exists():
        require(load(target / "STARTED.json")["identity"] == identity,
                "Actual worker/starttick/argv differs from its STARTED receipt")
        started = True
    else:
        started = False  # A Popen/STARTED race is still owned via exact PPID+dispatch.
    require(info["ppid"] == coordinator["pid"] or (allow_reparented and started
            and exited_observation(coordinator)["exact_identity_exited"]),
            "Worker ownership needs exact coordinator parent or STARTED-confirmed post-stop reparenting")
    return dict(unit=unit_id, identity=identity, gpu=job["gpu"], gpu_uuid=job["gpu_uuid"],
                actual_parent=info["ppid"], started_receipt_present=started,
                dispatch_sha256=sha(target / "DISPATCH.json"), observed_state=info["state"])


def scan_workers(coordinator, jobs, allow_reparented=False):
    workers, unknown_children, unexpected_scope = [], [], []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        info = proc_info(int(path.name))
        if info is None or info["state"] in ("Z", "X", "x"):
            continue
        args = info["identity"]["cmdline"]
        if info["ppid"] == coordinator["pid"]:
            if str(P1 / "scripts/gpu_unit_worker.py") in args:
                workers.append(validate_worker(info, coordinator, jobs, allow_reparented))
            else:
                unknown_children.append(info)
        elif str(P1 / "scripts/gpu_unit_worker.py") in args and str(P1) in args:
            if allow_reparented:
                workers.append(validate_worker(info, coordinator, jobs, allow_reparented))
            else:
                unexpected_scope.append(info)
    require(not unexpected_scope, "An orphan or foreign-parent same-root worker exists; inspect before signalling")
    return workers, unknown_children


def check_nvml_ownership(workers, nvml):
    known = {row["identity"]["pid"]: row for row in workers}
    own_seen = []
    for app in nvml:
        info = app["actual_process"]
        if info and str(P1 / "scripts/gpu_unit_worker.py") in info["identity"]["cmdline"]:
            require(app["pid"] in known and info["identity"] == known[app["pid"]]["identity"],
                    "A same-root CUDA worker is not a verified dispatch; never guess its ownership")
            require(app["uuid"] == known[app["pid"]]["gpu_uuid"], "Own worker CUDA UUID differs from its frozen dispatch")
            own_seen.append(app)
        elif app["pid"] in known and info and info["identity"] == known[app["pid"]]["identity"]:
            require(app["uuid"] == known[app["pid"]]["gpu_uuid"], "Own process consumed an unauthorized UUID")
    return own_seen


def metadata_snapshot():
    rows = []
    for path in sorted(P1.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = str(path.relative_to(P1))
        try:
            before = path.stat()
        except FileNotFoundError:
            rows.append(dict(path=relative, observation="LIVE_TEMPORARY_PATH_VANISHED_BEFORE_STAT"))
            continue
        row = dict(path=relative, bytes=before.st_size, inode=before.st_ino, mtime_ns=before.st_mtime_ns)
        # Raw Tensor/reference bytes are untouched; static source raw SHA is
        # already checked by verify_frozen_p1. All dispatch/progress/log/ledger
        # metadata and token chunks receive actual byte SHA, including failures.
        if not relative.lower().endswith(RAW_SUFFIXES):
            try:
                row["sha256"] = sha(path)
                after = path.stat()
            except FileNotFoundError:
                row["observation"] = "LIVE_TEMPORARY_PATH_VANISHED_DURING_METADATA_READ"
                rows.append(row)
                continue
            row["stable_during_read"] = (before.st_size, before.st_ino, before.st_mtime_ns) == (after.st_size, after.st_ino, after.st_mtime_ns)
            row["after_bytes"] = after.st_size
            row["after_mtime_ns"] = after.st_mtime_ns
        else:
            row["raw_bytes_not_modified_or_deleted"] = True
        rows.append(row)
    ledgers = {name: sha(P1 / name) for name in ("GPU_RESOURCE_LEDGER.jsonl", "RESOURCE_LEDGER.json", "FORMAL_PROGRESS.json")
               if (P1 / name).is_file()}
    return dict(observed_monotonic=time.monotonic(), files=rows, ledger_sha256=ledgers,
                source_freeze_sha256=PINS["SOURCE_FREEZE.json"], read_only_original_root=True,
                live_metadata_may_change_before_schedulers_are_frozen=True)


class SignalControl:
    def __init__(self, events_path, execute):
        self.path, self.execute = Path(events_path), execute
        self.fds, self.sent = {}, []
        self.pidfd_supported = hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal")

    def event(self, event, **values):
        require(self.execute, "Dry-run must not write signal events")
        with self.path.open("ab") as handle:
            handle.write(canonical(dict(event=event, monotonic=time.monotonic(), **values)))
            handle.flush()
            os.fsync(handle.fileno())

    def bind(self, expected):
        require(self.execute, "Dry-run must not open signal pidfds")
        old = self.fds.pop(expected["pid"], None)
        if old is not None and old[1] is not None:
            os.close(old[1])
        observation = exited_observation(expected)
        if observation["exact_identity_exited"]:
            self.fds[expected["pid"]] = (expected, None)
            self.event("BIND_NOT_NEEDED_ALREADY_EXITED", identity=expected, observation=observation)
            return
        require(exact_live(expected), "Exact process changed before signal identity binding")
        if not self.pidfd_supported:
            self.fds[expected["pid"]] = (expected, None)
            self.event("PIDFD_UNAVAILABLE_KERNEL_3_10_EXACT_IDENTITY_FALLBACK", identity=expected,
                       actual_kernel=os.uname().release, atomic_pidfd_signal_claimed=False)
            return
        try:
            fd = os.pidfd_open(expected["pid"], 0)
        except OSError as error:
            if error.errno not in (38, 22):
                raise
            self.pidfd_supported = False
            self.fds[expected["pid"]] = (expected, None)
            self.event("PIDFD_KERNEL_UNAVAILABLE_EXACT_IDENTITY_FALLBACK", identity=expected,
                       actual_kernel=os.uname().release, error=repr(error), atomic_pidfd_signal_claimed=False)
            return
        if not exact_live(expected):
            os.close(fd)
            raise RuntimeError("Process changed during pidfd binding; no signal sent")
        self.fds[expected["pid"]] = (expected, fd)

    def send(self, expected, sig):
        require(self.execute, "DRY_RUN_NO_SIGNALS")
        observation = exited_observation(expected)
        if observation["exact_identity_exited"]:
            self.event("SIGNAL_NOT_SENT_ALREADY_EXITED", signal=int(sig), identity=expected, observation=observation)
            return
        require(expected["pid"] in self.fds and self.fds[expected["pid"]][0] == expected,
                "Signal requires an exact already-bound process identity")
        require(exact_live(expected), "Exact identity changed before signalling")
        self.event("SIGNAL_REQUEST", signal=int(sig), signal_name=signal.Signals(sig).name, identity=expected)
        fd = self.fds[expected["pid"]][1]
        mode = "EXACT_PIDFD" if fd is not None else "EXACT_IDENTITY_IMMEDIATE_OS_KILL_KERNEL_3_10"
        try:
            if fd is not None:
                signal.pidfd_send_signal(fd, sig, None, 0)
            else:
                # No syscalls, file/event writes or delayed work between this
                # final identity check and the individual PID signal syscall.
                require(exact_live(expected), "Fallback exact identity changed immediately before signal")
                os.kill(expected["pid"], sig)
        except ProcessLookupError:
            require(exited_observation(expected)["exact_identity_exited"], "pidfd process lookup failed without exit evidence")
            self.event("SIGNAL_NOT_SENT_EXIT_RACE", signal=int(sig), identity=expected)
            return
        record = dict(signal=int(sig), signal_name=signal.Signals(sig).name, identity=expected,
                      signal_mode=mode, atomic_pidfd_signal_claimed=fd is not None)
        self.sent.append(record)
        self.event("SIGNAL_SENT_TO_EXACT_AUTHORIZED_PROCESS", **record)

    def close(self):
        for _, fd in self.fds.values():
            if fd is not None:
                os.close(fd)
        self.fds.clear()

    def kill_scheduler_pair(self, launcher, coordinator):
        """No log/fsync/wait between the two exact scheduler signal syscalls.

        Killing the launcher can orphan its stopped process group, causing
        kernel HUP/CONT. The already-bound coordinator must immediately be
        killed or observed naturally exited, never manually continued.
        """
        require(self.execute, "DRY_RUN_NO_SIGNALS")
        require(proc_info(coordinator["pid"])["state"] in ("T", "t"), "Coordinator must be STOPPED before pair removal")
        for expected in (launcher, coordinator):
            require(expected["pid"] in self.fds and self.fds[expected["pid"]][0] == expected,
                    "Both schedulers must be identity-bound before pair removal")
        self.event("SCHEDULER_PAIR_KILL_REQUEST", launcher=launcher, coordinator=coordinator,
                   order="launcher_then_coordinator_no_intermediate_log_wait_or_fsync", manual_SIGCONT=False)
        records = []
        for expected in (launcher, coordinator):
            observation = exited_observation(expected)
            if observation["exact_identity_exited"]:
                records.append(dict(identity=expected, sent=False, observation=observation))
                continue
            fd = self.fds[expected["pid"]][1]
            require(exact_live(expected), "Exact scheduler identity changed immediately before final kill")
            try:
                if fd is not None:
                    signal.pidfd_send_signal(fd, signal.SIGKILL, None, 0)
                else:
                    os.kill(expected["pid"], signal.SIGKILL)
            except ProcessLookupError:
                observation = exited_observation(expected)
                require(observation["exact_identity_exited"], "Scheduler lookup race without actual exit evidence")
                records.append(dict(identity=expected, sent=False, observation=observation))
                continue
            records.append(dict(identity=expected, signal=int(signal.SIGKILL), signal_name="SIGKILL", sent=True,
                                signal_mode="EXACT_PIDFD" if fd is not None else "EXACT_IDENTITY_IMMEDIATE_OS_KILL_KERNEL_3_10"))
        for record in records:
            if record["sent"]:
                self.sent.append(record)
            self.event("SCHEDULER_PAIR_REMOVAL_ACTUAL", **record)
        return records


def wait_stopped(identity, seconds=10):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        info = proc_info(identity["pid"])
        require(info is not None and info["identity"] == identity, "Scheduler exited/changed while freezing; inspect before proceeding")
        if info["state"] in ("T", "t"):
            return info
        time.sleep(.05)
    raise RuntimeError("Exact scheduler did not enter STOP state")


def observe_release(identities, seconds, events):
    deadline = time.monotonic() + seconds
    first_exit, first_release = {}, {}
    last = None
    while True:
        now = time.monotonic()
        observations = [exited_observation(x) for x in identities]
        nvml = query_nvml()
        live_gpu = []
        for expected in identities:
            pid = expected["pid"]
            exit_record = next(x for x in observations if x["expected"] == expected)
            if exit_record["exact_identity_exited"] and pid not in first_exit:
                first_exit[pid] = dict(observed_monotonic=now, **exit_record)
                events.event("FIRST_EXACT_PROCESS_EXIT_OBSERVED", observation=first_exit[pid])
            app = [a for a in nvml if a["pid"] == pid and (a["actual_process"] is None or
                   a["actual_process"]["identity"]["starttick"] == expected["starttick"])]
            if app:
                live_gpu.extend(app)
            elif pid not in first_release:
                first_release[pid] = dict(identity=expected, observed_monotonic=now,
                                          own_NVML_absent=True, exact_physical_GPUh="UNRESOLVED")
                events.event("FIRST_OWN_NVML_ABSENCE_OBSERVED", observation=first_release[pid])
        last = dict(processes=observations, NVML=nvml, own_gpu=live_gpu, monotonic=now)
        if all(x["exact_identity_exited"] for x in observations) and not live_gpu:
            return dict(status="PASS_EXACT_IDENTITIES_EXITED_AND_OWN_NVML_RELEASED", first_exit=list(first_exit.values()),
                        first_release=list(first_release.values()), final=last, exitcodes_not_invented=True)
        if now >= deadline:
            return dict(status="UNRESOLVED_AFTER_REAL_OBSERVATION_TIMEOUT", first_exit=list(first_exit.values()),
                        first_release=list(first_release.values()), final=last, timeout_seconds=seconds,
                        exitcodes_not_invented=True)
        time.sleep(.5)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--execute", action="store_true")
    ap.add_argument("--authorization-sha256")
    ap.add_argument("--expected-self-sha256")
    ap.add_argument("--schedulers-first", action="store_true",
                    help="Remove exact launcher/coordinator after binding workers, then TERM workers; handles launcher job-control resume")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    require(root == REMOTE_ROOT and Path(__file__).resolve() == root / "operations/stop_p1_authorized.py",
            "Only the exact independent55 continuation operations root is accepted")
    auth_sha, auth = validate_authorization(root / "HUMAN_CONTINUATION_AUTHORIZATION.json",
                                            args.authorization_sha256, args.execute)
    self_sha = sha(Path(__file__))
    if args.execute:
        require(args.expected_self_sha256 == self_sha, "EXECUTE requires the independently reviewed exact tool SHA")
    elif args.expected_self_sha256:
        require(args.expected_self_sha256 == self_sha, "Tool SHA differs")
    coordinator, launcher, launch_receipt = verify_frozen_p1()
    jobs = jobs_by_id()
    workers, unknown = scan_workers(coordinator, jobs)
    require(not unknown, "Unknown coordinator children exist before freezing; inspect instead of guessing")
    nvml = query_nvml()
    check_nvml_ownership(workers, nvml)
    preview = dict(status="READ_ONLY_DRY_RUN_NO_SIGNALS", execution_attempt=ATTEMPT,
                   human_authorization_sha256=auth_sha, tool_sha256=self_sha,
                   P1_source_freeze_sha256=PINS["SOURCE_FREEZE.json"], coordinator=coordinator,
                   launcher=launcher, workers=workers, NVML=nvml, launch_receipt=launch_receipt,
                   proposed_order=["SIGSTOP exact run_offline bash", "SIGSTOP exact coordinator",
                       "Rescan dispatch-owned workers after scheduler STOP", "SIGTERM exact live workers",
                       "Observe actual exit and ownNVML up to120s", "SIGKILL still-STOP exact coordinator",
                       "SIGKILL still-STOP exact launcher, never SIGCONT", "Verify nofuture scheduler/ownworker and UUID release"],
                   no_action_without_execute=True, no_formal_or_resident_launch=True)
    preview["schedulers_first"] = args.schedulers_first
    if args.schedulers_first:
        preview["proposed_order"] = ["SIGSTOP exact coordinator, accepting alreadyT", "Request exact launcher SIGSTOP without relying on persistentT",
            "Rescan and bind all own workers", "Adjacent exact KILL launcher then coordinator; accept natural HUP exit",
            "Rescan STARTED/dispatch-owned orphan workers", "TERM only exact bound own workers",
            "Actual exit/ownNVML release observation120s", "Independent PARTIAL cancellation receipt; no original finisher"]
    if not args.execute:
        print(json.dumps(preview, ensure_ascii=False, sort_keys=True), flush=True)
        return 0
    out = root / "operations/P1_AUTHORIZED_STOP"
    require(not out.exists(), "Stop attempt already exists; preserve and inspect, no automatic signal retry")
    out.mkdir(exist_ok=False)
    events = SignalControl(out / "SIGNAL_EVENTS.jsonl", True)
    save(out / "STARTED.json", dict(identity=proc_info(os.getpid()), monotonic=time.monotonic(),
                human_authorization_sha256=auth_sha, tool_sha256=self_sha, exact_scope=PINS,
                original_source_or_receipt_writes=0, restart_authorized=False))
    try:
        save(out / "PRE_STOP_PROCESS_AND_NVML.json", preview)
        save(out / "PRE_STOP_METADATA_CATALOG.json", metadata_snapshot())
        events.bind(launcher)
        events.bind(coordinator)
        if args.schedulers_first:
            events.send(coordinator, signal.SIGSTOP)
            wait_stopped(coordinator)
            events.send(launcher, signal.SIGSTOP)
        else:
            events.send(launcher, signal.SIGSTOP)
            wait_stopped(launcher)
            events.send(coordinator, signal.SIGSTOP)
            wait_stopped(coordinator)
        # A worker may have been Popen'ed immediately before SIGSTOP. Waiting
        # for stable known children does not resume either stopped scheduler.
        deadline = time.monotonic() + 10
        while True:
            workers, unknown = scan_workers(coordinator, jobs)
            if not unknown:
                break
            require(time.monotonic() < deadline, "Unknown post-fork child persists; schedulers remain STOPPED for review")
            time.sleep(.2)
        nvml = query_nvml()
        check_nvml_ownership(workers, nvml)
        save(out / "SCHEDULERS_FROZEN_AND_RESCANNED.json", dict(coordinator=proc_info(coordinator["pid"]),
                    launcher=proc_info(launcher["pid"]), workers=workers, NVML=nvml,
                    monotonic=time.monotonic(), no_new_dispatch_possible=True))
        # Include pre-stop workers that naturally exited during the freeze, so
        # their exact exit observations are not lost or reclassified as TERM.
        identities = {x["identity"]["pid"]: x["identity"] for x in preview["workers"]}
        for worker in workers:
            identities[worker["identity"]["pid"]] = worker["identity"]
            if exact_live(worker["identity"]):
                events.bind(worker["identity"])
        if args.schedulers_first:
            removal = events.kill_scheduler_pair(launcher, coordinator)
            save(out / "SCHEDULER_PAIR_REMOVAL.json", dict(actual=removal, manual_SIGCONT=False,
                 note="Kernel orphan-group HUP/CONT may naturally end coordinator; actual identities are rechecked"))
            # Discover any last Popen/exec race without accepting an arbitrary
            # same-user process. Reparented workers need durable exact STARTED.
            deadline = time.monotonic() + 10
            while True:
                try:
                    after_workers, unknown = scan_workers(coordinator, jobs, allow_reparented=True)
                    require(not unknown, "Unknown child after scheduler removal")
                    break
                except RuntimeError:
                    require(time.monotonic() < deadline, "Post-scheduler worker ownership unresolved; no GPU admission")
                    time.sleep(.2)
            nvml = query_nvml()
            check_nvml_ownership(after_workers, nvml)
            for worker in after_workers:
                identities[worker["identity"]["pid"]] = worker["identity"]
                prior = events.fds.get(worker["identity"]["pid"])
                if exact_live(worker["identity"]) and (prior is None or prior[0] != worker["identity"]):
                    events.bind(worker["identity"])
            workers = after_workers
            save(out / "POST_SCHEDULER_ROOT_WORKER_RESCAN.json", dict(workers=workers, NVML=nvml,
                 exact_coordinator_exit=exited_observation(coordinator), exact_launcher_exit=exited_observation(launcher)))
        for worker in workers:
            events.send(worker["identity"], signal.SIGTERM)
        release = observe_release(list(identities.values()), 120, events)
        save(out / "WORKER_EXIT_AND_NVML_RELEASE.json", release)
        require(release["status"] == "PASS_EXACT_IDENTITIES_EXITED_AND_OWN_NVML_RELEASED",
                "Worker exit/release unresolved: do not claim stop complete or use GPU; schedulers remain STOPPED")
        save(out / "POST_WORKER_METADATA_CATALOG.json", metadata_snapshot())
        # SIGTERM is not used on STOPPED bash/controller: resuming them could
        # dispatch the next sample or run the original unconditional finisher.
        if not args.schedulers_first:
            require(proc_info(coordinator["pid"])["state"] in ("T", "t") and
                    proc_info(launcher["pid"])["state"] in ("T", "t"), "A scheduler unexpectedly resumed")
            events.send(coordinator, signal.SIGKILL)
            events.send(launcher, signal.SIGKILL)
        schedulers = observe_release([coordinator, launcher], 30, events)
        save(out / "SCHEDULER_EXIT.json", schedulers)
        require(schedulers["status"] == "PASS_EXACT_IDENTITIES_EXITED_AND_OWN_NVML_RELEASED",
                "Scheduler exact exit unresolved; no resident admission")
        remaining, orphan_children = scan_workers(coordinator, jobs, allow_reparented=True)
        require(not remaining and not orphan_children, "An original P1 worker remains after cancellation")
        final_nvml = query_nvml()
        for app in final_nvml:
            info = app["actual_process"]
            require(not info or str(P1 / "scripts/gpu_unit_worker.py") not in info["identity"]["cmdline"],
                    "An original P1 GPU worker remains; do not use the device")
        for relative, expected in PINS.items():
            require(sha(P1 / relative) == expected, "Old frozen control/source changed during authorized cancellation")
        post = metadata_snapshot()
        save(out / "POST_STOP_METADATA_CATALOG.json", post)
        complete = []
        for job in jobs.values():
            if job["condition"] != "P1":
                continue
            target = P1 / "formal" / job["id"]
            if (target / "RESULT.json").is_file() and (target / "RESOURCE_INTERVAL.json").is_file():
                interval = load(target / "RESOURCE_INTERVAL.json")
                if interval["exit_returncode"] == 0 and interval["release"] == "PASS":
                    complete.append(dict(unit=job["id"], result_sha256=sha(target / "RESULT.json"),
                                         interval_sha256=sha(target / "RESOURCE_INTERVAL.json")))
        receipt = dict(status="PASS_HUMAN_AUTHORIZED_P1_STOP_EXACT_IDENTITIES_EXITED_OWN_GPU_RELEASED",
                       terminal="USER_CANCELLED_P1_PARTIAL_DO_NOT_RESUME", human_authorization_sha256=auth_sha,
                       tool_sha256=self_sha, P1_root=str(P1), P1_source_freeze_sha256=PINS["SOURCE_FREEZE.json"],
                       P1_permit_sha256=PINS["GPU_PERMIT.json"], original_launcher=launcher,
                       original_coordinator=coordinator, stopped_workers=list(identities.values()),
                       no_future_original_scheduler_dispatch=True, all_exact_own_processes_exited=True,
                       own_GPU_UUIDs_released=True, final_NVML=final_nvml,
                       selected_UUIDs_currently_idle=not any(x["uuid"] in UUIDS.values() for x in final_nvml),
                       original_completed_candidate_count=len(complete), original_completed_candidates=complete,
                       completed_candidates_require_independent_payload_validation=True,
                       in_flight_samples="HUMAN_CANCELLED_PARTIAL_NOT_SCORED_OR_CONCATENATED",
                       old_C1_S3_failure_terminations_preserved=True, old_source_rewritten=False,
                       original_SESSION_RESULT_fabricated=False, original_ledger_rewritten=False,
                       raw_Tensors_or_RAM_references_deleted=False, resident_GPU_dispatches=0,
                       exact_physical_GPUh="UNRESOLVED", monotonic=time.monotonic())
        save(out / "USER_STOP_RECEIPT.json", receipt)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        if not (out / "STOP_FAILURE.json").exists():
            save(out / "STOP_FAILURE.json", dict(status="AUTHORIZED_STOP_UNRESOLVED_PRESERVE_AND_INSPECT_NO_AUTO_RETRY",
                 error=repr(error), traceback=traceback.format_exc(), signals_sent=events.sent,
                 coordinator_current=proc_info(coordinator["pid"]), launcher_current=proc_info(launcher["pid"]),
                 schedulers_are_not_automatically_resumed=True, no_GPU_admission=True,
                 raw_Tensors_or_RAM_references_deleted=False))
        raise
    finally:
        events.close()


if __name__ == "__main__":
    raise SystemExit(main())
