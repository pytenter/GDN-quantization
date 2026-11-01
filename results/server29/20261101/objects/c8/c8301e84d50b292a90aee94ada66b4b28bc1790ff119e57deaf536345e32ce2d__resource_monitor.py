"""CPU controller resource wrapper. No CUDA/model imports or GPU time caps.

The injected sampler is the frozen read-only telemetry collector. Its complete
output and independent process observations are fsynced before interpretation.
An exact, SHA-pinned AuthorizedGraphicsPolicy is required; desktop authorization
never authorizes a compute PID. This module owns wait/reap for registered Popen
children: callers must not poll(), wait(), or reap them in another thread.
"""
from dataclasses import dataclass, field
import copy
import csv
import functools
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from lifecycle_visibility import Owner, inspect_sample, reconcile_triplet

GIB = 2 ** 30
MIB = 2 ** 20
RUNNING_FREE = 2 * GIB
IDLE_FREE = 22632464384
UNATTRIBUTED = 64 * MIB
TELEMETRY_LOCK = threading.RLock()  # libNVML init/shutdown is process-global.
DECISION_LOCK = threading.RLock()  # shared first STOP / final EXIT / Popen boundary
PAIRS = {0: "GPU-283734e9-d350-ed3b-a9f2-e3ec1494e186",
         1: "GPU-936da32c-c539-627e-7f50-bd12fb7e9521"}
TASK = "LING_KDA_GROUP32_FOUR_CODEC_AIME26_256K_SEED1_V1"

HARDWARE = {
 "GPU-283734e9-d350-ed3b-a9f2-e3ec1494e186": ("NVIDIA GeForce RTX 4090",24564),
 "GPU-936da32c-c539-627e-7f50-bd12fb7e9521": ("NVIDIA GeForce RTX 4090",24564),
 "GPU-07d9c381-888d-1111-a944-56abcc583573": ("NVIDIA GeForce RTX 3090",24576),
 "GPU-de53f243-b271-521c-d875-d96611ed8574": ("NVIDIA GeForce RTX 3090",24576)}

class ResourceFailure(RuntimeError):
    pass


def _require(condition, reason):
    if condition is not True:
        raise ResourceFailure(reason)


def _latched(method):
    @functools.wraps(method)
    def call(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as error:
            if not self.failed:
                self.fail(error)
            raise
    return call


def _integer(value, reason, minimum=0):
    _require(type(value) is int and minimum <= value < 2 ** 64 - 1, reason)
    return value


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       allow_nan=False, separators=(",", ":")) + "\n").encode()


def _exclusive_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _bytes(value)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest()}


def read_proc_identity(pid):
    """Actual effective UID/start/PPid/boot evidence, never launch intent/exe."""
    try:
        directory = Path("/proc") / str(_integer(pid, "INVALID_PID", 1))
        stat = (directory / "stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        status = {}
        for line in (directory / "status").read_text().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                status[key] = value.strip()
        return {"status": "PASS", "pid": pid,
                "uid": int(status["Uid"].split()[1]),
                "start_ticks": int(fields[19]), "ppid": int(fields[1]),
                "state": fields[0], "comm": (directory / "comm").read_text().strip(),
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "identity_scope": "actual /proc stat,status,comm and boot; exe not claimed"}
    except Exception as error:
        return {"status": "UNKNOWN", "pid": pid, "error": repr(error)}


def observe_unreaped(popen):
    """Observe our direct child's exit without freeing its PID for reuse."""
    try:
        _require(popen.returncode is None, "CHILD_REAPED_OUTSIDE_MONITOR")
        flags = os.WEXITED | os.WNOHANG | os.WNOWAIT
        value = os.waitid(os.P_PID, popen.pid, flags)
        if value is None:
            return {"status": "LIVE_UNREAPED", "pid": popen.pid}
        _require(value.si_pid == popen.pid, "WAITID_WRONG_CHILD")
        if value.si_code == os.CLD_EXITED:
            rc = value.si_status
        elif value.si_code in (os.CLD_KILLED, os.CLD_DUMPED):
            rc = -value.si_status
        else:
            raise ResourceFailure("WAITID_EXIT_STATE_UNKNOWN")
        return {"status": "EXITED_UNREAPED", "pid": popen.pid, "returncode": rc}
    except Exception as error:
        return {"status": "UNKNOWN", "pid": getattr(popen, "pid", None),
                "error": repr(error)}


def _identity_equal(actual, owner):
    return (isinstance(actual, dict) and actual.get("status") == "PASS" and
            all(type(actual.get(key)) is int for key in ("pid", "uid", "start_ticks")) and
            all(actual.get(key) == getattr(owner, key)
                for key in ("boot_id", "pid", "uid", "start_ticks")))


def _timing(value):
    _require(isinstance(value, dict), "UNKNOWN_QUERY_TIMESTAMPS")
    begin, end = value.get("started_monotonic"), value.get("completed_monotonic")
    _require(type(begin) in (float, int) and type(end) in (float, int) and
             math.isfinite(begin) and math.isfinite(end) and 0 <= begin <= end,
             "UNKNOWN_QUERY_TIMESTAMPS")
    # Preserve original float seconds in raw. These are conversions, not a
    # claim that the original clock had nanosecond measurement precision.
    return int(begin * 1_000_000_000), int(end * 1_000_000_000)


def _query_sets(snapshot, uuid):
    """Strictly parse the three sources without deleting or normalizing raw."""
    groups = snapshot.get("direct_NVML_processes", {})
    n = groups.get("compute", {})
    _require(n.get("status") == "PASS" and isinstance(n.get("processes"), list),
             "UNKNOWN_NVML_COMPUTE")
    nvml = [_integer(row.get("pid"), "UNKNOWN_NVML_COMPUTE_PID", 1)
            for row in n["processes"]]
    raw_csv = snapshot.get("compute", {})
    _require(raw_csv.get("returncode") == 0 and type(raw_csv.get("returncode")) is int and
             isinstance(raw_csv.get("stdout"), str), "UNKNOWN_SMI_COMPUTE_CSV")
    early = []
    for values in csv.reader(io.StringIO(raw_csv["stdout"])):
        if not values or all(not value.strip() for value in values):
            continue
        _require(len(values) == 3, "UNKNOWN_SMI_COMPUTE_CSV")
        actual, pid, used = (value.strip() for value in values)
        _require(actual == uuid and pid.isdigit() and int(pid) > 0,
                 "UNKNOWN_SMI_COMPUTE_CSV_IDENTITY")
        early.append({"uuid": actual, "pid": int(pid), "used_gpu_memory": used})
    captured = snapshot.get("early_SMI_compute_processes", {})
    _require(captured.get("status") == "PASS" and captured.get("processes") == early,
             "UNKNOWN_EARLY_SMI_CAPTURE_OR_CONFLICT")
    xml = snapshot.get("SMI_all_processes", {})
    _require(xml.get("status") == "PASS" and isinstance(xml.get("processes"), list),
             "UNKNOWN_SMI_XML_PROCESSES")
    xml_compute = []
    for row in xml["processes"]:
        _require(isinstance(row, dict) and row.get("type") in ("G", "C", "C+G", "G+C"),
                 "UNKNOWN_SMI_PROCESS_ROLE")
        pid = row.get("pid")
        _require(isinstance(pid, str) and pid.isdigit() and int(pid) > 0,
                 "UNKNOWN_SMI_XML_PID")
        if row["type"] != "G":
            xml_compute.append(int(pid))
    csv_pids = [row["pid"] for row in early]
    for values in (nvml, csv_pids, xml_compute):
        _require(len(values) == len(set(values)), "UNKNOWN_DUPLICATE_COMPUTE_PID")
    return [("NVML", nvml, n.get("query_timing")),
            ("SMI_COMPUTE_CSV", csv_pids, raw_csv.get("query_timing")),
            ("SMI_XML", xml_compute, snapshot.get("SMI_XML", {}).get("query_timing"))]


@dataclass
class Child:
    popen: object
    owner: Owner
    tag: str
    spawn_monotonic_ns: int
    spawn_wall_ns: int
    admission: dict
    initial_actual_identity: dict
    last_actual_identity: dict
    last_actual_identity_ns: int
    previous: dict
    previous_ref: dict
    phase: str = "INITIALIZING"
    pending: list = field(default_factory=list)
    nvml_owned_peak_bytes: object = None
    nvml_owned_peak_samples: int = 0
    nvml_owned_peak_unknown_samples: int = 0
    closed: bool = False
    lock: object = field(default_factory=threading.RLock, repr=False)


class ResourceMonitor:
    def __init__(self, directory, gpu_index, gpu_uuid, sampler, graphics_policy, *,
                 stop_path=None, proc_reader=read_proc_identity,
                 exit_observer=observe_unreaped, monotonic_ns=time.monotonic_ns,
                 wall_ns=time.time_ns, sleep=time.sleep, initialization_timeout_seconds=180,
                 max_adjacent_gap_seconds=15, telemetry_lock=TELEMETRY_LOCK):
        _require(type(gpu_index) is int and PAIRS.get(gpu_index) == gpu_uuid,
                 "UNAUTHORIZED_GPU_PAIR")
        _require(callable(sampler) and callable(proc_reader) and callable(exit_observer),
                 "MISSING_RUNTIME_DEPENDENCY")
        _require(type(initialization_timeout_seconds) in (int, float) and
                 math.isfinite(initialization_timeout_seconds) and initialization_timeout_seconds > 0,
                 "INITIALIZATION_BOUND_REQUIRED")
        _require(type(max_adjacent_gap_seconds) in (int, float) and
                 math.isfinite(max_adjacent_gap_seconds) and max_adjacent_gap_seconds > 0,
                 "NEIGHBOR_BOUND_REQUIRED")
        _require(callable(getattr(graphics_policy, "evidence", None)) and
                 callable(getattr(graphics_policy, "compute_union", None)),
                 "PINNED_GRAPHICS_POLICY_REQUIRED")
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stop_path = Path(stop_path or self.directory / "FIRST_FAILURE.json").resolve()
        self.index, self.uuid = gpu_index, gpu_uuid
        self.sampler, self.graphics = sampler, graphics_policy
        self.proc_reader, self.exit_observer = proc_reader, exit_observer
        self.monotonic_ns, self.wall_ns, self.sleep = monotonic_ns, wall_ns, sleep
        self.initialization_ns = int(initialization_timeout_seconds * 1e9)
        self.max_gap_ns = int(max_adjacent_gap_seconds * 1e9)
        self.telemetry_lock, self.lock = telemetry_lock, threading.RLock()
        self.active, self.jobs, self.admissions = None, [], {}
        self.last_capture_end = None
        self.latest_evidence_ref = None

    @property
    def pending(self):
        with self.lock:
            return [] if self.active is None else copy.deepcopy(self.active.pending)

    @property
    def failed(self):
        return self.stop_path.exists()

    def fail(self, reason, *, evidence_ref=None, details=None):
        record = {"status": "FIRST_RESOURCE_FAILURE_NO_RETRY", "reason": str(reason),
                  "task_id": TASK,
                  "gpu_index": self.index, "gpu_uuid": self.uuid,
                  "observed_wall_ns": self.wall_ns(), "observed_monotonic_ns": self.monotonic_ns(),
                  "evidence_ref": evidence_ref or self.latest_evidence_ref, "details": details,
                  "pending_resource_pass": False}
        with DECISION_LOCK:
            try:
                _exclusive_json(self.stop_path, record)
            except FileExistsError:
                pass
        raise ResourceFailure(str(reason))

    def _guard(self):
        _require(not self.failed, "EXISTING_FIRST_FAILURE_NO_RETRY")

    def _observation(self, child):
        at = self.monotonic_ns()
        identity = self.proc_reader(child.owner.pid)
        wait = self.exit_observer(child.popen)
        return {"monotonic_ns": at, "wall_ns": self.wall_ns(),
                "actual_proc_identity": identity, "parent_wait_observation": wait}

    def capture(self, child=None):
        """Persist raw collector output plus before/after facts before parsing."""
        self._guard()
        with self.telemetry_lock:
            begin = self.monotonic_ns()
            before = self._observation(child) if child is not None else None
            try:
                snapshot = self.sampler(self.uuid)
            except Exception as error:
                snapshot = {"sampler_error": repr(error)}
            after = self._observation(child) if child is not None else None
            end = self.monotonic_ns()
            record = {"gpu_index": self.index, "gpu_uuid": self.uuid,
                      "capture_begin_monotonic_ns": begin, "capture_end_monotonic_ns": end,
                      "capture_wall_ns": self.wall_ns(), "lifecycle_before": before,
                      "lifecycle_after": after, "snapshot": snapshot}
            # Preserve nonfinite raw values too; they are rejected below, not
            # silently changed to null before their first-failure evidence.
            data = (json.dumps(record, sort_keys=True, ensure_ascii=False,
                               allow_nan=True, separators=(",", ":")) + "\n").encode()
            path = self.directory / "RESOURCE_RAW.jsonl"
            with path.open("ab") as stream:
                offset = stream.tell()
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            ref = {"path": str(path), "offset": offset, "length": len(data),
                   "sha256": hashlib.sha256(data).hexdigest()}
            self.last_capture_end = end
            self.latest_evidence_ref = ref
            return record, ref

    def _safety(self, record, child=None):
        s = record["snapshot"]
        _require(isinstance(s, dict), "UNKNOWN_SNAPSHOT")
        _require(s.get("uuid") == self.uuid and len(s.get("target", [])) == 1,
                 "DEVICE_IDENTITY_FAILURE")
        target = s["target"][0]
        _require(target.get("uuid") == self.uuid and int(target["index"]) == self.index and
                 target.get("name") == HARDWARE[self.uuid][0] and
                 float(target["total_MiB"]) == HARDWARE[self.uuid][1], "DEVICE_IDENTITY_FAILURE")
        for key in ("physical", "compute", "SMI_XML"):
            _require(type(s.get(key, {}).get("returncode")) is int and
                     s[key]["returncode"] == 0, "UNKNOWN_SAFETY_API:" + key)
        n = s.get("direct_NVML", {})
        _require(n.get("status") == "PASS", "UNKNOWN_DIRECT_NVML")
        for key in ("memory", "memory_v2", "utilization", "temperature", "power_mW", "pstate", "SM_clock_MHz"):
            _require(key in n and not (isinstance(n[key], dict) and n[key].get("status", "PASS") != "PASS"),
                     "UNKNOWN_NVML_API:" + key)
        for key in ("temperature", "power_mW", "pstate", "SM_clock_MHz"):
            _require(type(n[key]) in (int, float) and math.isfinite(n[key]) and n[key] >= 0,
                     "UNKNOWN_NVML_API:" + key)
        for key in ("total", "used", "free"):
            _integer(n["memory"].get(key), "UNKNOWN_NVML_MEMORY_V1")
        m = n["memory_v2"]
        for key in ("total", "used", "free", "reserved"):
            _integer(m.get(key), "UNKNOWN_NVML_MEMORY_V2")
        _require(m["total"] == m["used"] + m["free"] + m["reserved"], "MEMORY_ARITHMETIC_FAILURE")
        _require(m["free"] >= RUNNING_FREE, "TRUE_FREE_BELOW_2GIB")
        for key in ("gpu", "memory"):
            value = n["utilization"].get(key)
            _require(type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 100,
                     "UNKNOWN_NVML_UTILIZATION")
        smi_util, smi_used = float(target["GPU_util"]), float(target["used_MiB"]) * MIB
        _require(math.isfinite(smi_util) and 0 <= smi_util <= 100 and
                 math.isfinite(smi_used) and smi_used >= 0, "UNKNOWN_PHYSICAL_API")
        sources = _query_sets(s, self.uuid)
        union = set().union(*(set(pids) for _, pids, _ in sources))
        allowed = set() if child is None else {child.owner.pid}
        _require(not (union - allowed), "FOREIGN_COMPUTE_STOP:" + repr(sorted(union - allowed)))
        graphics = self.graphics.evidence(s, self.uuid)
        bound_owner = None if child is None else {"pid": child.owner.pid}
        checked_union = self.graphics.compute_union(s, self.uuid, bound_owner)
        _require(set(checked_union) == union, "COMPUTE_UNION_POLICY_DISAGREEMENT")
        _require(graphics.get("free_bytes") == m["free"] and
                 graphics.get("allocated_bytes") == m["used"] and
                 graphics.get("reserved_bytes") == m["reserved"], "GRAPHICS_POLICY_MEMORY_FIELDS_CHANGED")
        residual = graphics.get("signed_unattributed_residual_bytes")
        _require(type(residual) is int and residual >= -UNATTRIBUTED, "UNKNOWN_NEGATIVE_MEMORY_RESIDUAL")
        if child is None:
            _require(not union and residual <= UNATTRIBUTED, "IDLE_UNATTRIBUTED_MEMORY_OR_COMPUTE")
            _require(abs(m["used"] - smi_used) <= UNATTRIBUTED, "IDLE_MEMORY_TELEMETRY_MISMATCH")
        queries, resolutions = [], []
        if child is not None:
            resolutions = self._validate_lifecycle(record, child, union)
            own_nvml = [row for row in s["direct_NVML_processes"]["compute"]["processes"]
                        if row.get("pid") == child.owner.pid]
            if own_nvml:
                measured = own_nvml[0].get("usedGpuMemory")
                if type(measured) is int and 0 <= measured < 2 ** 64 - 1:
                    child.nvml_owned_peak_bytes = max(child.nvml_owned_peak_bytes or 0, measured)
                    child.nvml_owned_peak_samples += 1
                else:
                    child.nvml_owned_peak_unknown_samples += 1
        owners = s.get("process_ownership")
        _require(isinstance(owners, list) and all(isinstance(row, dict) for row in owners),
                 "UNKNOWN_PROCESS_OWNERSHIP")
        for api, pids, timing in sources:
            begin, end = _timing(timing)
            _require(record["capture_begin_monotonic_ns"] <= begin <= end <= record["capture_end_monotonic_ns"],
                     "QUERY_OUTSIDE_REAL_CAPTURE_INTERVAL")
            rows = []
            for pid in pids:
                actual = [row for row in owners if row.get("pid") == pid]
                _require(len(actual) <= 1, "DUPLICATE_PROCESS_OWNERSHIP")
                if actual and actual[0].get("status") == "PASS":
                    _require(type(actual[0].get("uid")) is int and type(actual[0].get("start_ticks")) is int and
                             actual[0]["uid"] == child.owner.uid and actual[0]["start_ticks"] == child.owner.start_ticks,
                             "CAPTURED_OWN_IDENTITY_CHANGED")
                    _require("boot_id" not in actual[0] or actual[0]["boot_id"] == child.owner.boot_id,
                             "CAPTURED_OWN_BOOT_CHANGED")
                    basis = "actual_raw_process_ownership_plus_actual_before_after_lifecycle"
                else:
                    # A late API may mention our context after /proc collection
                    # lost visibility. Never edit that UNKNOWN raw observation.
                    _require(any(item.get("status") == "EXITED_UNREAPED" for item in
                                 (record["lifecycle_before"]["parent_wait_observation"],
                                  record["lifecycle_after"]["parent_wait_observation"])),
                             "UNKNOWN_OWN_RAW_IDENTITY_WHILE_LIVE")
                    basis = "derived_actual_before_identity_and_parent_unreaped_lifetime_not_raw_identity"
                rows.append({"boot_id": child.owner.boot_id, "pid": pid,
                             "uid": child.owner.uid, "start_ticks": child.owner.start_ticks})
                resolutions.append({"api": api, "pid": pid, "basis": basis,
                                    "actual_identity_evidence": child.last_actual_identity,
                                    "actual_identity_monotonic_ns": child.last_actual_identity_ns})
            queries.append({"api": api, "success": True, "gpu_uuid": self.uuid,
                            "begin_ns": begin, "end_ns": end, "compute_rows": rows})
        ordered = sorted(queries, key=lambda row: row["begin_ns"])
        _require(all(a["end_ns"] <= b["begin_ns"] for a, b in zip(ordered, ordered[1:])),
                 "UNKNOWN_NONSEQUENTIAL_QUERY_INTERVALS")
        return {"visibility": {"queries": queries}, "compute_union": sorted(union),
                "graphics": graphics, "identity_resolutions": resolutions,
                "SMI_util": smi_util, "NVML_util": n["utilization"]["gpu"],
                "free_bytes": m["free"], "signed_unattributed_residual_bytes": residual,
                "resource_pass": False}

    def _validate_lifecycle(self, record, child, union):
        _require(child.popen.pid == child.owner.pid and child.popen.returncode is None and not child.closed,
                 "OWN_CHILD_LIFETIME_LOST_OR_ALREADY_REAPED")
        resolutions = []
        for key in ("lifecycle_before", "lifecycle_after"):
            evidence = record[key]
            wait = evidence["parent_wait_observation"]
            _require(wait.get("pid") == child.owner.pid and
                     wait.get("status") in ("LIVE_UNREAPED", "EXITED_UNREAPED"), "UNKNOWN_PARENT_CHILD_WAIT")
            if wait["status"] == "EXITED_UNREAPED":
                _require(type(wait.get("returncode")) is int and wait["returncode"] == 0,
                         "NONZERO_OWN_CHILD_EXIT")
            actual = evidence["actual_proc_identity"]
            if actual.get("status") == "PASS":
                _require(_identity_equal(actual, child.owner) and actual.get("ppid") == os.getpid(),
                         "OWN_ACTUAL_IDENTITY_OR_PARENT_CHANGED")
                child.last_actual_identity, child.last_actual_identity_ns = actual, evidence["monotonic_ns"]
            else:
                _require(wait["status"] == "EXITED_UNREAPED" and
                         _identity_equal(child.last_actual_identity, child.owner) and
                         0 <= evidence["monotonic_ns"] - child.last_actual_identity_ns <= self.max_gap_ns,
                         "UNKNOWN_OWN_LIFECYCLE_IDENTITY")
                resolutions.append({"status": "DERIVED_UNREAPED_IDENTITY_CONTINUITY_NOT_RAW_PASS",
                                    "actual_before_identity": child.last_actual_identity,
                                    "actual_before_monotonic_ns": child.last_actual_identity_ns})
        return resolutions

    def _persist_derived(self, tag, value):
        return _exclusive_json(self.directory / "derived" / (tag + ".json"), value)

    def _window(self, tag, purpose):
        rows, values, refs = [], [], []
        for number in range(3):
            if number:
                self.sleep(5)
            self._guard()
            record, ref = self.capture()
            try:
                value = self._safety(record)
                _require(value["free_bytes"] >= IDLE_FREE, "IDLE_TRUE_FREE_BELOW_FROZEN_HEADROOM")
            except Exception as error:
                self.fail(error, evidence_ref=ref)
            rows.append(record)
            values.append(value)
            refs.append(ref)
        _require(all(rows[i + 1]["capture_begin_monotonic_ns"] - rows[i]["capture_end_monotonic_ns"] >= 5_000_000_000
                     for i in range(2)), "WINDOW_SPACING_NOT_FIVE_SECONDS")
        base = values[0]["graphics"]["graphics"]
        _require(all(value["graphics"]["graphics"] == base for value in values), "GRAPHICS_IDENTITY_NOT_STABLE")
        residuals = [max(0, value["signed_unattributed_residual_bytes"]) for value in values]
        _require(max(residuals) - min(residuals) <= UNATTRIBUTED, "UNATTRIBUTED_MEMORY_GROWTH")
        util = [max(value["SMI_util"], value["NVML_util"]) for value in values]
        _require(sum(value > 2 for value in util) < 2 and util[-1] <= 2,
                 "SUSTAINED_OR_CURRENT_MATERIAL_GPU_OCCUPANCY")
        receipt = {"status": "PASS", "purpose": purpose, "gpu_index": self.index,
                   "gpu_uuid": self.uuid, "raw_refs": refs, "sample_count": 3,
                   "spacing_seconds": 5, "minimum_true_free_bytes": min(v["free_bytes"] for v in values),
                   "completed_monotonic_ns": self.monotonic_ns(), "values": values}
        ref = self._persist_derived(tag + "." + purpose, receipt)
        return receipt, ref, values[-1]["visibility"]

    @_latched
    def admit(self, tag):
        with self.lock:
            self._guard()
            _require(self.active is None and tag not in self.admissions and
                     not any(job["tag"] == tag for job in self.jobs), "DUPLICATE_OR_OVERLAPPING_LAUNCH")
            _require(isinstance(tag, str) and tag and Path(tag).name == tag, "INVALID_JOB_TAG")
            try:
                receipt, ref, visibility = self._window(tag, "ADMISSION")
            except ResourceFailure as error:
                if not self.failed:
                    self.fail(error)
                raise
            self._guard()
            self.admissions[tag] = {"receipt": receipt, "ref": ref, "visibility": visibility,
                                    "consumed": False}
            return ref

    @_latched
    def register_child(self, popen, tag, admission_receipt, *, spawn_monotonic_ns, spawn_wall_ns):
        """Call immediately after Popen with times captured immediately before it."""
        with self.lock:
            self._guard()
            _require(self.active is None and tag in self.admissions and
                     not self.admissions[tag]["consumed"] and
                     self.admissions[tag]["ref"] == admission_receipt, "MISSING_OR_USED_ADMISSION")
            _require(_sha(admission_receipt["path"]) == admission_receipt["sha256"], "ADMISSION_RECEIPT_CHANGED")
            _integer(spawn_monotonic_ns, "MISSING_ACTUAL_SPAWN_TIME")
            _integer(spawn_wall_ns, "MISSING_ACTUAL_SPAWN_WALL_TIME")
            _require(self.admissions[tag]["receipt"]["completed_monotonic_ns"] <= spawn_monotonic_ns <= self.monotonic_ns(),
                     "SPAWN_TIME_PRECEDES_ADMISSION_OR_IS_FUTURE")
            _require(spawn_monotonic_ns - self.admissions[tag]["receipt"]["completed_monotonic_ns"] <= self.max_gap_ns and
                     self.monotonic_ns() - spawn_monotonic_ns <= self.max_gap_ns,
                     "ADMISSION_OR_CHILD_REGISTRATION_STALE")
            _require(popen.returncode is None, "CHILD_ALREADY_REAPED_AT_REGISTRATION")
            actual = self.proc_reader(popen.pid)
            waited = self.exit_observer(popen)
            self.latest_evidence_ref = self._persist_derived(tag + ".REGISTRATION_RAW", {
                "actual_proc_identity": actual, "parent_wait_observation": waited,
                "popen_pid": popen.pid, "spawn_monotonic_ns": spawn_monotonic_ns,
                "spawn_wall_ns": spawn_wall_ns, "observed_monotonic_ns": self.monotonic_ns(),
                "admission_ref": admission_receipt, "scope": "raw actual registration observations before identity judgement"})
            _require(actual.get("status") == "PASS" and actual.get("ppid") == os.getpid(),
                     "OWN_CHILD_REGISTRATION_IDENTITY_UNKNOWN")
            _require(actual.get("uid") == os.geteuid(), "OWN_CHILD_EFFECTIVE_UID_NOT_CONTROLLER_UID")
            owner = Owner(actual.get("boot_id"), actual.get("pid"), actual.get("start_ticks"),
                          actual.get("uid"), self.uuid)
            _require(_identity_equal(actual, owner), "OWN_CHILD_REGISTRATION_IDENTITY_INVALID")
            _integer(owner.pid, "INVALID_REGISTERED_PID", 1)
            _integer(owner.uid, "INVALID_REGISTERED_UID")
            _integer(owner.start_ticks, "INVALID_REGISTERED_START", 1)
            _require(isinstance(owner.boot_id, str) and bool(owner.boot_id), "INVALID_REGISTERED_BOOT")
            _require(waited.get("status") in ("LIVE_UNREAPED", "EXITED_UNREAPED") and waited.get("pid") == owner.pid,
                     "UNREGISTERED_PARENT_CHILD_LIFETIME")
            child = Child(popen, owner, tag, spawn_monotonic_ns, spawn_wall_ns,
                          self.admissions[tag], actual, actual, self.monotonic_ns(),
                          self.admissions[tag]["visibility"], self.admissions[tag]["ref"])
            self.admissions[tag]["consumed"] = True
            self.active = child
            self._persist_derived(tag + ".REGISTERED", {"status": "REGISTERED_ACTUAL_DIRECT_CHILD",
                                  "actual_identity": actual, "parent_wait_observation": waited,
                                  "admission_ref": admission_receipt, "spawn_monotonic_ns": spawn_monotonic_ns,
                                  "spawn_wall_ns": spawn_wall_ns, "gpu_uuid": self.uuid})
            return child

    def _resolve_initialization(self, child, following, following_ref):
        intervals, results = [], []
        for pending in child.pending:
            result = reconcile_triplet(child.previous, pending["visibility"], following,
                                       child.owner, "INITIALIZING", child.spawn_monotonic_ns,
                                       self.monotonic_ns(), live_identity_before=pending["live_before"],
                                       live_identity_after=pending["live_after"],
                                       max_adjacent_gap_ns=self.max_gap_ns)
            _require(result["status"] == "OWN_INITIALIZATION_VISIBILITY_RECONCILED" and
                     result["resource_pass"] is False, result["status"])
            intervals.append(result["transition_feasible_interval_ns"])
            results.append({"raw_ref": pending["raw_ref"], "visibility_subgate": result})
        if intervals:
            _require(max(x[0] for x in intervals) <= min(x[1] for x in intervals),
                     "UNKNOWN_MULTIPLE_CONTEXT_TRANSITIONS")
            self._persist_derived(child.tag + ".INIT_RECONCILIATION", {
                "status": "OWN_INITIALIZATION_VISIBILITY_EXPLANATION_ONLY",
                "previous_ref": child.previous_ref, "following_ref": following_ref,
                "pending": results, "resource_pass": False})
        child.pending.clear()

    @_latched
    def observe(self, child):
        with self.lock, child.lock:
            self._guard()
            _require(self.active is child and not child.closed, "UNKNOWN_ACTIVE_CHILD")
            record, ref = self.capture(child)
            try:
                value = self._safety(record, child)
                visibility = value["visibility"]
                status, observations = inspect_sample(visibility, child.owner)
                _require(status == "VISIBILITY_OBSERVED", status)
                latest = child.pending[-1]["visibility"] if child.pending else child.previous
                previous_end = max(q["end_ns"] for q in latest["queries"])
                current_begin = min(q["begin_ns"] for q in visibility["queries"])
                _require(0 <= current_begin - previous_end <= self.max_gap_ns,
                         "UNKNOWN_UNBOUNDED_MONITORING_GAP")
                with DECISION_LOCK:
                    self._guard()
                    present = [row[2] for row in observations]
                    life = record["lifecycle_after"]["parent_wait_observation"]["status"]
                    if child.phase == "INITIALIZING":
                        _require(self.monotonic_ns() - child.spawn_monotonic_ns <= self.initialization_ns,
                                 "UNKNOWN_INITIALIZATION_BOUND_EXCEEDED")
                        if all(present):
                            self._resolve_initialization(child, visibility, ref)
                            child.phase = "RUNNING"
                        elif not any(present):
                            _require(not child.pending, "UNKNOWN_INITIALIZATION_VISIBILITY_REVERSED")
                            _require(value["signed_unattributed_residual_bytes"] <= UNATTRIBUTED,
                                     "UNEXPLAINED_MATERIAL_MEMORY_WITHOUT_COMPUTE")
                            child.previous, child.previous_ref = visibility, ref
                            return {"status": "INITIALIZING_NO_CONTEXT_NOT_RESOURCE_PASS",
                                    "resource_pass": False, "raw_ref": ref, "lifecycle": life}
                        else:
                            child.pending.append({"visibility": visibility, "raw_ref": ref,
                                "live_before": record["lifecycle_before"]["parent_wait_observation"]["status"] == "LIVE_UNREAPED" and
                                               _identity_equal(record["lifecycle_before"]["actual_proc_identity"], child.owner),
                                "live_after": life == "LIVE_UNREAPED" and
                                              _identity_equal(record["lifecycle_after"]["actual_proc_identity"], child.owner)})
                            _require(len(child.pending) <= 2, "UNKNOWN_BOUNDARY_PENDING_LIMIT")
                            return {"status": "DEFER_OWN_INITIALIZATION_NOT_RESOURCE_PASS",
                                    "resource_pass": False, "raw_ref": ref, "lifecycle": life}
                    if child.phase in ("RUNNING", "EXIT_PENDING") and not all(present):
                        child.phase = "EXIT_PENDING"
                        child.pending.append({"visibility": visibility, "raw_ref": ref,
                                              "mixed": any(present), "record": record})
                        _require(len(child.pending) <= 2, "UNKNOWN_BOUNDARY_PENDING_LIMIT")
                        return {"status": "DEFER_OWN_EXIT_NOT_RESOURCE_PASS", "resource_pass": False,
                                "raw_ref": ref, "lifecycle": life}
                    _require(not child.pending, "UNKNOWN_EXIT_VISIBILITY_REAPPEARED")
                    child.previous, child.previous_ref = visibility, ref
                    if life == "EXITED_UNREAPED":
                        return {"status": "NATURAL_EXIT_OBSERVED_NOT_RESOURCE_PASS",
                                "resource_pass": False, "raw_ref": ref, "lifecycle": life}
                    value.update(status="RUNNING_RESOURCE_PASS", resource_pass=True, raw_ref=ref)
                    return value
            except Exception as error:
                self.fail(error, evidence_ref=ref)

    @_latched
    def finish(self, child, validate_result):
        with self.lock, child.lock:
            self._guard()
            _require(self.active is child and not child.closed, "UNKNOWN_ACTIVE_CHILD")
            observation_begin = self.monotonic_ns()
            waited = self.exit_observer(child.popen)
            self.latest_evidence_ref = self._persist_derived(child.tag + ".FINISH_WAIT_RAW", {
                "parent_wait_observation": waited, "observed_begin_monotonic_ns": observation_begin,
                "observed_end_monotonic_ns": self.monotonic_ns(),
                "scope": "actual WNOWAIT observation before exit classification; not wait/reap or resource PASS"})
            _require(waited.get("status") == "EXITED_UNREAPED" and waited.get("pid") == child.owner.pid,
                     "FINISH_REQUIRES_NATURAL_UNREAPED_EXIT")
            try:
                last_visibility = child.pending[-1]["visibility"] if child.pending else child.previous
                last_query_end = max(query["end_ns"] for query in last_visibility["queries"])
                _require(0 <= self.monotonic_ns() - last_query_end <= self.max_gap_ns,
                         "UNKNOWN_UNBOUNDED_EXIT_OBSERVATION_GAP")
                _require(type(waited.get("returncode")) is int and waited["returncode"] == 0,
                         "NONZERO_OWN_CHILD_EXIT")
                wait_begin = self.monotonic_ns()
                rc = child.popen.wait()
                wait_end = self.monotonic_ns()
                _require(type(rc) is int and rc == 0, "LEGAL_PARENT_WAIT_NONZERO")
                result = validate_result(child)
                _require(isinstance(result, dict) and result.get("status") == "PASS" and
                         all(result.get(key) is True for key in
                             ("source_verified", "numerical_pass", "profile_verified", "environment_verified")) and
                         result.get("gpu_uuid") == self.uuid, "ACTUAL_RESULT_BINDING_NOT_PASS")
                refs = result.get("evidence_refs")
                _require(isinstance(refs, list) and len(refs) > 0, "RESULT_EVIDENCE_REFS_REQUIRED")
                for item in refs:
                    _require(Path(item["path"]).is_absolute() and _sha(item["path"]) == item["sha256"],
                             "RESULT_EVIDENCE_SHA_CHANGED")
                _require(child.phase != "INITIALIZING", "INITIALIZATION_NEVER_STABLY_VISIBLE")
                terminal, terminal_ref, _ = self._window(child.tag, "TERMINAL")
                following = terminal["values"][0]["visibility"]
                explanations, intervals = [], []
                for pending in child.pending:
                    if pending.get("mixed"):
                        resolved = reconcile_triplet(child.previous, pending["visibility"], following,
                            child.owner, "EXITED", child.spawn_monotonic_ns, wait_end,
                            legal_wait_rc=rc, terminal_window_pass=terminal["status"] == "PASS",
                            actual_result_valid=result["status"] == "PASS", max_adjacent_gap_ns=self.max_gap_ns)
                        _require(resolved["status"] == "OWN_NATURAL_EXIT_VISIBILITY_RECONCILED" and
                                 resolved["resource_pass"] is False, resolved["status"])
                        intervals.append(resolved["transition_feasible_interval_ns"])
                    else:
                        resolved = {"status": "OWN_CONTEXT_ABSENT_AFTER_VERIFIED_NATURAL_EXIT",
                                    "resource_pass": False}
                    explanations.append({"raw_ref": pending["raw_ref"], "resolution": resolved})
                if intervals:
                    _require(max(x[0] for x in intervals) <= min(x[1] for x in intervals),
                             "UNKNOWN_MULTIPLE_EXIT_CONTEXT_TRANSITIONS")
                # Include *all* pending samples, including all-empty ones.
                # Skipping an early all-empty sample could hide a later context
                # reappearance while one mixed sample looked individually valid.
                if child.pending:
                    all_queries = [query for view in
                        [child.previous] + [item["visibility"] for item in child.pending] + [following]
                        for query in view["queries"]]
                    present_begin = [query["begin_ns"] for query in all_queries if query["compute_rows"]]
                    absent_end = [query["end_ns"] for query in all_queries if not query["compute_rows"]]
                    _require(present_begin and absent_end and max(present_begin) <= min(absent_end),
                             "UNKNOWN_EXIT_CONTEXT_REAPPEARED_ACROSS_PENDING_SAMPLES")
                self._guard()
                end = self.monotonic_ns()
                charged = (end - child.spawn_monotonic_ns) / 1e9
                job = {"tag": child.tag, "status": "PASS", "gpu_index": self.index,
                       "gpu_uuid": self.uuid, "owner": child.owner.__dict__, "returncode": rc,
                       "admission_ref": child.admission["ref"], "terminal_ref": terminal_ref,
                       "result_verification": result, "exit_visibility_explanations": explanations,
                       "wait_begin_monotonic_ns": wait_begin, "wait_end_monotonic_ns": wait_end,
                       "spawn_monotonic_ns": child.spawn_monotonic_ns, "spawn_wall_ns": child.spawn_wall_ns,
                       "charged_seconds": charged, "charged_gpu_hours": charged / 3600,
                       "pure_spawn_to_wait_seconds": (wait_end - child.spawn_monotonic_ns) / 1e9,
                       "charged_scope": "spawn through completed strict terminal CPU gate, per physical GPU",
                       "gpu_time_caps": None, "NVML_peak_scope": "periodic raw samples only; not CUDA peak",
                       "NVML_sampled_owned_peak_bytes": child.nvml_owned_peak_bytes,
                       "NVML_owned_peak_sample_count": child.nvml_owned_peak_samples,
                       "NVML_owned_peak_unknown_sample_count": child.nvml_owned_peak_unknown_samples,
                       "CUDA_peak": None}
                with DECISION_LOCK:
                    self._guard()
                    completion_ref = self._persist_derived(child.tag + ".EXIT", job)
                    child.closed, child.pending = True, []
                    self.jobs.append(job)
                    self.active = None
                return {"status": "PASS", "completion_ref": completion_ref, "job": job}
            except Exception as error:
                if child.popen.returncode is not None:
                    self._record_failed_reaped(child, str(error))
                self.fail(error, details={"tag": child.tag, "wait_observation": waited})

    def _record_failed_reaped(self, child, reason):
        """No successful terminal gate is invented for failed/reaped work."""
        job = {"tag": child.tag, "status": "FAILED_NOT_QUALIFIED", "gpu_index": self.index,
               "gpu_uuid": self.uuid, "returncode": child.popen.returncode,
               "charged_seconds": (self.monotonic_ns() - child.spawn_monotonic_ns) / 1e9,
               "charged_scope": "spawn through failure cleanup observation; terminal PASS absent",
               "gpu_time_caps": None, "reason": reason, "resource_pass": False,
               "actual_stop_cursor": None, "actual_stop_cursor_status": "UNKNOWN"}
        job["charged_gpu_hours"] = job["charged_seconds"] / 3600
        ref = self._persist_derived(child.tag + ".FAILED_EXIT", job)
        child.closed, child.pending = True, []
        self.jobs.append(job)
        self.active = None
        return {"status": "EXACT_OWN_CHILD_REAPED_AFTER_FAILURE", "job": job, "completion_ref": ref}

    def accounting(self):
        """Current task/lane only: root adds its historical prior exactly once."""
        with self.lock:
            completed = sum(job["charged_seconds"] for job in self.jobs)
            active_seconds = 0 if self.active is None else (self.monotonic_ns() - self.active.spawn_monotonic_ns) / 1e9
            return {"completed_charged_seconds": completed, "active_charged_seconds": active_seconds,
                    "current_gpu_hours": (completed + active_seconds) / 3600,
                    "prior_gpu_hours": 0, "gpu_time_caps": None,
                    "active_worker": None if self.active is None else {
                        "pid": self.active.owner.pid, "gpu_uuid": self.uuid, "tag": self.active.tag},
                    "scope": "this physical GPU lane; root combines lanes and adds prior once"}

    @_latched
    def terminate_exact(self, child, *, term_timeout_seconds=30, kill_timeout_seconds=30):
        """Failure cleanup only. Never signal an unverified/reused/unknown PID."""
        with self.lock, child.lock:
            _require(self.failed, "CLEANUP_REQUIRES_PERSISTED_FIRST_FAILURE")
            _require(type(term_timeout_seconds) in (int, float) and 0 < term_timeout_seconds <= 60 and
                     type(kill_timeout_seconds) in (int, float) and 0 < kill_timeout_seconds <= 30,
                     "BOUNDED_CLEANUP_TIMEOUT_REQUIRED")
            _require(self.active is child and not child.closed, "UNKNOWN_CLEANUP_CHILD")
            waited = self.exit_observer(child.popen)
            _require(waited.get("pid") == child.owner.pid and
                     waited.get("status") in ("LIVE_UNREAPED", "EXITED_UNREAPED"), "UNKNOWN_CLEANUP_LIFETIME")
            if waited["status"] == "LIVE_UNREAPED":
                actual = self.proc_reader(child.owner.pid)
                _require(_identity_equal(actual, child.owner) and actual.get("ppid") == os.getpid(),
                         "UNKNOWN_CLEANUP_IDENTITY_NO_SIGNAL")
                # Parent-owned unreaped lifetime reserves the PID through signal.
                os.kill(child.owner.pid, signal.SIGTERM)
            try:
                child.popen.wait(timeout=term_timeout_seconds)
            except subprocess.TimeoutExpired:
                # Never send the second signal based only on the first check.
                again = self.exit_observer(child.popen)
                _require(again.get("pid") == child.owner.pid and
                         again.get("status") in ("LIVE_UNREAPED", "EXITED_UNREAPED"),
                         "UNKNOWN_CLEANUP_LIFETIME_AFTER_TERM_NO_KILL")
                if again["status"] == "LIVE_UNREAPED":
                    actual = self.proc_reader(child.owner.pid)
                    _require(_identity_equal(actual, child.owner) and actual.get("ppid") == os.getpid(),
                             "UNKNOWN_CLEANUP_IDENTITY_AFTER_TERM_NO_KILL")
                    os.kill(child.owner.pid, signal.SIGKILL)
                try:
                    child.popen.wait(timeout=kill_timeout_seconds)
                except subprocess.TimeoutExpired:
                    unknown = {"status": "OWN_CHILD_RELEASE_UNKNOWN_REQUIRES_USER",
                               "gpu_uuid": self.uuid, "tag": child.tag, "owner": child.owner.__dict__,
                               "resource_pass": False, "worker_reaped": False,
                               "charged_seconds_lower_bound": (self.monotonic_ns() - child.spawn_monotonic_ns) / 1e9,
                               "actual_stop_cursor": None, "active_registration_retained": True}
                    self._persist_derived(child.tag + ".CLEANUP_UNKNOWN", unknown)
                    return unknown
            return self._record_failed_reaped(child, "global first failure cleanup")
