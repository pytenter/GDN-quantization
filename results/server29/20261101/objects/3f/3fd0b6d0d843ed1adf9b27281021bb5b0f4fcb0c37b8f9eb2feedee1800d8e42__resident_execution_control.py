"""Independent stdlib control for the human-authorized resident continuation.

No Torch import, model creation, GPU dispatch, or remote signalling on import.
The qualification permit is immutable; a separate one-shot formal permit is
created only after every real qualification process has exited and released.
"""
from pathlib import Path
import fcntl
import hashlib
import importlib
import json
import os
import signal
import subprocess
import sys
import threading
import time

TASK = "QWEN_GDN_OSCAR_RESIDENT_ROTATION_EFFICIENT_QUALIFICATION_V1"
ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/data/ydai/miniconda3/envs/bitdecode/bin/python"
UUIDS = {1: "GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3",
         2: "GPU-b7ed673d-18ba-39ca-d7bc-c846af4f6b31",
         3: "GPU-30b07aba-a728-7649-4899-e09d33f8bc75"}
_EVENT_LOCK = threading.RLock()
_LOCKS = []


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode() + b"\n"


def atomic_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical(value))
        handle.flush()
        os.fsync(handle.fileno())


def status(root, phase, **fields):
    root = Path(root)
    row = dict(status=phase, monotonic=time.monotonic())
    row.update(fields)
    path = root / "PHASE_STATUS.json"
    temp = path.with_name(path.name + ".tmp." + str(os.getpid()) + "." + str(threading.get_ident()))
    with temp.open("wb") as handle:
        handle.write(canonical(row)); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)
    return row


def event(path, event_type, **fields):
    row = dict(event=event_type, monotonic=time.monotonic())
    row.update(fields)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _EVENT_LOCK, path.open("ab") as handle:
        handle.write(canonical(row)); handle.flush(); os.fsync(handle.fileno())
    return row


def process_identity(pid):
    try:
        p = Path("/proc") / str(pid)
        raw = (p / "stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        if fields[0] in ("Z", "X", "x"):
            return None
        return dict(pid=int(pid), starttick=int(fields[19]),
                    cmdline=(p / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0"))
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(identity):
    return isinstance(identity, dict) and process_identity(identity["pid"]) == identity


def _guards(root=ROOT):
    root = Path(root).resolve()
    require(root == ROOT, "Control module cannot operate another execution root")
    sys.path.insert(0, str(root / "src"))
    return importlib.import_module("qualification_guards")


def query_gpu():
    return _guards().vendor(ROOT).query_gpu()


def verify_sources(root):
    require(sha(Path(root) / "src/resident_rotation.py") ==
            "4d505a51b1aeeb8235acfa6feee1caba38328d6ec654628559190b5e0a8f1f95",
            "Resident scientific adapter changed")
    return _guards(root).verify_sources(root)


def verify_model_and_external(root, fresh_external=True):
    return _guards(root).verify_model_and_external(root, fresh_external=fresh_external)


def check_after_model(root, gpu, pid):
    return _guards(root).check_after_model(root, gpu, pid)


def validate_prior_p1_stop(root):
    return {str(g): _guards(root).p1_queue_released(root, g) for g in UUIDS}


def stop_own(identity, root, reason):
    """Only this exact continuation child; never a PID group or foreign job."""
    root = Path(root).resolve()
    require(root == ROOT and identity is not None, "Missing own process identity")
    args = identity["cmdline"]
    require(str(root) in args and any(str(root / "scripts" / name) in args for name in
            ("qualify_gpu.py", "resident_formal_worker.py")), "Refuse to signal a foreign process")
    sent = []
    for sig, duration in ((signal.SIGTERM, 15), (signal.SIGKILL, 15)):
        if not same_process(identity):
            break
        # Actual Linux3.10 has no pidfd. Recheck exact starttick and argv right
        # before each individual PID syscall; never claim atomic pidfd safety.
        require(same_process(identity), "Own identity changed before signal")
        os.kill(identity["pid"], sig)
        sent.append(dict(signal=signal.Signals(sig).name, monotonic=time.monotonic()))
        deadline = time.monotonic() + duration
        while same_process(identity) and time.monotonic() < deadline:
            time.sleep(.2)
    result = dict(identity=identity, reason=reason, signals=sent,
                  exact_identity_exited=not same_process(identity),
                  atomic_pidfd_signal_claimed=False, monotonic=time.monotonic())
    event(root / "GPU_RESOURCE_LEDGER.jsonl", "OWN_CHILD_STOP_OBSERVATION", **result)
    return result


def _interval_ok(root, target, identity):
    interval = read(target / "RESOURCE_INTERVAL.json")
    require(interval["identity"] == identity and interval["exit_returncode"] == 0
            and interval["release"] == "PASS" and not same_process(identity),
            "Qualification/formal process exit or release is not genuine PASS")
    for app in query_gpu()["apps"]:
        if int(app["pid"]) == identity["pid"]:
            current = process_identity(identity["pid"])
            require(current is not None and current != identity,
                    "Historical own PID still appears in NVML without a distinct reused identity")
    return interval


def validate_qualification_result(root, unit):
    root = Path(root)
    target = root / "qualification" / unit["id"]
    started = read(target / "STARTED.json")
    identity = started["identity"]
    require(not (target / "FIRST_FAILURE.json").exists(), "Unit has a real first failure")
    result = read(target / "RESULT.json")
    require(result["status"] == "PASS" and result["unit"] == unit
            and result["pid"] == identity["pid"]
            and result["source_freeze_sha256"] == sha(root / "SOURCE_FREEZE.json")
            and result["permit_sha256"] == sha(root / "GPU_PERMIT.json"),
            "Qualification result is not bound to the exact manifest/source/process")
    require(started["source_freeze_sha256"] == result["source_freeze_sha256"]
            and started["permit_sha256"] == result["permit_sha256"], "STARTED source/permit differs")
    if unit.get("repeat") == 1 and unit.get("reference_unit"):
        require(read(target / "REFERENCE_RELEASED.json")["status"] == "PASS",
                "Fresh-pair reference was not released after durable comparison PASS")
    return result


def validate_formal_result(root, job):
    root = Path(root)
    target = root / "formal" / job["id"]
    require(not (target / "FIRST_FAILURE.json").exists(), "Formal unit has a real first failure")
    wrapper = read(target / "RESULT.json")
    from resident_record_validation import validate
    validate(wrapper, job, root)
    started = read(target / "STARTED.json")
    require(wrapper["identity"] == started["identity"], "Formal result differs from actual STARTED identity")
    return wrapper


def performance_audit(root, units, results):
    from resident_performance_admission import audit
    return audit(root, units, results)


def authorize_formal(root, qualification_receipts):
    root = Path(root)
    require(not (root / "FIRST_FAILURE.json").exists(), "A real failure blocks formal admission")
    require(not (root / "AUTHORIZATION_FORMAL.json").exists()
            and not (root / "FORMAL_GPU_PERMIT.json").exists(), "Formal authorization already attempted")
    source_sha = verify_sources(root)
    verify_model_and_external(root)
    validate_prior_p1_stop(root)
    units = read(root / "QUALIFICATION_JOB_MANIFEST.json")["units"]
    require(len(units) == 58 and len({x["id"] for x in units}) == 58,
            "Expected exactly58 unique frozen qualification units")
    names = {x["unit"] for x in qualification_receipts}
    require(names == {u["id"] for u in units} and len(qualification_receipts) == 58,
            "Formal admission lacks the exact58 controller receipts")
    audited, actual_results = [], {}
    for unit in units:
        target = root / "qualification" / unit["id"]
        wrapper = validate_qualification_result(root, unit)
        actual_results[unit["id"]] = wrapper
        identity = read(target / "STARTED.json")["identity"]
        _interval_ok(root, target, identity)
        audited.append(dict(unit=unit["id"], identity=identity,
                            result_sha256=sha(target / "RESULT.json"),
                            interval_sha256=sha(target / "RESOURCE_INTERVAL.json")))
        if unit["kind"].startswith("perf_"):
            require(wrapper["result"]["TPS"] > 0, "Measured performance TPS missing or nonpositive")
    obs = query_gpu()
    require(not any(x["uuid"] in UUIDS.values() for x in obs["apps"]),
            "Authorized UUIDs must be idle after all qualification workers exit")
    jobs = read(root / "FORMAL_JOB_MANIFEST.json")["jobs"]
    require(len(jobs) == 60 and len({j["id"] for j in jobs}) == 60
            and {j["condition"] for j in jobs} == {"C1", "S3"}, "Frozen formal roster differs")
    grant = read(root / "HUMAN_CONTINUATION_AUTHORIZATION.json")
    require(grant["formal_conditionally_authorized"] is True, "Human conditional formal scope absent")
    base = read(root / "GPU_PERMIT.json")
    require(base["ALLOW_GPU"] == "QUALIFICATION_ONLY" and base["source_freeze_sha256"] == source_sha,
            "Base qualification permit differs")
    performance = performance_audit(root, units, actual_results)
    atomic_new(root / "PERFORMANCE_ADMISSION.json", performance)
    qualification = dict(status="PASS_ALL58_REAL_UNITS_EXIT0_AND_RELEASED",
                         source_freeze_sha256=source_sha, units=audited,
                         observation=obs, monotonic=time.monotonic())
    atomic_new(root / "GPU_QUALIFICATION_FINAL.json", qualification)
    auth = dict(status="ALLOW_FIXED60_FORMAL_ONCE_AFTER_ALL58_REAL_GATES_PASS",
                task_id=TASK, execution_attempt=root.name, source_freeze_sha256=source_sha,
                base_permit_sha256=sha(root / "GPU_PERMIT.json"),
                qualification_sha256=sha(root / "GPU_QUALIFICATION_FINAL.json"),
                performance_admission_sha256=sha(root / "PERFORMANCE_ADMISSION.json"),
                jobs_sha256=sha(root / "FORMAL_JOB_MANIFEST.json"),
                human_authorization_sha256=sha(root / "HUMAN_CONTINUATION_AUTHORIZATION.json"),
                planned=60, devices=list(UUIDS.values()), monotonic=time.monotonic())
    atomic_new(root / "AUTHORIZATION_FORMAL.json", auth)
    atomic_new(root / "FORMAL_GPU_PERMIT.json", dict(task_id=TASK, ALLOW_GPU="FORMAL_ONLY",
                source_freeze_sha256=source_sha, authorization_sha256=sha(root / "AUTHORIZATION_FORMAL.json"),
                devices=list(UUIDS.values()), jobs_sha256=auth["jobs_sha256"],
                human_authorization_sha256=auth["human_authorization_sha256"]))
    return auth


def admit_formal(root, job_id, gpu, pid):
    root = Path(root).resolve()
    require(root == ROOT and pid == os.getpid() and gpu in UUIDS, "Invalid formal root/GPU/current PID")
    permit = read(root / "FORMAL_GPU_PERMIT.json")
    require(permit["task_id"] == TASK and permit["ALLOW_GPU"] == "FORMAL_ONLY",
            "No formal permission; do not import Torch")
    source = verify_sources(root)
    require(permit["source_freeze_sha256"] == source
            and permit["authorization_sha256"] == sha(root / "AUTHORIZATION_FORMAL.json")
            and permit["jobs_sha256"] == sha(root / "FORMAL_JOB_MANIFEST.json"),
            "Formal permit/source/roster binding differs")
    auth = read(root / "AUTHORIZATION_FORMAL.json")
    performance = read(root / "PERFORMANCE_ADMISSION.json")
    qualification = read(root / "GPU_QUALIFICATION_FINAL.json")
    require(auth["planned"] == 60 and auth["source_freeze_sha256"] == source
            and auth["performance_admission_sha256"] == sha(root / "PERFORMANCE_ADMISSION.json")
            and performance["status"] == "PASS_OBJECTIVE_EFFICIENT_RESIDENT_EXECUTION_AND_ACTUAL_COST_DISCLOSED"
            and auth["qualification_sha256"] == sha(root / "GPU_QUALIFICATION_FINAL.json")
            and qualification["status"] == "PASS_ALL58_REAL_UNITS_EXIT0_AND_RELEASED"
            and qualification["source_freeze_sha256"] == source
            and len(qualification["units"]) == len({x["unit"] for x in qualification["units"]}) == 58
            and auth["jobs_sha256"] == permit["jobs_sha256"]
            and auth["base_permit_sha256"] == sha(root / "GPU_PERMIT.json")
            and auth["human_authorization_sha256"] == permit["human_authorization_sha256"]
            == sha(root / "HUMAN_CONTINUATION_AUTHORIZATION.json"),
            "Full qualification authorization differs")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == UUIDS[gpu], "Formal CVD must be actual physical UUID")
    jobs = read(root / "FORMAL_JOB_MANIFEST.json")["jobs"]
    selected = [x for x in jobs if x["id"] == job_id]
    require(len(selected) == 1, "Formal job identity not unique")
    job = selected[0]
    require(job["gpu"] == gpu and job["gpu_uuid"] == UUIDS[gpu]
            and job["condition"] in ("C1", "S3") and job["seed"] == 1
            and job["max_new_tokens"] == 81920, "Formal immutable job protocol differs")
    target = root / "formal" / job_id
    require(target.is_dir() and not (target / "STARTED.json").exists(), "Formal already STARTED or never dispatched")
    dispatch = read(target / "DISPATCH.json")
    parent = process_identity(os.getppid())
    expected_parent = [PYTHON, "-u", "-B", str(root / "scripts/resident_controller.py"),
                       "--root", str(root), "--execute"]
    # The offline shell invokes the controller using this exact frozen argv.
    require(parent == dispatch["controller_identity"] and same_process(parent)
            and parent["cmdline"] == expected_parent
            and dispatch["unit"] == job_id and dispatch["gpu_uuid"] == UUIDS[gpu]
            and dispatch["gpu"] == gpu and dispatch["phase"] == "formal"
            and dispatch["unit_record"] == job
            and dispatch["source_freeze_sha256"] == source
            and dispatch["permit_sha256"] == sha(root / "FORMAL_GPU_PERMIT.json")
            and dispatch["base_gpu_permit_sha256"] == sha(root / "GPU_PERMIT.json")
            and dispatch["formal_authorization_sha256"] == sha(root / "AUTHORIZATION_FORMAL.json")
            and dispatch["plan_sha256"] == sha(root / "FORMAL_JOB_MANIFEST.json"),
            "Formal dispatcher parent/identity/source differs")
    validate_prior_p1_stop(root)
    rows = verify_model_and_external(root)
    guard = _guards(root)
    guard._own_gpu(root, gpu)
    space = os.statvfs("/data")
    require(space.f_bavail * space.f_frsize >= 8 * 1024**3, "Data free headroom below8GiB")
    lock = (root / "operations" / ("GPU_" + str(gpu) + ".lock")).open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); _LOCKS.append(lock)
    guard._own_gpu(root, gpu)
    driver = guard.driver_identity(root, gpu, rows)
    identity = process_identity(pid)
    admission = dict(job=job_id, identity=identity, source_freeze_sha256=source,
                     base_permit_sha256=sha(root / "GPU_PERMIT.json"),
                     formal_permit_sha256=sha(root / "FORMAL_GPU_PERMIT.json"),
                     authorization_sha256=sha(root / "AUTHORIZATION_FORMAL.json"),
                     gpu=gpu, gpu_uuid=UUIDS[gpu], driver=driver, monotonic=time.monotonic())
    atomic_new(target / "ADMISSION.json", admission)
    return dict(job=job, target=target, identity=identity, admission=admission)
