#!/usr/bin/env python3
"""Stdlib-only isolated CPU control tests. Never imports Torch or queries GPUs.

Actual production controller is run in a fresh subprocess with a temporary fake
control module/fake child programs; production has no test mode or injection
switch. Admission tests call actual production control functions, replacing only
external source/model/P1/NVML/process observations inside the test process. All
manifests, permits, results and identities are explicitly synthetic fixtures.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CHECKS = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def get(path):
    return json.loads(Path(path).read_text())


def check(name, ok, **evidence):
    if not ok:
        raise AssertionError(name)
    CHECKS.append(dict(name=name, status="PASS", **evidence))


def rejects(name, function):
    try:
        function()
    except (RuntimeError, KeyError, FileNotFoundError, FileExistsError) as error:
        check(name, True, rejection=str(error))
        return
    raise AssertionError("Expected rejection: " + name)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FAKE_CONTROL = r'''"""CPU fixture only, installed solely in temporary test roots."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
TASK="SYNTHETIC_CPU_CONTROL_TEST_ONLY"
ROOT=Path(__file__).resolve().parents[1]
PYTHON=sys.executable
UUIDS={1:"CPU_FIXTURE_UUID_1",2:"CPU_FIXTURE_UUID_2",3:"CPU_FIXTURE_UUID_3"}
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_new(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("x")as f:json.dump(value,f,sort_keys=True);f.write("\n")
def event(p,kind,**fields):
 row=dict(event=kind,monotonic=time.monotonic());row.update(fields)
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("a")as f:f.write(json.dumps(row,sort_keys=True)+"\n")
 return row
def status(root,phase,**fields):
 row=dict(status=phase,monotonic=time.monotonic());row.update(fields)
 (Path(root)/"PHASE_STATUS.json").write_text(json.dumps(row));return row
def process_identity(pid):
 try:
  if Path("/proc").is_dir():
   p=Path("/proc")/str(pid);s=(p/"stat").read_text();fields=s[s.rfind(")")+2:].split()
   if fields[0]in("Z","X","x"):return None
   return dict(pid=pid,starttick=int(fields[19]),cmdline=(p/"cmdline").read_bytes().rstrip(b"\0").decode().split("\0"))
  value=subprocess.run(["ps","-p",str(pid),"-o","stat=","-o","lstart=","-o","command="],capture_output=True,text=True).stdout.strip()
  if not value or value.split()[0].startswith("Z"):return None
  parts=value.split(None,6)
  args=parts[6].split()
  # macOS Python launcher execs its framework binary after Popen returns.
  # This fake-observer-only normalization retains actual script/root/args;
  # production Linux /proc PID/starttick/argv is never normalized or changed.
  if args and "python"in Path(args[0]).name.lower():args[0]="CPU_FIXTURE_PYTHON_EXECUTABLE"
  return dict(pid=pid,starttick="CPU_FIXTURE_ps_lstart_seconds:"+" ".join(parts[1:6]),cmdline=args)
 except(FileNotFoundError,ProcessLookupError,IndexError):return None
def same_process(identity):return isinstance(identity,dict)and process_identity(identity["pid"])==identity
def query_gpu():
 foreign=read(ROOT/"FAKE_FOREIGN.json")
 return dict(status="CPU_FAKE_NVML_ONLY",apps=[dict(pid=foreign["pid"],uuid="CPU_FOREIGN_UUID")])
def verify_sources(root):return sha(Path(root)/"SOURCE_FREEZE.json")
def verify_model_and_external(root):return dict(status="SYNTHETIC_NO_MODEL_OR_EXTERNAL_ACCESS")
def validate_prior_p1_stop(root):return dict(status="SYNTHETIC_NO_REAL_P1_ACCESS")
def validate_qualification_result(root,unit):
 target=Path(root)/"qualification"/unit["id"];row=read(target/"RESULT.json")
 if row["status"]!="PASS"or row["unit"]!=unit:raise RuntimeError("Synthetic result mismatch")
 return row
def validate_formal_result(root,job):raise RuntimeError("No formal allowed in failure fixture")
def authorize_formal(root,receipts):
 atomic_new(Path(root)/"UNEXPECTED_FORMAL_CALL.json",dict(receipts=receipts))
 raise RuntimeError("CPU failure fixture must never admit formal")
def stop_own(identity,root,reason):
 if str(Path(root))not in identity["cmdline"]or str(Path(root)/"scripts/qualify_gpu.py")not in identity["cmdline"]:
  raise RuntimeError("Foreign identity rejected")
 if same_process(identity):os.kill(identity["pid"],signal.SIGTERM)
 event(Path(root)/"STOP_TARGETS.jsonl","CPU_TEST_OWN_SIGNAL",identity=identity)
 return dict(identity=identity,synthetic=True)
'''

FAKE_CHILD = r'''import argparse,json,os,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--root",type=Path);p.add_argument("--unit");p.add_argument("--gpu",type=int);a=p.parse_args()
sys.path.insert(0,str(a.root/"src"));import resident_execution_control as C
time.sleep(.25)
u=next(x for x in C.read(a.root/"QUALIFICATION_JOB_MANIFEST.json")["units"]if x["id"]==a.unit)
t=a.root/"qualification"/a.unit;t.mkdir(exist_ok=False)
C.atomic_new(t/"STARTED.json",dict(identity=C.process_identity(os.getpid()),synthetic=True))
C.atomic_new(t/"PROGRESS.json",dict(phase="SYNTHETIC_CPU_ONLY",token=1))
if a.unit=="GPU2_FAIL":
 time.sleep(.5);C.atomic_new(t/"FIRST_FAILURE.json",dict(status="SYNTHETIC_EXPECTED_FAILURE"));raise SystemExit(7)
if "HOLD"in a.unit:time.sleep(45)
C.atomic_new(t/"RESULT.json",dict(status="PASS",unit=u,result=dict(synthetic=True)))
'''


def synthetic_rosters():
    units = [dict(id="GPU1_CODEC", gpu=1, kind="codec"),
             dict(id="GPU1_FP128", gpu=1, kind="fp128"),
             dict(id="GPU1_HOLD", gpu=1, kind="teacher"),
             dict(id="GPU2_FAIL", gpu=2, kind="teacher"),
             dict(id="GPU3_HOLD", gpu=3, kind="teacher")]
    units += [dict(id="SYNTHETIC_UNUSED_%02d" % i, gpu=1, kind="teacher") for i in range(53)]
    for u in units:
        u["gpu_uuid"] = "CPU_FIXTURE_UUID_%s" % u["gpu"]
    jobs = [dict(id="SYNTHETIC_FORMAL_%02d" % i, condition=("C1" if i % 2 == 0 else "S3"),
                 gpu=1, gpu_uuid="CPU_FIXTURE_UUID_1") for i in range(60)]
    return units, jobs


def subprocess_failure(directory):
    r = directory / "actual_controller_subprocess"
    (r / "scripts").mkdir(parents=True)
    (r / "src").mkdir()
    shutil.copy2(ROOT / "scripts/resident_controller.py", r / "scripts/resident_controller.py")
    (r / "src/resident_execution_control.py").write_text(FAKE_CONTROL)
    (r / "scripts/qualify_gpu.py").write_text(FAKE_CHILD)
    (r / "scripts/resident_formal_worker.py").write_text("raise RuntimeError('Formal must not run')\n")
    units, jobs = synthetic_rosters()
    put(r / "QUALIFICATION_JOB_MANIFEST.json", dict(units=units))
    put(r / "FORMAL_JOB_MANIFEST.json", dict(jobs=jobs))
    put(r / "SOURCE_FREEZE.json", dict(synthetic=True, never_scientific_evidence=True))
    put(r / "GPU_PERMIT.json", dict(synthetic=True))
    foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    put(r / "FAKE_FOREIGN.json", dict(pid=foreign.pid, never_signal=True))
    command = [sys.executable, "-B", str(r / "scripts/resident_controller.py"), "--root", str(r), "--execute"]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=45)
        (r / "SUBPROCESS_STDOUT.txt").write_text(proc.stdout)
        (r / "SUBPROCESS_STDERR.txt").write_text(proc.stderr)
        check("actual_controller_subprocess_expected_failure_exit", proc.returncode == 1, returncode=proc.returncode)
        session = get(r / "SESSION_RESULT.json")
        check("failure_path_persists_actual_SESSION", session["status"] == "FAIL_CLOSED_STOP")
        check("failure_does_not_admit_or_dispatch_formal", not (r / "AUTHORIZATION_FORMAL.json").exists()
              and not (r / "FORMAL_GPU_PERMIT.json").exists() and not any((r / "formal").iterdir())
              and not (r / "UNEXPECTED_FORMAL_CALL.json").exists())
        check("only_two_dependency_units_complete_before_real_child_failure", session["qualification_completed"] == 2)
        check("partial_count_zero_no_accuracy", get(r / "FORMAL_GENERATIONS_COMPLETE.json")["count"] == 0
              and get(r / "FORMAL_GENERATIONS_COMPLETE.json")["no_accuracy_claim"] is True)
        check("failure_all_exact_own_processes_released", session["own_release_all_PASS"] is True
              and session["all_own_processes_released"] is True)
        stops = [json.loads(line) for line in (r / "STOP_TARGETS.jsonl").read_text().splitlines()]
        check("both_actual_own_sibling_children_stopped", len(stops) == 2
              and all(x["identity"]["pid"] != foreign.pid for x in stops))
        check("foreign_process_survives_controller_failure", foreign.poll() is None)
        dispatches = sorted((r / "dispatch/qualification").glob("*.json"))
        check("failure_attempts_exactly_five_and_no_duplicate_dispatch", len(dispatches) == 5
              and session["attempted_units"] == 5 and len({get(p)["unit"] for p in dispatches}) == 5)
        intervals = [get(p) for p in (r / "qualification").glob("*/RESOURCE_INTERVAL.json")]
        check("all_five_exit_and_release_intervals_persist", len(intervals) == 5
              and all(row["release"] == "PASS" for row in intervals)
              and any(row["exit_returncode"] == 7 for row in intervals))
        check("same_host_monotonic_interval_bounds", all(row["dispatch_monotonic_lower"] <= row["identity_monotonic_upper"]
              <= row["exit_first_observed_monotonic"] <= row["release_first_observed_monotonic"] for row in intervals))
        preserved = {str(p.relative_to(r)): sha(p) for p in dispatches + [r / "CONTROLLER_STARTED.json", r / "SESSION_RESULT.json"]}
        second = subprocess.run(command, capture_output=True, text=True, timeout=10)
        check("controller_cannot_restart_after_started", second.returncode != 0
              and all(sha(r / p) == value for p, value in preserved.items()))
        return dict(path=str(r), returncode=proc.returncode, actual_child_count=5,
                    foreign_process_was_not_signalled=True, no_GPU_access=True)
    finally:
        # Test children only, exact source-root + worker path. This is teardown
        # for a failed test harness, never a production cleanup or GPU retry.
        fake = load("_fixture_teardown", r / "src/resident_execution_control.py")
        for identity_file in (r / "operations").glob("*_WORKER_IDENTITY.json"):
            identity = get(identity_file)["identity"]
            if fake.same_process(identity):
                fake.stop_own(identity, r.resolve(), "CPU fixture teardown only")
        foreign.terminate()
        foreign.wait(timeout=5)


def admission_fixture(r):
    units = [dict(id="CPU_UNIT_%02d" % i, gpu=1 + i % 3, gpu_uuid="CPU_FIXTURE_UUID_%s" % (1 + i % 3),
                  kind=("perf_late" if i == 57 else "teacher")) for i in range(58)]
    jobs = [dict(id="CPU_FORMAL_%02d" % i, condition="C1" if i % 2 == 0 else "S3") for i in range(60)]
    put(r / "SOURCE_FREEZE.json", dict(status="SYNTHETIC_NO_SCIENTIFIC_SOURCE"))
    source = sha(r / "SOURCE_FREEZE.json")
    put(r / "GPU_PERMIT.json", dict(ALLOW_GPU="QUALIFICATION_ONLY", source_freeze_sha256=source, synthetic=True))
    permit = sha(r / "GPU_PERMIT.json")
    put(r / "QUALIFICATION_JOB_MANIFEST.json", dict(units=units))
    put(r / "FORMAL_JOB_MANIFEST.json", dict(jobs=jobs))
    put(r / "HUMAN_CONTINUATION_AUTHORIZATION.json", dict(formal_conditionally_authorized=True, synthetic=True))
    for i, unit in enumerate(units):
        t = r / "qualification" / unit["id"]
        identity = dict(pid=800000 + i, starttick=100 + i, cmdline=["CPU_SYNTHETIC_FIXTURE_ONLY"])
        put(t / "STARTED.json", dict(identity=identity, source_freeze_sha256=source, permit_sha256=permit))
        put(t / "RESULT.json", dict(status="PASS", unit=unit, pid=identity["pid"],
            source_freeze_sha256=source, permit_sha256=permit, result=dict(TPS=12.5)))
        put(t / "RESOURCE_INTERVAL.json", dict(identity=identity, exit_returncode=0, release="PASS"))
    return units, [dict(unit=u["id"]) for u in units]


def actual_control_tests(directory):
    module_path = ROOT / "src/resident_execution_control.py"
    positive = directory / "admission_positive"
    units, receipts = admission_fixture(positive)
    module = load("_actual_resident_control_cpu_positive", module_path)
    # Test-local observation injection only; production module has no test API.
    module.ROOT = positive
    module.UUIDS = {i: "CPU_FIXTURE_UUID_%s" % i for i in (1, 2, 3)}
    module.verify_sources = lambda root: sha(root / "SOURCE_FREEZE.json")
    module.verify_model_and_external = lambda *a, **k: dict(synthetic=True)
    module.validate_prior_p1_stop = lambda root: dict(synthetic=True)
    module.performance_audit = lambda *a: dict(status="SYNTHETIC_PERFORMANCE_FIXTURE_ONLY")
    module.same_process = lambda identity: False
    module.query_gpu = lambda: dict(apps=[])
    auth = module.authorize_formal(positive, receipts)
    check("actual_authorize_formal_accepts_exact58_nested_result_TPS", auth["planned"] == 60
          and len(get(positive / "GPU_QUALIFICATION_FINAL.json")["units"]) == 58)
    check("formal_separate_permit_base_qualification_permit_unchanged",
          get(positive / "FORMAL_GPU_PERMIT.json")["ALLOW_GPU"] == "FORMAL_ONLY"
          and get(positive / "GPU_PERMIT.json")["ALLOW_GPU"] == "QUALIFICATION_ONLY")
    check("formal_authorization_binds_performance_admission",
          auth["performance_admission_sha256"] == sha(positive / "PERFORMANCE_ADMISSION.json"))
    rejects("actual_authorization_no_second_attempt", lambda: module.authorize_formal(positive, receipts))

    cases = [
        ("missing_58th_controller_receipt", lambda r, u, rows, m: rows.pop()),
        ("duplicate_manifest_unit", lambda r, u, rows, m: put(r / "QUALIFICATION_JOB_MANIFEST.json", dict(units=u[:-1] + [u[0]]))),
        ("persisted_global_failure", lambda r, u, rows, m: put(r / "FIRST_FAILURE.json", dict(status="SYNTHETIC_EXPECTED_FAILURE"))),
        ("wrong_RESULT_source_identity", lambda r, u, rows, m: mutate(r / "qualification" / u[0]["id"] / "RESULT.json", "source_freeze_sha256", "bad")),
        ("wrong_RESULT_permit_identity", lambda r, u, rows, m: mutate(r / "qualification" / u[0]["id"] / "RESULT.json", "permit_sha256", "bad")),
        ("wrong_RESULT_unit_record", lambda r, u, rows, m: mutate(r / "qualification" / u[0]["id"] / "RESULT.json", "unit", u[1])),
        ("RESULT_PID_differs_STARTED", lambda r, u, rows, m: mutate(r / "qualification" / u[0]["id"] / "RESULT.json", "pid", -1)),
        ("nonzero_process_exit", lambda r, u, rows, m: mutate(r / "qualification" / u[0]["id"] / "RESOURCE_INTERVAL.json", "exit_returncode", 7)),
        ("release_not_PASS", lambda r, u, rows, m: mutate(r / "qualification" / u[0]["id"] / "RESOURCE_INTERVAL.json", "release", "UNRESOLVED")),
        ("exact_own_process_still_live", lambda r, u, rows, m: setattr(m, "same_process", lambda identity: True)),
        ("selected_GPU_has_foreign_owner", lambda r, u, rows, m: setattr(m, "query_gpu", lambda: dict(apps=[dict(pid=999999, uuid="CPU_FIXTURE_UUID_2")]))),
        ("nonpositive_actual_TPS", lambda r, u, rows, m: mutate(r / "qualification" / u[-1]["id"] / "RESULT.json", "result", dict(TPS=0))),
        ("missing_actual_TPS", lambda r, u, rows, m: mutate(r / "qualification" / u[-1]["id"] / "RESULT.json", "result", dict(metrics=dict(TPS=1)))),
        ("formal_roster_missing_sample", lambda r, u, rows, m: mutate(r / "FORMAL_JOB_MANIFEST.json", "jobs", get(r / "FORMAL_JOB_MANIFEST.json")["jobs"][:-1])),
        ("human_formal_permission_false", lambda r, u, rows, m: mutate(r / "HUMAN_CONTINUATION_AUTHORIZATION.json", "formal_conditionally_authorized", False)),
        ("base_permit_not_qualification_only", lambda r, u, rows, m: mutate(r / "GPU_PERMIT.json", "ALLOW_GPU", "FORMAL_ONLY")),
        ("objective_performance_rejection", lambda r, u, rows, m: setattr(m, "performance_audit", lambda *a: (_ for _ in ()).throw(RuntimeError("SYNTHETIC_PERFORMANCE_REJECT")))),
    ]
    for index, (name, mutation) in enumerate(cases):
        r = directory / ("negative_%02d" % index)
        u, rows = admission_fixture(r)
        m = load("_actual_resident_control_cpu_negative_%d" % index, module_path)
        m.ROOT, m.UUIDS = r, module.UUIDS
        m.verify_sources = lambda root: sha(root / "SOURCE_FREEZE.json")
        m.verify_model_and_external = lambda *a, **k: dict(synthetic=True)
        m.validate_prior_p1_stop = lambda root: dict(synthetic=True)
        m.performance_audit = lambda *a: dict(status="SYNTHETIC_PERFORMANCE_FIXTURE_ONLY")
        m.same_process = lambda identity: False
        m.query_gpu = lambda: dict(apps=[])
        mutation(r, u, rows, m)
        rejects("actual_authorize_rejects_" + name, lambda: m.authorize_formal(r, rows))
        check("rejection_has_no_formal_permit_" + name, not (r / "FORMAL_GPU_PERMIT.json").exists())

    event_path = directory / "explicit_monotonic_events.jsonl"
    row = module.event(event_path, "CPU_EXPLICIT_MONOTONIC", monotonic=123.25, synthetic=True)
    check("actual_event_duplicate_explicit_monotonic_safe", row["monotonic"] == 123.25
          and get_jsonl(event_path)[0]["monotonic"] == 123.25)
    identity = dict(pid=456789, starttick=100, cmdline=["fixture"])
    module.process_identity = lambda pid: dict(identity)
    check("actual_same_process_exact_identity", module.same_process(identity) is False) # current observation injected false
    original_same = load("_actual_identity_only", module_path)
    original_same.process_identity = lambda pid: dict(identity)
    check("actual_identity_exact_match_accepted", original_same.same_process(identity))
    original_same.process_identity = lambda pid: dict(identity, starttick=101)
    check("actual_identity_same_PID_new_starttick_rejected", not original_same.same_process(identity))
    original_same.process_identity = lambda pid: dict(identity, cmdline=["foreign"])
    check("actual_identity_same_PID_new_argv_rejected", not original_same.same_process(identity))
    original_same.ROOT = positive
    with patch.object(original_same.os, "kill") as kill:
        rejects("actual_stop_own_rejects_foreign_argv_before_signal", lambda: original_same.stop_own(identity, positive, "CPU test"))
        check("foreign_rejection_makes_no_signal_syscall", not kill.called)
    return dict(positive_actual_metadata_count=58, negative_admission_cases=len(cases), no_real_GPU_or_P1_access=True)


def mutate(path, key, value):
    row = get(path); row[key] = value; put(path, row)


def get_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def scheduler_tests(directory):
    ctrl = load("_actual_controller_reference_cpu", ROOT / "scripts/resident_controller.py")
    r = directory / "reference_scheduler"
    (r / "operations").mkdir(parents=True)
    put(r / "SOURCE_FREEZE.json", dict(synthetic=True))
    control = types.SimpleNamespace(TASK="CPU_FIXTURE_ONLY", UUIDS={i: "CPU_UUID_%d" % i for i in (1,2,3)},
        process_identity=lambda pid: dict(pid=pid, starttick=1, cmdline=["CPU_FIXTURE_ONLY"]),
        sha=sha, read=get, atomic_new=lambda p,v: put(p,v),
        event=lambda p,k,**kw: append_event(p,k,**kw))
    c = ctrl.Controller(r, control)
    a0 = dict(id="CPU_PAIR_A_R0", gpu=1, repeat=0, reference_unit="CPU_PAIR_A")
    a1 = dict(a0, id="CPU_PAIR_A_R1", repeat=1)
    b0 = dict(id="CPU_PAIR_B_R0", gpu=2, repeat=0, reference_unit="CPU_PAIR_B")
    ram = directory / "synthetic_RAM_not_dev_shm"
    original_path = Path
    def map_path(value):
        return ram if str(value) == "/dev/shm" else original_path(value)
    with patch.object(ctrl, "Path", map_path), patch.object(ctrl.os, "statvfs", return_value=types.SimpleNamespace(f_bavail=99, f_frsize=1024**3)):
        rejects("reference_R1_without_R0_reservation_rejected", lambda: c.reserve_reference_pair(a1))
        rejects("reference_99GiB_before_worker_rejected", lambda: c.reserve_reference_pair(a0))
        check("RAM_rejection_has_no_worker_dispatch", not c.active and not c.attempted)
    with patch.object(ctrl, "Path", map_path), patch.object(ctrl.os, "statvfs", return_value=types.SimpleNamespace(f_bavail=100, f_frsize=1024**3)):
        check("reference_100GiB_reservation_accepted", c.reserve_reference_pair(a0))
        check("other_GPU_pair_waits_while_global_pair_held", c.reserve_reference_pair(b0) is False)
        rejects("R1_completion_before_R0_PASS_rejected", lambda: c.finish_reference_unit(a1))
        c.finish_reference_unit(a0)
        check("R0_PASS_keeps_global_pair_reservation", c.reference_pair["R0_validated"] is True and c.pair_handle is not None)
        rejects("reference_R0_cannot_complete_twice", lambda: c.finish_reference_unit(a0))
        put(c.target("qualification", a1) / "REFERENCE_RELEASED.json", dict(status="PASS", paired_actual_bytes_PASS=True))
        owner = Path(c.reference_pair["RAM_root"]) / "PAIR_OWNER.json"
        put(owner, dict(synthetic=True))
        rejects("R1_cleanup_receipt_cannot_hide_live_reference_owner", lambda: c.finish_reference_unit(a1))
        owner.unlink()
        c.finish_reference_unit(a1)
        check("R1_PASS_and_actual_cleanup_releases_global_pair", c.reference_pair is None and c.pair_handle is None)
        check("next_GPU_pair_can_reserve_after_first_pair_finished", c.reserve_reference_pair(b0))
        c.finish_reference_unit(b0)
        b1 = dict(b0, id="CPU_PAIR_B_R1", repeat=1)
        put(c.target("qualification", b1) / "REFERENCE_RELEASED.json", dict(status="PASS", paired_actual_bytes_PASS=True))
        c.finish_reference_unit(b1)
    u = dict(id="CPU_DUPLICATE", gpu=1, gpu_uuid="CPU_UUID_1")
    c.attempted.add(("qualification", u["id"]))
    rejects("same_unit_dispatch_attempt_cannot_repeat", lambda: c.dispatch("qualification", u))
    # A None identity after successful Popen must become a genuine unresolved
    # launch failure; never install a fabricated identity or PID-only signal.
    c.C.verify_sources = lambda root: "synthetic"
    c.C.query_gpu = lambda: dict(apps=[])
    for name in ("GPU_PERMIT.json", "QUALIFICATION_JOB_MANIFEST.json"):
        put(r / name, dict(synthetic=True))
    c.C.PYTHON = sys.executable
    c.C.process_identity = lambda pid: None
    (r / "scripts").mkdir()
    (r / "scripts/qualify_gpu.py").write_text("pass\n")
    class FakePopen:
        pid = 9999999
        def poll(self): return 0
    with patch.object(ctrl.subprocess, "Popen", return_value=FakePopen()) as launched:
        rejects("Popen_None_identity_is_fail_closed", lambda: c.dispatch("qualification", dict(u, id="CPU_NONE_IDENTITY")))
    check("unresolved_identity_preserved_no_PID_only_cleanup", len(c.unresolved_launches) == 1
          and c.unresolved_launches[0]["identity"] == "UNRESOLVED_NO_PID_ONLY_SIGNAL_ALLOWED" and not c.active)
    rejects("controller_missing_exact_identity_rejected", lambda: ctrl.Controller(r, c.C))
    c.C.process_identity = lambda pid: dict(pid=pid, starttick=1, cmdline=["CPU_FIXTURE_ONLY"])
    c.C.same_process = lambda identity: False
    poller = ctrl.Controller(r / "poll_race", c.C)
    identity = dict(pid=789012, starttick=123, cmdline=["CPU_OWN_FIXTURE"])
    class ExitBetweenObservations:
        def __init__(self): self.observations = iter([None, 0])
        def poll(self): return next(self.observations)
    record = dict(process=ExitBetweenObservations(), identity=identity, uuid="CPU_UUID_1",
                  gpu=1, phase="qualification", unit=dict(id="CPU_EXIT_RACE"),
                  exit_observed=None, exit_returncode=None)
    poller.active[1] = record
    released = []
    def record_completion(row, release):
        released.append((row["exit_returncode"], release)); del poller.active[row["gpu"]]
    poller.complete_released = record_completion
    poller.poll()
    check("normal_exit_between_poll_and_proc_uses_actual_exit_path", released == [(0, "PASS")] and not poller.active)
    class StillLive:
        def poll(self): return None
    record = dict(record, process=StillLive(), exit_observed=None, exit_returncode=None)
    poller.active[1] = record
    rejects("still_live_changed_identity_remains_failure", poller.poll)
    poller.active.clear()
    c.C.query_gpu = lambda: dict(apps=[dict(pid=identity["pid"], uuid="CPU_UUID_1")])
    c.C.process_identity = lambda pid: dict(identity)
    check("matching_NVML_exact_own_identity_is_not_released", poller.own_nvml_present(identity, "CPU_UUID_1"))
    c.C.process_identity = lambda pid: dict(identity, starttick=124, cmdline=["CPU_FOREIGN_REUSED_PID"])
    check("positive_foreign_PID_reuse_is_not_own_NVML", not poller.own_nvml_present(identity, "CPU_UUID_1"))
    c.C.process_identity = lambda pid: None
    check("NVML_row_without_readable_current_identity_stays_unresolved", poller.own_nvml_present(identity, "CPU_UUID_1"))
    # Changing only timestamps cannot indefinitely postpone the 1800-second
    # actual phase/token progress watchdog.
    progress_target = r / "synthetic_watchdog"
    put(progress_target / "PROGRESS.json", dict(phase="CPU", token=1, monotonic=1000))
    record = dict(record, target=progress_target, last_progress_monotonic=1000.,
                  progress_signature=None, latest_progress=None)
    c.C.same_process = lambda identity: True
    c.C.query_gpu = lambda: dict(apps=[])
    poller.active[1] = record
    with patch.object(ctrl.time, "monotonic", return_value=1000.): poller.poll()
    put(progress_target / "PROGRESS.json", dict(phase="CPU", token=1, monotonic=2801))
    with patch.object(ctrl.time, "monotonic", return_value=2801.):
        rejects("timestamp_only_change_does_not_reset_actual_progress_watchdog", poller.poll)
    poller.active.clear()
    return dict(reference_memory_observation="synthetic99/100GiB boundary only; no actual RAM/GPU allocation",
                controller_identity_scope="real implementation with fake observation dependency")


def append_event(path, kind, **fields):
    row = dict(event=kind, monotonic=time.monotonic()); row.update(fields)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream: stream.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path)
    args=parser.parse_args()
    evidence = args.evidence_root or args.output.parent / (args.output.stem + "_evidence")
    evidence.mkdir(parents=True, exist_ok=False)
    try:
        subprocess_result = subprocess_failure(evidence)
        admission = actual_control_tests(evidence)
        scheduler = scheduler_tests(evidence)
        check("test_process_did_not_import_Torch", "torch" not in sys.modules)
        result=dict(status="PASS_ISOLATED_STDLIB_CPU_CONTROL_TESTS", check_count=len(CHECKS), checks=CHECKS,
            controller_subprocess=subprocess_result, actual_control_admission=admission, scheduler=scheduler,
            source_pins={name:sha(ROOT/name) for name in ("scripts/resident_controller.py", "scripts/run_resident_offline.sh",
                "src/resident_execution_control.py", "src/resident_rotation.py", "scripts/qualify_gpu.py")},
            CPU_only=True, GPU_units_dispatched=0, real_model_loaded=False, real_P1_accessed=False,
            cannot_claim_GPU_qualification_or_TPS=True, monotonic=time.monotonic())
    except BaseException as error:
        import traceback
        result=dict(status="FAIL_CPU_CONTROL_TEST_TOOL", error=repr(error), traceback=traceback.format_exc(),
                    check_count=len(CHECKS), checks=CHECKS, CPU_only=True, GPU_units_dispatched=0)
        put(args.output, result)
        raise
    put(args.output, result)
    print(json.dumps(dict(status=result["status"],check_count=len(CHECKS),evidence_root=str(evidence))))

if __name__=="__main__":main()
