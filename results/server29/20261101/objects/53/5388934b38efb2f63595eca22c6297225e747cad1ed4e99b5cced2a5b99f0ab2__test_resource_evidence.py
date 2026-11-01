"""CPU-only regression of durable resource evidence; fixtures are not GPU evidence."""
import ast
import copy
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

assert os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_TEST_MUST_HIDE_GPUS"
W=Path(__file__).resolve().parent
D=Path(os.environ["LING_PROSPECTIVE_RESULTS"]).resolve()
import gpu_admission as g
import resource_evidence as evidence

UUID="GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae"
HISTORY=Path("/data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1/execution_05")
HISTORY_EVIDENCE=[]
FIXTURE_RESULTS=[]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def fixture_snapshot():
    value=copy.deepcopy(json.loads((HISTORY/"admission/gpu3__runtime_canary.window.jsonl").read_text().splitlines()[0]))
    value["CPU_SYNTHETIC_FIXTURE_NOT_GPU_QUALIFICATION"]=True
    return value


def append_compute(snapshot,pid,api="both"):
    if api in ["both","NVML"]:
        snapshot["direct_NVML_processes"]["compute"]["processes"].append(
            dict(pid=pid,usedGpuMemory=1024,gpuInstanceId=0,computeInstanceId=0))
    if api in ["both","SMI"]:
        snapshot["SMI_all_processes"]["processes"].append(
            dict(pid=str(pid),type="C",process_name="CPU_SYNTHETIC_FIXTURE",used_memory="1 MiB"))
    return snapshot


def own_fixture():
    owner=g.proc_identity(os.getpid())
    return append_compute(fixture_snapshot(),owner["pid"]),owner


class ResourceRegression(unittest.TestCase):
    def setUp(self):
        audits=D/"audits";audits.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(prefix="CPU_SYNTHETIC_RESOURCE_",dir=audits)
        self.root=Path(self.tmp.name)
        self.raw=self.root/"resource.raw.jsonl"
        self.pinned=self.root/"FAILED_RESOURCE_SNAPSHOT.json"

    def tearDown(self):
        self.tmp.cleanup()

    def assess(self,snapshot,owner=None,assess_fn=None):
        return evidence.assess_with_evidence(snapshot,UUID,owner,
                    raw_path=self.raw,pinned_path=self.pinned,assess_fn=assess_fn)

    def assert_pinned(self,error,snapshot=None):
        self.assertTrue(self.pinned.exists())
        self.assertEqual(str(self.pinned),error.evidence_path)
        self.assertEqual(sha(self.pinned),error.evidence_sha256)
        pin=load(self.pinned);reference=pin["raw_reference"]
        raw=Path(reference["path"]).read_bytes()
        line=raw[reference["offset"]:reference["offset"]+reference["bytes"]]
        self.assertEqual(hashlib.sha256(line).hexdigest(),reference["line_sha256"])
        row=json.loads(line)
        self.assertEqual(pin["snapshot"],row["snapshot"])
        self.assertEqual(pin["snapshot_sha256"],row["snapshot_sha256"])
        self.assertEqual(hashlib.sha256(evidence.canonical_bytes(row["snapshot"])).hexdigest(),row["snapshot_sha256"])
        if snapshot is not None:self.assertEqual(pin["snapshot"],snapshot)
        self.assertTrue(pin["acceptance_predicate_unchanged"])
        return pin

    def fail_snapshot(self,snapshot,owner=None):
        try:self.assess(snapshot,owner)
        except evidence.ResourceEvidenceError as error:
            return error,self.assert_pinned(error,snapshot)
        raise AssertionError("RESOURCE_FIXTURE_WAS_INCORRECTLY_ACCEPTED")

    def test_01_raw_fsync_source_order_and_runtime_readback_before_assess(self):
        # No fsync mock. Runtime predicate reads the already appended full row.
        # AST proves the helper cannot return before actual flush/fsync completes.
        append_tree=ast.parse(inspect.getsource(evidence.append_raw_snapshot))
        calls=[n for n in ast.walk(append_tree) if isinstance(n,ast.Call)]
        flush=next(n.lineno for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=="flush")
        fsync=next(n.lineno for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=="fsync")
        returned=max(n.lineno for n in ast.walk(append_tree) if isinstance(n,ast.Return))
        self.assertLessEqual(flush,fsync);self.assertLess(fsync,returned)
        wrapper=ast.parse(inspect.getsource(evidence.assess_with_evidence))
        append_line=next(n.lineno for n in ast.walk(wrapper) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="append_raw_snapshot")
        assess_line=next(n.lineno for n in ast.walk(wrapper) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="assess_fn")
        self.assertLess(append_line,assess_line)
        snapshot=fixture_snapshot()
        def original_predicate_with_readback(s,u,o,b):
            rows=self.raw.read_bytes().splitlines()
            self.assertEqual(len(rows),1)
            self.assertEqual(json.loads(rows[0])["snapshot"],snapshot)
            self.assertFalse(self.pinned.exists())
            return g.assess(s,u,o,b)
        result=self.assess(snapshot,assess_fn=original_predicate_with_readback)
        self.assertEqual(result,g.assess(snapshot,UUID))
        self.assertFalse(self.pinned.exists())

    def test_02_all_six_actual_historical_admission_snapshots_replay_PASS(self):
        count=0
        for tag in ["gpu3__runtime_canary","pg19-test-26183__FP__parent"]:
            path=HISTORY/"admission"/(tag+".window.jsonl")
            snapshots=[json.loads(line) for line in path.read_text().splitlines() if line.strip()][:3]
            self.assertEqual(len(snapshots),3)
            rows=[]
            for index,snapshot in enumerate(snapshots):
                before=evidence.canonical_bytes(snapshot)
                result=self.assess(snapshot)
                self.assertEqual(result,g.assess(snapshot,UUID))
                self.assertEqual(before,evidence.canonical_bytes(snapshot))
                rows.append(dict(index=index,snapshot_sha256=hashlib.sha256(before).hexdigest(),status="PASS_ORIGINAL_ASSESS_REPLAY"))
                count+=1
            HISTORY_EVIDENCE.append(dict(path=str(path),sha256=sha(path),samples=rows))
        self.assertEqual(count,6)
        self.assertEqual(len(self.raw.read_bytes().splitlines()),6)
        self.assertFalse(self.pinned.exists())

    def test_03_GPU0_occupancy_and_fuser_are_diagnostics_only(self):
        snapshot=fixture_snapshot()
        snapshot["physical"]["stdout"]+="\n0, GPU-OTHER, NVIDIA GeForce RTX 3090, 24576, 22000, 100, 100, P2, 2500, 400, 70, 535.113.01, 00000000:01:00.0"
        snapshot["fuser"]=dict(returncode=0,stdout="11111 22222 33333",stderr="CPU_SYNTHETIC_OTHER_DEVICE_DIAGNOSTIC")
        self.assertEqual(self.assess(snapshot)["compute_PIDs"],[])

    def test_04_same_UID_nonowner_still_STOP(self):
        snapshot,owner=own_fixture();pid=2147483001;append_compute(snapshot,pid)
        original=evidence.read_process_metadata
        def fixture_pid_resolver(p):
            if p==pid:return dict(status="PASS",pid=p,uid=owner["uid"],ppid=1,start_ticks=11,comm="CPU_SYNTHETIC_SAME_UID")
            return original(p)
        with patch.object(evidence,"read_process_metadata",fixture_pid_resolver):
            error,pin=self.fail_snapshot(snapshot,owner)
        self.assertEqual(error.forbidden_pids,[pid])
        self.assertEqual(pin["classification"]["compute_union"],sorted([owner["pid"],pid]))
        self.assertEqual(next(p for p in pin["classification"]["processes"] if p["pid"]==pid)["classification"],"SAME_UID_NONOWNER_FORBIDDEN")

    def test_05_different_UID_nonowner_still_STOP(self):
        snapshot,owner=own_fixture();pid=2147483002;append_compute(snapshot,pid)
        original=evidence.read_process_metadata
        def fixture_pid_resolver(p):
            if p==pid:return dict(status="PASS",pid=p,uid=owner["uid"]+1,ppid=1,start_ticks=12,comm="CPU_SYNTHETIC_FOREIGN_UID")
            return original(p)
        with patch.object(evidence,"read_process_metadata",fixture_pid_resolver):
            error,pin=self.fail_snapshot(snapshot,owner)
        self.assertEqual(error.forbidden_pids,[pid])
        self.assertEqual(next(p for p in pin["classification"]["processes"] if p["pid"]==pid)["classification"],"FOREIGN_NONOWNER_FORBIDDEN")

    def test_06_unknown_nonowner_still_STOP(self):
        snapshot,owner=own_fixture();pid=2147483003;append_compute(snapshot,pid)
        original=evidence.read_process_metadata
        def fixture_pid_resolver(p):
            if p==pid:return dict(status="UNKNOWN",pid=p,error="CPU_SYNTHETIC_UNSEEN_PID")
            return original(p)
        with patch.object(evidence,"read_process_metadata",fixture_pid_resolver):
            error,pin=self.fail_snapshot(snapshot,owner)
        self.assertEqual(error.forbidden_pids,[pid])
        self.assertEqual(next(p for p in pin["classification"]["processes"] if p["pid"]==pid)["classification"],"UNSEEN_OR_UNKNOWN_NONOWNER_FORBIDDEN")

    def test_07_real_CPU_child_in_synthetic_compute_union_still_STOP(self):
        # The child has no GPU visibility and executes no torch/CUDA imports.
        child=subprocess.Popen([sys.executable,"-B","-c","import sys;sys.stdin.read()"],
                               stdin=subprocess.PIPE,env=dict(os.environ,CUDA_VISIBLE_DEVICES=""))
        try:
            snapshot,owner=own_fixture();append_compute(snapshot,child.pid)
            error,pin=self.fail_snapshot(snapshot,owner)
            self.assertEqual(error.forbidden_pids,[child.pid])
            item=next(p for p in pin["classification"]["processes"] if p["pid"]==child.pid)
            self.assertEqual(item["live_metadata"]["ppid"],owner["pid"])
            self.assertEqual(item["classification"],"CHILD_OF_OWNER_FORBIDDEN")
        finally:
            child.stdin.close();child.wait(timeout=10)

    def test_08_SMI_only_nonowner_exact_forbidden_union(self):
        snapshot,owner=own_fixture();pid=2147483004;append_compute(snapshot,pid,"SMI")
        error,pin=self.fail_snapshot(snapshot,owner)
        self.assertEqual(error.forbidden_pids,[pid])
        self.assertEqual(pin["classification"]["compute_union"],sorted([owner["pid"],pid]))
        self.assertNotIn(pid,[p["pid"] for p in snapshot["direct_NVML_processes"]["compute"]["processes"]])

    def test_09_NVML_only_nonowner_exact_forbidden_union(self):
        snapshot,owner=own_fixture();pid=2147483005;append_compute(snapshot,pid,"NVML")
        error,pin=self.fail_snapshot(snapshot,owner)
        self.assertEqual(error.forbidden_pids,[pid])
        self.assertEqual(pin["classification"]["compute_union"],sorted([owner["pid"],pid]))

    def test_10_partial_compute_query_still_FAIL_with_raw(self):
        snapshot=fixture_snapshot()
        snapshot["direct_NVML_processes"]["compute"]=dict(status="UNKNOWN",returncode=999)
        error,pin=self.fail_snapshot(snapshot)
        self.assertIn("UNKNOWN",pin["classification"]["status"])
        self.assertIn("PROCESS_VISIBILITY_UNKNOWN",str(error.original_error))

    def test_11_unknown_NVML_memory_query_still_FAIL_with_raw(self):
        snapshot=fixture_snapshot();snapshot["direct_NVML"]["status"]="UNKNOWN"
        error,pin=self.fail_snapshot(snapshot)
        self.assertIn("NVML_UNAVAILABLE",str(error.original_error))

    def test_12_same_PID_starttime_reuse_is_rejected_by_original_predicate(self):
        snapshot,owner=own_fixture();owner=copy.deepcopy(owner);owner["start_ticks"]+=1
        error,pin=self.fail_snapshot(snapshot,owner)
        self.assertIn("OWN_PROCESS_IDENTITY_CHANGED",str(error.original_error))
        self.assertEqual(error.forbidden_pids,[])
        self.assertEqual(pin["owner"],owner)

    def test_13_first_pinned_failure_is_never_overwritten(self):
        snapshot=fixture_snapshot();append_compute(snapshot,2147483006)
        first,pin=self.fail_snapshot(snapshot);before=sha(self.pinned)
        next_snapshot=fixture_snapshot();append_compute(next_snapshot,2147483007)
        with self.assertRaises(evidence.ResourceEvidenceError) as second:
            self.assess(next_snapshot)
        self.assertEqual(sha(self.pinned),before)
        self.assertEqual(second.exception.evidence_sha256,before)
        self.assertEqual(load(self.pinned),pin)
        self.assertEqual(len(self.raw.read_bytes().splitlines()),2)

    def test_14_sampling_exception_unknown_is_durable_not_empty(self):
        error=subprocess.CalledProcessError(9,["CPU_SYNTHETIC_QUERY"],output="partial stdout",stderr="partial stderr")
        with self.assertRaises(evidence.ResourceEvidenceError) as captured:
            evidence.record_sampling_failure(error,UUID,raw_path=self.raw,pinned_path=self.pinned)
        pin=self.assert_pinned(captured.exception)
        self.assertEqual(pin["status"],"UNKNOWN")
        self.assertTrue(pin["snapshot"]["no_successful_snapshot_available"])
        self.assertEqual(pin["snapshot"]["error"]["returncode"],9)
        self.assertEqual(pin["snapshot"]["error"]["stdout"],"partial stdout")
        self.assertIn("UNKNOWN",pin["classification"]["status"])

    def test_15_append_offsets_and_pinned_SHA_reproduce_exact_bytes(self):
        self.assess(fixture_snapshot())
        snapshot=fixture_snapshot();append_compute(snapshot,2147483008)
        error,pin=self.fail_snapshot(snapshot)
        self.assertGreater(pin["raw_reference"]["offset"],0)
        self.assertEqual(pin["snapshot_sha256"],hashlib.sha256(evidence.canonical_bytes(snapshot)).hexdigest())
        self.assertEqual(len(self.raw.read_bytes().splitlines()),2)

    def fixture_subprocess(self,program,extra_files=None):
        root=self.root/"ISOLATED_SYNTHETIC_CONTROL_D";root.mkdir()
        for name,value in (extra_files or {}).items():
            (root/name).write_text(json.dumps(value))
        env=dict(os.environ,LING_PROSPECTIVE_RESULTS=str(root),CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1")
        env["PYTHONPATH"]=str(W)+(os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        run=subprocess.run([sys.executable,"-B","-c",program],cwd=W,env=env,text=True,capture_output=True,timeout=30)
        self.assertEqual(run.returncode,0,run.stdout+run.stderr)
        return json.loads(run.stdout.splitlines()[-1]),root

    def test_16_first_STOP_cannot_be_overwritten_actual_subprocess(self):
        program='''import hashlib,json,os
from cleanroom_common import stop,D
first=stop("CPU_SYNTHETIC_FIRST_STOP",stage="CPU_TEST_ONLY")
before=hashlib.sha256((D/"STOP.json").read_bytes()).hexdigest()
second=stop("CPU_SYNTHETIC_SECOND_STOP",stage="CPU_TEST_ONLY")
after=hashlib.sha256((D/"STOP.json").read_bytes()).hexdigest()
assert before==after and first==second
print(json.dumps(dict(first_SHA=before,second_SHA=after,reason=second["reason"],fixture_only=True,no_GPU_execution=True)))
'''
        result,root=self.fixture_subprocess(program)
        self.assertEqual(result["first_SHA"],result["second_SHA"])
        self.assertEqual(result["reason"],"CPU_SYNTHETIC_FIRST_STOP")
        self.assertNotEqual(root,D)

    def test_17_same_task3090_budget_excludes_independent4090(self):
        import cleanroom_finalize as finalizer
        value=finalizer.resource_budget_context(dict(prior_same_task_GPUh=1.007654354178602,
            independent_RTX4090_qualification_GPUh=3.4473476),1.25)
        self.assertEqual(value["same_task_cumulative_RTX3090_GPUh"],1.25+1.007654354178602)
        self.assertEqual(value["remaining_same_task_RTX3090_GPUh"],34.75-1.007654354178602)
        self.assertEqual(value["independent_RTX4090_qualification_GPUh"],3.4473476)
        with self.assertRaises(AssertionError):
            finalizer.resource_budget_context(dict(prior_same_task_RTX3090_GPUh=0),36.1)

    def test_18_actual_singleGPU_controller_ledger_stops_below36(self):
        program='''import json,time
from cleanroom_run import Controller,D
controller=Controller()
controller.completed=[dict(wall_seconds=(35.75-controller.prior_GPUh)*3600)]
controller.active={"CPU_SYNTHETIC_NOT_GPU_WORKER":dict(started_monotonic=time.monotonic()-.06*3600)}
row=controller.ledger()
assert 35.8<row["cumulative_same_task_GPU_hours"]<36.
assert (D/"STOP.json").exists()
print(json.dumps(dict(ledger=row,STOP=json.loads((D/"STOP.json").read_text()),fixture_only=True,actual_GPU_workers=0)))
'''
        routing=load(D/"EXECUTION_ROUTING.json")
        row,root=self.fixture_subprocess(program,{"EXECUTION_ROUTING.json":routing,"preregistration.json":dict(prior_same_task_GPUh=1.007654354178602)})
        self.assertEqual(row["ledger"]["MAX_GPU_HOURS"],36.0)
        self.assertEqual(row["actual_GPU_workers"],0)
        self.assertIn("BUDGET",row["STOP"]["reason"])
        self.assertNotEqual(root,D)

    def test_19_pin_uses_fsynced_raw_even_if_predicate_mutates_metadata(self):
        snapshot=fixture_snapshot();append_compute(snapshot,2147483009)
        original=copy.deepcopy(snapshot)
        def original_predicate_with_side_effect(s,u,o,b):
            s["CPU_SYNTHETIC_PREDICATE_METADATA_SIDE_EFFECT"]=True
            return g.assess(s,u,o,b)
        with self.assertRaises(evidence.ResourceEvidenceError) as failure:
            self.assess(snapshot,assess_fn=original_predicate_with_side_effect)
        pin=self.assert_pinned(failure.exception,original)
        self.assertNotIn("CPU_SYNTHETIC_PREDICATE_METADATA_SIDE_EFFECT",pin["snapshot"])
        self.assertIn("CPU_SYNTHETIC_PREDICATE_METADATA_SIDE_EFFECT",snapshot)

    def test_98_only_readonly_PID_resolver_was_mocked_and_predicate_unchanged(self):
        source=Path(__file__).read_text();tree=ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=="object" and isinstance(node.func.value,ast.Name) and node.func.value.id=="patch":
                self.assertEqual(node.args[1].value,"read_process_metadata")
        self.assertNotIn("torch",sys.modules)
        self.assertNotIn("fla",sys.modules)


def main():
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ResourceRegression))
    report=dict(status="PASS" if result.wasSuccessful() and not result.skipped else "FAIL",
                scope="CPU_RESOURCE_EVIDENCE_REGRESSION_ONLY",tests=result.testsRun,skips=len(result.skipped),
                failures=len(result.failures),errors=len(result.errors),log=stream.getvalue(),
                actual_historical_snapshot_replay=HISTORY_EVIDENCE,
                test_source_SHA256=sha(__file__),
                source_SHA256={n:sha(W/n) for n in ["resource_evidence.py","cleanroom_run.py","gpu_admission.py"]},
                fixtures="explicit CPU synthetic snapshots, not live NVML or scientific evidence",
                mocks="read_process_metadata classification resolver only; original gpu_admission.assess unchanged",
                actual_GPU_workers=0,numerical_qualification=False,science_trajectory_count=0,
                no_model_or_CUDA_or_FLA_import=True,epoch=time.time())
    path=D/"audits/CPU_RESOURCE_EVIDENCE_REGRESSION.json"
    with path.open("x") as f:
        json.dump(report,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    print(stream.getvalue(),flush=True)
    print(json.dumps(dict(status=report["status"],tests=result.testsRun,report=str(path),SHA256=sha(path))),flush=True)
    assert result.wasSuccessful() and not result.skipped


if __name__=="__main__":main()
