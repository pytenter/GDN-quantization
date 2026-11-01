"""CPU lifecycle regressions. All telemetry and artifacts are explicit fixtures, never GPU evidence."""
import ast,copy,hashlib,inspect,io,json,os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
assert os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU_REGRESSION_MUST_HIDE_CUDA"
assert os.environ.get("PYTHONDONTWRITEBYTECODE")=="1","CPU_REGRESSION_NO_PYCACHE"
W=Path(__file__).resolve().parent
D=Path(os.environ["LING_PROSPECTIVE_RESULTS"]).resolve()
UUID="GPU-de53f243-b271-521c-d875-d96611ed8574"
OLD_D=Path("/data/zypan/experiment_runs/LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1_GPU3_SINGLE_V2")
ORIGINAL_GPU_ADMISSION_SHA="e51b98fefe0c1c3359f8e7995ed7439e6bc3c50fec66f28d79d995d56f78f21a"
import gpu_admission as g
import lifecycle_guard as lifecycle
import resource_evidence as evidence
import cleanroom_common as common
CHILD_EVENTS=[]
HISTORICAL_REPLAY=[]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def canonical(v):return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def fixture_query(stdout=""):
 return dict(command=["CPU_SYNTHETIC_FIXTURE_NO_GPU_QUERY"],returncode=0,stdout=stdout,stderr="")
def baseline():
 return [dict(pid=4040,uid=120,start_ticks=3403,comm="Xorg",status="PASS")]
def idle_fixture():
 total=24576*2**20;reserved=452*2**20;used=15*2**20
 return dict(CPU_SYNTHETIC_FIXTURE_NOT_GPU_EVIDENCE=True,uuid=UUID,epoch=100.,
  physical=fixture_query(),target=[dict(index="6",uuid=UUID,name="NVIDIA GeForce RTX 3090",total_MiB="24576",used_MiB="15",GPU_util="0")],
  compute=fixture_query(),early_SMI_compute_processes=dict(status="PASS",processes=[]),fuser=dict(command=["CPU_FIXTURE"],returncode=1,stdout="",stderr=""),
  direct_NVML=dict(status="PASS",temperature=30,power_mW=20000,pstate=8,SM_clock_MHz=210,memory=dict(total=total,free=total-reserved-used,used=reserved+used),
   memory_v2=dict(version=33554472,total=total,free=total-reserved-used,used=used,reserved=reserved),utilization=dict(gpu=0,memory=0)),
  direct_NVML_processes=dict(compute=dict(status="PASS",api="CPU_FIXTURE",processes=[]),
   graphics=dict(status="PASS",api="CPU_FIXTURE",processes=[dict(pid=4040,usedGpuMemory=4677632)])),
  SMI_XML=fixture_query("<nvidia_smi_log><timestamp>CPU_FIXTURE</timestamp><gpu><uuid>"+UUID+"</uuid><fb_memory_usage><used>15 MiB</used></fb_memory_usage></gpu></nvidia_smi_log>"),
  SMI_all_processes=dict(status="PASS",processes=[dict(pid="4040",type="G",name="/usr/lib/xorg/Xorg",used_memory="4 MiB")]),
  process_ownership=baseline())

def add_pid(s,pid,api="all",owner=None):
 if api in ("all","NVML"):s["direct_NVML_processes"]["compute"]["processes"].append(dict(pid=pid,usedGpuMemory=1024))
 if api in ("all","SMI_XML"):s["SMI_all_processes"]["processes"].append(dict(pid=str(pid),type="C",name="CPU_FIXTURE",used_memory="1 MiB"))
 if api in ("all","SMI_EARLY"):
  s["compute"]["stdout"]+=f"{UUID}, {pid}, 1\n"
  s["early_SMI_compute_processes"]["processes"]=lifecycle.early_compute_processes(s,UUID)
 if owner is not None:
  s["process_ownership"].append(dict(owner,status="PASS",comm="CPU_FIXTURE"))
 return s

def material_fixture(owner,listed=True):
 s=idle_fixture();m=s["direct_NVML"]["memory_v2"];m["used"]=1000*2**20;m["free"]=m["total"]-m["reserved"]-m["used"]
 s["direct_NVML"]["memory"]["used"]=m["used"]+m["reserved"];s["direct_NVML"]["memory"]["free"]=m["free"]
 s["target"][0]["used_MiB"]="1000"
 if listed:add_pid(s,owner["pid"],owner=owner)
 return s

class CPUChild:
 def __init__(self,root,exit_code=0,artifacts=None):
  self.root=Path(root);self.exit_code=exit_code;self.process=None;self.artifacts=artifacts
 def __enter__(self):
  self.root.mkdir(parents=True,exist_ok=True)
  if self.artifacts is not None:(self.root/"CPU_INPUT_ARTIFACTS.json").write_text(json.dumps(self.artifacts))
  # A real interpreter with CUDA hidden. No model/library imports.
  program=r'''import json,os,pathlib,sys,time
root=pathlib.Path(sys.argv[1]);code=int(sys.argv[2])
print("CPU_CHILD_READY",flush=True)
sys.stdin.readline()
(root/"CPU_RESULT.json").write_text(json.dumps(dict(status="CPU_SYNTHETIC_RESULT",pid=os.getpid())))
(root/"CPU_PROFILE.json").write_text(json.dumps(dict(status="CPU_SYNTHETIC_PROFILE",CUDA_used=False)))
if (root/"CPU_INPUT_ARTIFACTS.json").exists():
 for name,value in json.loads((root/"CPU_INPUT_ARTIFACTS.json").read_text()).items():(root/name).write_text(json.dumps(value,sort_keys=True))
sys.exit(code)
'''
  env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1")
  self.process=subprocess.Popen([sys.executable,"-B","-u","-c",program,str(self.root),str(self.exit_code)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
  assert self.process.stdout.readline().strip()=="CPU_CHILD_READY"
  self.identity=g.proc_identity(self.process.pid);assert self.identity["pid"]==self.process.pid
  return self
 def natural_finish_inside_sampling(self):
  started=time.monotonic();before=self.process.poll();assert before is None
  self.process.stdin.write("EXIT\n");self.process.stdin.flush()
  # Waiting models a read-only sampling call spanning a real child exit.
  rc=self.process.wait(timeout=5);time.sleep(.025)
  CHILD_EVENTS.append(dict(pid=self.identity["pid"],identity=self.identity,poll_before=before,poll_after=rc,elapsed=time.monotonic()-started,CPU_only=True,fixture_artifacts_not_science=True))
  return before,rc
 def __exit__(self,*exc):
  if self.process.poll() is None:
   try:self.process.stdin.write("EXIT\n");self.process.stdin.flush();self.process.wait(timeout=5)
   except Exception:self.process.terminate();self.process.wait(timeout=5)
  for stream in [self.process.stdin,self.process.stdout,self.process.stderr]:
   if stream:stream.close()

class LifecycleRegression(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix="CPU_LIFECYCLE_FIXTURE_",dir=D/"audits")
  self.root=Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def check(self,s,owner,before=None,after=0):
  return lifecycle.assess_lifecycle_sample(s,UUID,owner,baseline(),before,after,assess_fn=g.assess)
 def assert_rejected(self,s,owner,before=None,after=0,regex=None):
  with self.assertRaises((AssertionError,ValueError,KeyError,RuntimeError)) as caught:self.check(s,owner,before,after)
  if regex:self.assertIn(regex,str(caught.exception))
  return str(caught.exception)

 def test_01_real_child_exits_during_sample_transition_is_not_resource_PASS(self):
  with CPUChild(self.root) as c:
   s=material_fixture(c.identity,listed=False);s["compute"]["stdout"]=f"{UUID}, {c.identity['pid']}, 1000\n";s["early_SMI_compute_processes"]["processes"]=lifecycle.early_compute_processes(s,UUID)
   before,after=c.natural_finish_inside_sampling()
   v=self.check(s,c.identity,before,after)
   self.assertEqual(v["lifecycle_state"],"NATURAL_EXIT_TRANSITION_NOT_RESOURCE_PASS")
   self.assertNotEqual(v.get("status"),"PASS")
   self.assertTrue((self.root/"CPU_RESULT.json").exists());self.assertTrue((self.root/"CPU_PROFILE.json").exists())
 def test_02_live_child_complete_owner_still_uses_original_predicate(self):
  with CPUChild(self.root) as c:
   v=self.check(material_fixture(c.identity),c.identity,None,c.process.poll())
   self.assertEqual(v["lifecycle_state"],"RUNNING_RESOURCE_PASS")
 def test_03_live_child_missing_owner_is_original_line43_STOP(self):
  with CPUChild(self.root) as c:self.assert_rejected(material_fixture(c.identity,False),c.identity,None,None,"MATERIAL_MEMORY_WITHOUT_ATTRIBUTABLE_COMPUTE_PID")
 def test_04_nonzero_natural_child_exit_rejected(self):
  with CPUChild(self.root,7) as c:
   before,after=c.natural_finish_inside_sampling();self.assertEqual(after,7);self.assert_rejected(idle_fixture(),c.identity,before,after)
 def test_05_signal_exit_rejected(self):
  with CPUChild(self.root) as c:
   before=c.process.poll();c.process.send_signal(signal.SIGTERM);after=c.process.wait(timeout=5)
   self.assertLess(after,0);self.assert_rejected(idle_fixture(),c.identity,before,after)
 def test_06_already_exited_before_sample_is_rejected(self):
  with CPUChild(self.root) as c:
   c.natural_finish_inside_sampling();self.assert_rejected(idle_fixture(),c.identity,0,0)
 def test_07_actual_live_PID_starttime_change_rejected(self):
  with CPUChild(self.root) as c:
   wrong=dict(c.identity,start_ticks=c.identity["start_ticks"]+1)
   self.assert_rejected(material_fixture(wrong),wrong,None,None,"OWN_PROCESS_IDENTITY_CHANGED")
 def test_08_actual_live_UID_change_rejected(self):
  with CPUChild(self.root) as c:
   wrong=dict(c.identity,uid=c.identity["uid"]+1)
   self.assert_rejected(material_fixture(wrong),wrong,None,None,"OWN_PROCESS_IDENTITY_CHANGED")
 def test_09_three_sources_foreign_still_rejected_after_exit(self):
  with CPUChild(self.root) as c:
   c.natural_finish_inside_sampling()
   for api in ["SMI_EARLY","NVML","SMI_XML"]:
    with self.subTest(api=api):
     s=add_pid(idle_fixture(),2147483001,api);self.assert_rejected(s,c.identity)
 def test_10_same_UID_real_other_CPU_child_is_not_authorized(self):
  with CPUChild(self.root/"a") as owner,CPUChild(self.root/"b") as other:
   self.assertEqual(owner.identity["uid"],other.identity["uid"])
   s=add_pid(material_fixture(owner.identity),other.identity["pid"],owner=other.identity)
   self.assert_rejected(s,owner.identity,None,None)
 def test_11_UNKNOWN_returncode_for_each_SMI_source_rejected_after_exit(self):
  with CPUChild(self.root) as c:
   c.natural_finish_inside_sampling()
   for source in ["physical","compute","SMI_XML"]:
    with self.subTest(source=source):
     s=idle_fixture();s[source]["returncode"]=None;self.assert_rejected(s,c.identity)
 def test_12_UNKNOWN_NVML_and_parsed_process_sources_rejected_after_exit(self):
  with CPUChild(self.root) as c:
   c.natural_finish_inside_sampling()
   for source in ["direct_NVML","SMI_all_processes"]:
    s=idle_fixture();s[source]["status"]="UNKNOWN"
    with self.subTest(source=source):self.assert_rejected(s,c.identity)
   for group in ["compute","graphics"]:
    s=idle_fixture();s["direct_NVML_processes"][group]["status"]="UNKNOWN"
    with self.subTest(group=group):self.assert_rejected(s,c.identity)
 def test_13_UNKNOWN_NVML_memory_utilization_rejected_after_exit(self):
  with CPUChild(self.root) as c:
   c.natural_finish_inside_sampling()
   for field in ["memory","memory_v2","utilization"]:
    s=idle_fixture();s["direct_NVML"][field]=dict(status="UNKNOWN",returncode=999)
    with self.subTest(field=field):self.assert_rejected(s,c.identity)
 def test_14_missing_query_sources_are_not_empty_idle(self):
  with CPUChild(self.root) as c:
   c.natural_finish_inside_sampling()
   for source in ["physical","compute","SMI_XML","direct_NVML","direct_NVML_processes","SMI_all_processes","early_SMI_compute_processes"]:
    s=idle_fixture();del s[source]
    with self.subTest(source=source):self.assert_rejected(s,c.identity)
 def test_15_malformed_early_CSV_rejected(self):
  with CPUChild(self.root) as c:
   c.natural_finish_inside_sampling();s=idle_fixture();s["compute"]["stdout"]="not,a,valid,row"
   self.assert_rejected(s,c.identity)
 def test_16_wrong_target_UUID_rejected(self):
  with CPUChild(self.root) as c:
   c.natural_finish_inside_sampling();s=idle_fixture();s["target"][0]["uuid"]="GPU_OTHER"
   self.assert_rejected(s,c.identity)
 def test_17_graphics_baseline_change_rejected(self):
  with CPUChild(self.root) as c:
   c.natural_finish_inside_sampling();s=idle_fixture();s["process_ownership"][0]["start_ticks"]+=1
   self.assert_rejected(s,c.identity)
 def test_18_unknown_graphics_owner_rejected(self):
  with CPUChild(self.root) as c:
   c.natural_finish_inside_sampling();s=idle_fixture();s["process_ownership"][0]["status"]="UNKNOWN"
   self.assert_rejected(s,c.identity)
 def test_19_memory_arithmetic_rejected(self):
  with CPUChild(self.root) as c:
   c.natural_finish_inside_sampling();s=idle_fixture();s["direct_NVML"]["memory_v2"]["free"]-=1
   self.assert_rejected(s,c.identity)
 def test_20_insufficient_true_headroom_rejected(self):
  with CPUChild(self.root) as c:
   c.natural_finish_inside_sampling();s=idle_fixture();m=s["direct_NVML"]["memory_v2"];m["free"]=g.HEADROOM-1;m["used"]=m["total"]-m["reserved"]-m["free"]
   self.assert_rejected(s,c.identity)
 def test_21_terminal_exact_three_original_idle_samples_PASS(self):
  rows=[idle_fixture() for unused in range(3)]
  v=lifecycle.assess_terminal_window(rows,UUID,baseline(),assess_window_fn=g.assess_window)
  self.assertEqual(v["status"],"PASS");self.assertEqual(v["sample_count"],3)
 def test_22_terminal_wrong_sample_count_rejected(self):
  for count in [0,1,2,4]:
   with self.subTest(count=count),self.assertRaises((AssertionError,ValueError)):
    lifecycle.assess_terminal_window([idle_fixture() for unused in range(count)],UUID,baseline(),assess_window_fn=g.assess_window)
 def test_23_terminal_any_one_foreign_sample_rejected(self):
  for index in range(3):
   for api in ["SMI_EARLY","NVML","SMI_XML"]:
    rows=[idle_fixture() for unused in range(3)];add_pid(rows[index],2147483002,api)
    with self.subTest(index=index,api=api),self.assertRaises((AssertionError,ValueError,KeyError)):
     lifecycle.assess_terminal_window(rows,UUID,baseline(),assess_window_fn=g.assess_window)
 def test_24_terminal_any_one_UNKNOWN_sample_rejected(self):
  for index in range(3):
   rows=[idle_fixture() for unused in range(3)];rows[index]["direct_NVML_processes"]["compute"]["status"]="UNKNOWN"
   with self.subTest(index=index),self.assertRaises((AssertionError,ValueError,KeyError)):
    lifecycle.assess_terminal_window(rows,UUID,baseline(),assess_window_fn=g.assess_window)
 def test_25_terminal_material_GPU_util_rejected(self):
  rows=[idle_fixture() for unused in range(3)];rows[-1]["direct_NVML"]["utilization"]["gpu"]=3
  with self.assertRaisesRegex(AssertionError,"OCCUPANCY"):lifecycle.assess_terminal_window(rows,UUID,baseline(),assess_window_fn=g.assess_window)
 def test_26_terminal_material_allocation_rejected(self):
  owner=dict(pid=2147483003,uid=1,start_ticks=1);rows=[idle_fixture(),idle_fixture(),material_fixture(owner,False)]
  with self.assertRaises(AssertionError):lifecycle.assess_terminal_window(rows,UUID,baseline(),assess_window_fn=g.assess_window)
 def test_27_historical_failed_raw_keeps_original_line43_STOP(self):
  p=OLD_D/"FAILED_RESOURCE_SNAPSHOT.json";raw=p.read_bytes();pin=json.loads(raw)
  self.assertEqual(hashlib.sha256(raw).hexdigest(),"22cd8873acee4ed2f17732cca03bdb4bf612890bebc6f6e08ee05c67fa9d14a9")
  s=pin["snapshot"];owner=pin["owner"];before=hashlib.sha256(canonical(s)).hexdigest()
  with patch.object(g,"proc_identity",lambda pid:copy.deepcopy(owner)):
   with self.assertRaisesRegex(AssertionError,"MATERIAL_MEMORY_WITHOUT_ATTRIBUTABLE_COMPUTE_PID"):g.assess(copy.deepcopy(s),pin["uuid"],owner,pin["baseline"])
  self.assertEqual(before,hashlib.sha256(canonical(s)).hexdigest())
  HISTORICAL_REPLAY.append(dict(path=str(p),SHA256=sha(p),snapshot_SHA256=before,original_assertion="MATERIAL_MEMORY_WITHOUT_ATTRIBUTABLE_COMPUTE_PID",old_STOP_retained=True,retroactive_PASS=False,PID_resolver_fixture_only=True))
 def test_28_raw_snapshot_fsynced_before_lifecycle_predicate(self):
  s=idle_fixture();raw=self.root/"raw.jsonl";pin=self.root/"FAIL.json"
  def predicate(snapshot,uuid,owner,base):
   rows=raw.read_text().splitlines();self.assertEqual(len(rows),1)
   self.assertEqual(json.loads(rows[0])["snapshot"],snapshot)
   return lifecycle.assess_lifecycle_sample(snapshot,uuid,owner,base,None,0,assess_fn=g.assess)
  value=evidence.assess_with_evidence(s,UUID,dict(pid=1),baseline(),raw_path=raw,pinned_path=pin,assess_fn=predicate)
  self.assertEqual(value["lifecycle_state"],"NATURAL_EXIT_TRANSITION_NOT_RESOURCE_PASS");self.assertFalse(pin.exists())
  tree=ast.parse(inspect.getsource(evidence.append_raw_snapshot))
  self.assertTrue(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="fsync" for n in ast.walk(tree)))
 def test_29_first_STOP_is_not_overwritten(self):
  with patch.object(common,"D",self.root):
   common.stop("CPU_FIXTURE_FIRST",stage="CPU_TEST_ONLY");before=sha(self.root/"STOP.json")
   common.stop("CPU_FIXTURE_SECOND",stage="CPU_TEST_ONLY")
   self.assertEqual(before,sha(self.root/"STOP.json"));self.assertEqual(load(self.root/"STOP.json")["reason"],"CPU_FIXTURE_FIRST")
 def test_30_gpu_admission_predicate_bytes_unchanged(self):
  self.assertEqual(sha(W/"gpu_admission.py"),ORIGINAL_GPU_ADMISSION_SHA)
 def test_31_controller_actual_sampling_is_bracketed_by_poll(self):
  import cleanroom_run as runner
  source=inspect.getsource(runner.Controller.job);self.assertIn("poll_before",source);self.assertIn("poll_after",source)
  tree=ast.parse(source.lstrip());names=[n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
  self.assertIn("assess_lifecycle_sample",names)
  attrs=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
  self.assertIn("_completion_bindings",attrs);self.assertIn("_terminal_gate",attrs)
  self.assertIn("validate_completion_assets",inspect.getsource(runner.Controller._completion_bindings))
  self.assertIn("assess_terminal_window",inspect.getsource(runner.Controller._terminal_gate))
  assigns={n.targets[0].id:n.lineno for n in ast.walk(tree) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
  sample_lines=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="sample"]
  self.assertEqual(len(sample_lines),1);self.assertLess(assigns["poll_before"],sample_lines[0]);self.assertLess(sample_lines[0],assigns["poll_after"])
 def completion_fixture(self,kind="parent"):
  expected=dict(PYTHONHASHSEED="0",CUBLAS_WORKSPACE_CONFIG=":4096:8",LING_GATED_RMSNORM_FIXED_CONFIG="BT16_W8",FLA_CACHE_MODE="default",FLA_CONFIG_DIR=str(W/"frozen_fla_configs"),CUDA_VISIBLE_DEVICES="6",LING_TARGET_GPU_UUID=UUID)
  result=dict(status="PASS",document="pg19-test-41603",condition="H",host=dict(gpu_uuid=UUID),CPU_SYNTHETIC_FIXTURE_NOT_GPU_EVIDENCE=True)
  if kind=="canary":result["UUID"]=UUID
  if kind=="formal1":result["identity"]=dict(UUID=UUID,source_commit="CPU_FIXTURE_SOURCE")
  if kind=="formal2":
   del result["host"];result.update(gpu_uuid=UUID,endpoint=32768,vocabulary_dimension=157184)
  profile=dict(status="PASS_ACTUAL_FROZEN_PROFILE",actual_environment=expected.copy(),hash_randomization=0,profile=dict(status="PASS_ACTUAL_FROZEN_RUNTIME_PROFILE",forward_kernels_executed=True,environment={k:expected[k] for k in ["PYTHONHASHSEED","CUBLAS_WORKSPACE_CONFIG","LING_GATED_RMSNORM_FIXED_CONFIG","FLA_CACHE_MODE","FLA_CONFIG_DIR"]}),CPU_SYNTHETIC_FIXTURE_NOT_GPU_EVIDENCE=True)
  environment=dict(environment=expected.copy(),CPU_SYNTHETIC_FIXTURE_NOT_GPU_EVIDENCE=True)
  kw=dict(kind=kind,document="pg19-test-41603",condition="H",uuid=UUID,expected_environment=expected,source_verified=True,source_commit="CPU_FIXTURE_SOURCE")
  return result,profile,environment,kw
 def test_32_completion_binding_all_worker_kinds_positive(self):
  for kind in ["canary","parent","fresh","formal1","formal2"]:
   with self.subTest(kind=kind):
    a,b,c,kw=self.completion_fixture(kind)
    if kind=="formal2":
     # True frozen formal_worker final publish layout, not a mirror of the helper.
     tree=ast.parse((W/"formal_worker.py").read_text())
     outputs=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="dict" and any(k.arg=="endpoint" and isinstance(k.value,ast.Constant) and k.value.value==32768 for k in n.keywords)]
     layouts=[{k.arg for k in n.keywords} for n in outputs]
     self.assertTrue(any("gpu_uuid" in keys and "host" not in keys for keys in layouts))
     self.assertNotIn("host",a);self.assertEqual(a["gpu_uuid"],UUID)
    v=lifecycle.validate_completion_assets(a,b,c,**kw)
    self.assertEqual(v["status"],"PASS_ARTIFACT_BINDING_ONLY_NOT_JOB_COMPLETE")
 def test_33_completion_rejects_missing_or_failed_result_PROFILE(self):
  for which in ["result_missing","result_FAIL","profile_missing","profile_FAIL"]:
   a,b,c,kw=self.completion_fixture()
   if which=="result_missing":a=None
   if which=="result_FAIL":a["status"]="FAIL"
   if which=="profile_missing":b=None
   if which=="profile_FAIL":b["status"]="FAIL"
   with self.subTest(which=which),self.assertRaises((AssertionError,KeyError,TypeError)):lifecycle.validate_completion_assets(a,b,c,**kw)
 def test_34_completion_rejects_environment_seed_profile_and_forward_drift(self):
  for which in ["launch_env","actual_env","hash_flag","inner_status","forward","inner_env"]:
   a,b,c,kw=self.completion_fixture()
   if which=="launch_env":c["environment"]["PYTHONHASHSEED"]="1"
   if which=="actual_env":b["actual_environment"]["FLA_CACHE_MODE"]="bad"
   if which=="hash_flag":b["hash_randomization"]=1
   if which=="inner_status":b["profile"]["status"]="UNKNOWN"
   if which=="forward":b["profile"]["forward_kernels_executed"]=False
   if which=="inner_env":b["profile"]["environment"]["FLA_CONFIG_DIR"]="/CPU_BAD"
   with self.subTest(which=which),self.assertRaises(AssertionError):lifecycle.validate_completion_assets(a,b,c,**kw)
 def test_35_completion_rejects_scientific_source_device_and_input_identity_drift(self):
  for which in ["document","condition","uuid","source_verified","formal1source"]:
   a,b,c,kw=self.completion_fixture("formal1" if which=="formal1source" else "parent")
   if which=="document":a["document"]="pg19-test-OTHER"
   if which=="condition":a["condition"]="OTHER"
   if which=="uuid":a["host"]["gpu_uuid"]="GPU_OTHER"
   if which=="source_verified":kw["source_verified"]=False
   if which=="formal1source":a["identity"]["source_commit"]="CPU_WRONG"
   with self.subTest(which=which),self.assertRaises(AssertionError):lifecycle.validate_completion_assets(a,b,c,**kw)
 def test_36_per_query_timing_envelope_valid_and_missing_invalid(self):
  keys=["physical","compute","fuser","direct_NVML","direct_NVML_processes","SMI_XML","process_ownership"]
  good=idle_fixture();good["query_timings"]={k:dict(started_monotonic=float(i),completed_monotonic=float(i)+.01) for i,k in enumerate(keys)}
  self.assertTrue(lifecycle.validate_timing_envelope(good))
  for key in keys:
   bad=copy.deepcopy(good);del bad["query_timings"][key]
   with self.subTest(key=key),self.assertRaises(AssertionError):lifecycle.validate_timing_envelope(bad)
  for value in [float("nan"),-1,False,"UNKNOWN"]:
   bad=copy.deepcopy(good);bad["query_timings"]["compute"]["started_monotonic"]=value
   with self.subTest(value=str(value)),self.assertRaises(AssertionError):lifecycle.validate_timing_envelope(bad)
 def test_37_same_task_prior_is_charged_separate4090_is_not(self):
  import cleanroom_finalize as finalizer
  prior=load(D/"preregistration.json")["prior_same_task_GPUh"]
  self.assertAlmostEqual(prior,1.007654354178602,places=14)
  value=finalizer.resource_budget_context(dict(prior_same_task_GPUh=prior,independent_RTX4090_qualification_GPUh=3.447347587855119),.25)
  self.assertAlmostEqual(value["same_task_cumulative_RTX3090_GPUh"],prior+.25,places=14)
  self.assertAlmostEqual(value["remaining_same_task_RTX3090_GPUh"],36-prior-.25,places=14)
  self.assertAlmostEqual(value["current_execution_cap_GPUh"],36-prior,places=14)
  with self.assertRaises(AssertionError):finalizer.resource_budget_context(dict(prior_same_task_GPUh=prior),36-prior+.001)
 def test_38_controller_actual_prior_active_ledger_protective_STOP(self):
  import cleanroom_run as runner
  prior=load(D/"preregistration.json")["prior_same_task_GPUh"]
  fixture=self.root/"BUDGET_CPU_FIXTURE";fixture.mkdir()
  common.publish(fixture/"EXECUTION_ROUTING.json",load(D/"EXECUTION_ROUTING.json"))
  common.publish(fixture/"preregistration.json",dict(prior_same_task_GPUh=prior))
  with patch.object(runner,"D",fixture),patch.object(common,"D",fixture):
   c=runner.Controller();c.completed=[dict(wall_seconds=(35.75-prior)*3600)]
   c.active={"CPU_SYNTHETIC_NOT_GPU_WORKER":dict(started_monotonic=time.monotonic()-.06*3600)}
   value=c.ledger();self.assertAlmostEqual(value["prior_same_task_GPUh"],prior)
   self.assertGreater(value["cumulative_same_task_GPU_hours"],35.8);self.assertLess(value["cumulative_same_task_GPU_hours"],36)
   self.assertAlmostEqual(value["cumulative_same_task_GPU_hours"],value["actual_new_GPU_hours"]+prior,places=13)
   self.assertEqual(load(fixture/"STOP.json")["reason"],"GPU_BUDGET_PROTECTIVE_STOP")
 def test_39_real_CPU_child_exit_complete_artifacts_and_terminal_full_chain(self):
  a,b,c,kw=self.completion_fixture()
  artifacts={"result.json":a,"result.PROFILE.json":b,"environment.json":c}
  with CPUChild(self.root,artifacts=artifacts) as child:
   transition=material_fixture(child.identity,False)
   before,after=child.natural_finish_inside_sampling()
   value=self.check(transition,child.identity,before,after)
   self.assertEqual(value["lifecycle_state"],"NATURAL_EXIT_TRANSITION_NOT_RESOURCE_PASS")
   digests={name:sha(self.root/name) for name in artifacts}
   accepted=lifecycle.validate_completion_assets(load(self.root/"result.json"),load(self.root/"result.PROFILE.json"),load(self.root/"environment.json"),**kw)
   self.assertEqual(accepted["status"],"PASS_ARTIFACT_BINDING_ONLY_NOT_JOB_COMPLETE")
   terminal=lifecycle.assess_terminal_window([idle_fixture() for unused in range(3)],UUID,baseline(),assess_window_fn=g.assess_window)
   self.assertEqual(terminal["sample_count"],3);self.assertEqual(terminal["status"],"PASS")
   self.assertEqual(digests,{name:sha(self.root/name) for name in artifacts})
   CHILD_EVENTS[-1]["CPU_SYNTHETIC_completion_artifacts_SHA256"]=digests
   CHILD_EVENTS[-1]["terminal_fixture_count"]=3
 def test_40_missing_any_mandatory_NVML_field_after_exit_rejected(self):
  with CPUChild(self.root) as child:
   child.natural_finish_inside_sampling()
   for field in ["memory","memory_v2","utilization","temperature","power_mW","pstate","SM_clock_MHz"]:
    snapshot=idle_fixture();del snapshot["direct_NVML"][field]
    with self.subTest(field=field):self.assert_rejected(snapshot,child.identity)
 def test_41_early_SMI_capture_UNKNOWN_or_conflict_after_exit_rejected(self):
  with CPUChild(self.root) as child:
   child.natural_finish_inside_sampling()
   for kind in ["UNKNOWN","conflict"]:
    snapshot=idle_fixture()
    if kind=="UNKNOWN":snapshot["early_SMI_compute_processes"]["status"]="UNKNOWN"
    else:snapshot["early_SMI_compute_processes"]["processes"]=[dict(uuid=UUID,pid=2147483006,used_gpu_memory="1")]
    with self.subTest(kind=kind):self.assert_rejected(snapshot,child.identity)
 def test_42_each_source_foreign_has_durable_first_pin_after_child_exit(self):
  with CPUChild(self.root) as child:
   child.natural_finish_inside_sampling()
   for api in ["SMI_EARLY","NVML","SMI_XML"]:
    snapshot=add_pid(idle_fixture(),2147483007,api);raw=self.root/(api+".jsonl");pin=self.root/(api+".FAIL.json")
    predicate=lambda ss,u,o,b:lifecycle.assess_lifecycle_sample(ss,u,o,b,None,0,assess_fn=g.assess)
    with self.subTest(api=api),self.assertRaises(evidence.ResourceEvidenceError) as failure:
     evidence.assess_with_evidence(snapshot,UUID,child.identity,baseline(),raw_path=raw,pinned_path=pin,assess_fn=predicate)
    pinned=load(pin);self.assertEqual(pinned["forbidden_pids"],[2147483007])
    self.assertEqual(pinned["snapshot"],snapshot);ref=pinned["raw_reference"]
    raw_bytes=raw.read_bytes()[ref["offset"]:ref["offset"]+ref["bytes"]]
    self.assertEqual(hashlib.sha256(raw_bytes).hexdigest(),ref["line_sha256"])
    first_sha=sha(pin)
    with self.assertRaises(evidence.ResourceEvidenceError):evidence.assess_with_evidence(add_pid(idle_fixture(),2147483008,api),UUID,child.identity,baseline(),raw_path=raw,pinned_path=pin,assess_fn=predicate)
    self.assertEqual(sha(pin),first_sha)
 def test_43_no_model_or_CUDA_was_imported(self):
  self.assertNotIn("torch",sys.modules);self.assertNotIn("fla",sys.modules)

def main():
 (D/"audits").mkdir(exist_ok=True)
 stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromTestCase(LifecycleRegression)
 result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
 report=dict(status="PASS" if result.wasSuccessful() and not result.skipped else "FAIL",scope="CPU_LIFECYCLE_REGRESSION_ONLY",
  tests=result.testsRun,skips=len(result.skipped),failures=len(result.failures),errors=len(result.errors),log=stream.getvalue(),
  actual_CPU_child_lifecycle_events=CHILD_EVENTS,historical_failed_raw_replay=HISTORICAL_REPLAY,
  fixtures="All GPU telemetry and completion artifacts are CPU synthetic fixtures, not GPU/model evidence",
  mocks="read-only original PID resolver only for replay of an already exited historical owner; common.D only isolates CPU first-STOP test",
  actual_GPU_workers=0,numerical_qualification=False,science_trajectory_count=0,no_model_or_CUDA_import=True,
  source_SHA256={n:sha(W/n) for n in ["test_lifecycle_guard.py","lifecycle_guard.py","gpu_telemetry.py","gpu_admission.py","resource_evidence.py","cleanroom_run.py","cleanroom_common.py","cleanroom_finalize.py"]},
  prior_report_preserved=dict(path="audits/CPU_LIFECYCLE_REGRESSION_PRE_FORMAL2_SCHEMA.json",SHA256="1bca4bb01658f1cc7f180b35dba019bed67a8091833f4b4e1534630c54f5118a",status="PASS_CPU_ONLY_SUPERSEDED",reason="Independent prepare audit found frozen formal2 uses top-level gpu_uuid, no host; control helper and CPU fixture corrected before any GPU launch; final report binds corrected guard and true publish AST layout"),
  preparation_failure_preserved=dict(path="audits/CPU_LIFECYCLE_PREPARATION_01.json",SHA256=sha(D/"audits/CPU_LIFECYCLE_PREPARATION_01.json"),status="FAIL",reason="CPU test AST fixture incorrectly expected completion and terminal pure calls inline in Controller.job; production uses _completion_bindings and _terminal_gate methods; test corrected to inspect real call chain plus poll/sample ordering; no production predicate changed"),
  epoch=time.time())
 out=Path(sys.argv[1]) if len(sys.argv)>1 else D/"audits/CPU_LIFECYCLE_REGRESSION.json"
 assert out.parent.resolve()==(D/"audits").resolve(),"REPORT_ONLY_NEW_D_AUDITS"
 with out.open("x") as f:json.dump(report,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
 print(stream.getvalue(),flush=True);print(json.dumps(dict(status=report["status"],tests=result.testsRun,report=str(out),SHA256=sha(out))),flush=True)
 assert result.wasSuccessful() and not result.skipped
if __name__=="__main__":main()
