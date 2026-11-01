"""CPU fixture only, installed solely in temporary test roots."""
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
