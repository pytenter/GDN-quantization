#!/usr/bin/env python3
"""Engineering-only actual CPU-import native identity gate; no GPU model."""
import contextlib,hashlib,io,json,os,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import library_identity as L

def save(name,value):
 raw=(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode();L.atomic_bytes_new(HERE/name,raw);return {'path':str(HERE/name),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
started=time.time();summary=None;error=None
try:
 namespace={'__name__':'_engineering_cpu_closure_probe','__file__':str(HERE/'cpu_import_probe.py')}
 with contextlib.redirect_stdout(io.StringIO())as captured:
  exec(compile((HERE/'cpu_import_probe.py').read_text(),str(HERE/'cpu_import_probe.py'),'exec'),namespace)
 observation=namespace['result'];save('CPU_IMPORT_OBSERVATION.json',observation)
 if observation['status']!='PASS_CPU_IMPORT_CLOSURE_DISCOVERY':raise RuntimeError('Exact CPU import probe did not PASS')
 contract=json.loads((HERE/'STRICT_CPU_GATE_CONTRACT.json').read_bytes())
 def progress(path,done):print(json.dumps({'phase':'PREFLIGHT_CPU_SHA','path':path,'bytes':done}),file=sys.stderr,flush=True)
 summary=L.library_identity_gate(contract,HERE/'gate','PREFLIGHT',progress=progress,context={'engineering_probe':'CPU_IMPORT_ONLY_NO_MODEL_CONSTRUCTOR_OR_GPU','old_STOP_retained':True,'prospective_union':'ALL_OLD_BINDINGS_PLUS_CLOSURE_DISCOVERY_PROSPECTIVE_ROWS'})
except BaseException as exc:
 error={'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()}
 if isinstance(exc,L.LibraryIdentityError):summary=exc.summary
status='PASS_ACTUAL_CPU_IMPORT_PREFLIGHT_ONLY'if error is None else'FAIL_CPU_ENGINEERING_PREFLIGHT'
out={'schema':'ACTUAL_CPU_IMPORT_LIBRARY_PREFLIGHT_V1','status':status,'started_unix':started,'finished_unix':time.time(),'pid':os.getpid(),'helper_sha256':hashlib.sha256((HERE/'library_identity.py').read_bytes()).hexdigest(),'worker_sources_not_executed':True,'contract_sha256':hashlib.sha256((HERE/'STRICT_CPU_GATE_CONTRACT.json').read_bytes()).hexdigest(),'closure_discovery_sha256':hashlib.sha256((HERE/'CLOSURE_DISCOVERY_CPU_IMPORT.json').read_bytes()).hexdigest(),'gate_summary':summary,'error':error,'no_GPU_calls':True,'no_model_constructor':True,'no_model_forward':True,'not_GPU_numerical_qualification':True,'new_GPU_root_not_staged_or_touched':True,'old_sealed_root_not_modified':True}
receipt=save('CPU_GATE_RESULT.json',out);print(json.dumps({'receipt':receipt,'status':status,'gate_status':summary['status']if summary else None,'selected_libraries':len(summary['identities'])if summary else 0,'error':error},ensure_ascii=False))
sys.exit(0 if error is None else 2)
