"""Synthetic durable library-receipt reducer tests; no engine or GPU imports."""
import copy,json,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
import cublas_compare as C
checks=[]
def ck(name,call,fail=False):
 try:call()
 except (RuntimeError,KeyError,FileNotFoundError):
  if not fail:raise
 else:
  if fail:raise AssertionError('Expected rejection: '+name)
 checks.append(dict(name=name,status='PASS'))
def fixture(root):
 identities=[dict(path='/synthetic/libtorch_cpu.so',bytes=3,sha256=C.sha_bytes(b'CPU'),role='torch')]
 digest=C.sha_bytes(C.canonical(identities));out=dict(runtime_library_identity=identities,runtime_library_digest=digest)
 for stage,key in [('PREFLIGHT','runtime_library_preflight'),('POSTFLIGHT','runtime_library_postflight')]:
  path=root/'library_identity'/(stage+'_MAPS.txt');path.parent.mkdir(exist_ok=True);path.write_bytes(b'SYNTHETIC_CPU_ONLY_MAPS\n')
  maps=dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=C.sha_bytes(path.read_bytes()))
  report=dict(schema='QWEN_LOADED_LIBRARY_IDENTITY_V2',stage=stage,status='PASS',identities=identities,identity_digest=digest,maps_receipt=maps,maps_final_receipt=maps,issues=[],missing_bindings=[],mismatches=[])
  ip=path.with_name(stage+'_IDENTITY.json');C.atomic_new(ip,report)
  out[key]=dict(report,identity_receipt=dict(path=str(ip.relative_to(root)),bytes=ip.stat().st_size,sha256=C.sha_bytes(ip.read_bytes())))
 return out
with tempfile.TemporaryDirectory() as d:
 root=Path(d);r=fixture(root);ck('complete_pre_and_post_receipts_accepted',lambda:C.verify_library_gates(root,r))
 for name,change in [
  ('missing_preflight',lambda x:x.pop('runtime_library_preflight')),
  ('failed_preflight',lambda x:x['runtime_library_preflight'].update(status='FAIL_CLOSED_STOP')),
  ('wrong_stage',lambda x:x['runtime_library_postflight'].update(stage='PREFLIGHT')),
  ('absolute_path',lambda x:x['runtime_library_preflight']['maps_receipt'].update(path='/tmp/foreign_maps')),
  ('parent_escape',lambda x:x['runtime_library_preflight']['maps_receipt'].update(path='../foreign_maps')),
  ('forged_semantic_digest',lambda x:x['runtime_library_preflight'].update(identity_digest='0'*64)),
  ('legacy_post_identity_mismatch',lambda x:x.update(runtime_library_identity=[])),
  ('legacy_post_digest_mismatch',lambda x:x.update(runtime_library_digest='0'*64))]:
  bad=copy.deepcopy(r);change(bad);ck(name,lambda:C.verify_library_gates(root,bad),True)
 mp=root/r['runtime_library_preflight']['maps_receipt']['path'];old=mp.read_bytes();mp.write_bytes(old+b'CHANGED')
 ck('changed_maps_SHA_bytes_rejected',lambda:C.verify_library_gates(root,r),True);mp.write_bytes(old)
 ip=root/r['runtime_library_preflight']['identity_receipt']['path'];old=ip.read_bytes();ip.write_bytes(old+b' ')
 ck('changed_identity_receipt_SHA_bytes_rejected',lambda:C.verify_library_gates(root,r),True);ip.write_bytes(old)
 report=json.loads(old);report['missing_bindings']=['/synthetic/libtorchaudio.so'];ip.write_bytes(C.canonical(report))
 bad=copy.deepcopy(r);bad['runtime_library_preflight']['identity_receipt'].update(bytes=ip.stat().st_size,sha256=C.sha_bytes(ip.read_bytes()))
 ck('rehashed_report_with_unresolved_missing_library_rejected',lambda:C.verify_library_gates(root,bad),True)
assert 'torch' not in sys.modules
out=dict(status='PASS_CPU_SYNTHETIC_LIBRARY_RECEIPTS_NOT_GPU_PROOF',checks=checks,GPU_dispatches=0,engine_imports=0)
root=Path(__file__).resolve().parents[1];(root/'qualification/LIBRARY_RECEIPTS_CPU_TESTS.json').write_text(json.dumps(out,indent=2)+'\n');print(out['status'],len(checks),'checks')
