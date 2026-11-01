"""Finish interrupted CPU file preparation, not a GPU/worker/finisher restart."""
import ast, base64, json
from pathlib import Path
import remote
P=remote.ROOT;L=P/'revisions/replay_binding_v3'
tree=ast.parse((P/'operations/apply_replay_binding_v3.py').read_text())
assignment=next(x for x in tree.body if isinstance(x,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='code'for t in x.targets))
ns={};exec(compile(ast.Module(body=[assignment],type_ignores=[]),'<original-CPU-preparation-code>','exec'),ns)
original=ns['code']
prefix=original[:original.index("assert sha(O/'SOURCE_FREEZE.json')")].replace('assert not R.exists()',"assert R.exists()and not(R/'SOURCE_FREEZE.json').exists()and not(R/'GPU_RUN_STARTED.json').exists()")
audit=(P/'revisions/native_admission_v2/operations/FINAL_INDEPENDENT_READONLY_VERIFICATION.json').read_bytes()
extra="""
assert sha(O/'SOURCE_FREEZE.json')=='740d66ce7bacd5c3f45a79ca927579788a1d33f7854ece3ad73e217035514547'
assert sha(O/'FINAL_SEAL_RECEIPT.json')=='3436f902a60e97f30005de2ab6933a2d434096875287bd0a3615c0261e3810d5'
rebuild={'ROOT.json','HUMAN_REPAIR_AUTHORIZATION.json','SOURCE_MODEL_RUNTIME_FREEZE.json','GPU_UUID_AND_PERMISSION_AUDIT.json','GROUP64_CPU_CODEC_QUALIFICATION.json','REPAIR_IMPLEMENTATION_AND_VALIDATION.md'}
external={'analysis/EXTERNAL_NATIVE_BINDINGS.json','analysis/EXTERNAL_PYTHON_BINDINGS.json','analysis/MODEL_TOKENIZER_FRESH_SHA.json'}
for x in read(O/'SOURCE_FREEZE.json')['files']:
 name=x['path'];preserve=name in rebuild or(name.startswith('analysis/')and name not in external)
 q=R/('provenance/NATIVE_ADMISSION_V2/SOURCE_CONTROL/'+name if preserve else name)
 assert q.is_file()and q.stat().st_size==x['bytes']and sha(q)==x['sha256'],name
(R/'operations').mkdir(exist_ok=True)
write(R/'operations/CPU_PREPARATION_LOCAL_ONLY_AUDIT_COPY_FAILURE.json',dict(status='PRE_FREEZE_CPU_FILE_PREPARATION_FAILURE_PRESERVED',error='FileNotFoundError: prior FINAL_INDEPENDENT_READONLY_VERIFICATION.json is Mac-only, absent in remote prior operations',GPU_dispatches=0,original_apply_script_unmodified=True,resolution='Copy only exact SHA-bound Mac audit into new provenance; no old remote source/seal edit; resume CPU preparation only'))
import base64
audit=base64.b64decode(AUDIT)
assert hashlib.sha256(audit).hexdigest()=='d6d2140a3ac0e72e3a8e34f819a127615874c4abea45289cc2cd2395de633d6d'
""".replace('AUDIT',repr(base64.b64encode(audit).decode()))
suffix=original[original.index("for n in ['SOURCE_FREEZE.json','GPU_PERMIT.json'"):]
suffix=suffix.replace('shutil.copy2(O/n,q)',"(q.write_bytes(audit) if n=='operations/FINAL_INDEPENDENT_READONLY_VERIFICATION.json' else shutil.copy2(O/n,q))")
result=remote.run(prefix+extra+suffix,timeout=240)
L.mkdir(parents=True)
for n,s in result.pop('files').items():
 p=L/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
(L/'operations').mkdir(exist_ok=True);(L/'monitoring').mkdir(exist_ok=True)
(L/'operations/remote.py').write_bytes((P/'operations/remote.py').read_bytes())
remote.save('operations/REPLAY_BINDING_V3_APPLIED.json',result)
print(json.dumps(result))
