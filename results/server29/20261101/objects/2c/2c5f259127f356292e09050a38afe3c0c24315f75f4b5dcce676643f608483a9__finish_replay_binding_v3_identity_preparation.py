"""Finish pre-freeze identity files after inspecting the relative production loader path."""
import ast,json
from pathlib import Path
import remote
P=remote.ROOT;L=P/'revisions/replay_binding_v3'
tree=ast.parse((P/'operations/apply_replay_binding_v3.py').read_text())
a=next(x for x in tree.body if isinstance(x,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='code'for t in x.targets))
ns={};exec(compile(ast.Module(body=[a],type_ignores=[]),'<CPU-preparation>','exec'),ns);original=ns['code']
prefix=original[:original.index("assert sha(O/'SOURCE_FREEZE.json')")].replace('assert not R.exists()',"assert R.exists()and not(R/'SOURCE_FREEZE.json').exists()and not(R/'GPU_RUN_STARTED.json').exists()and not(R/'ROOT.json').exists()")
stage=r'''
assert sha(O/'SOURCE_FREEZE.json')=='740d66ce7bacd5c3f45a79ca927579788a1d33f7854ece3ad73e217035514547'
assert sha(O/'FINAL_SEAL_RECEIPT.json')=='3436f902a60e97f30005de2ab6933a2d434096875287bd0a3615c0261e3810d5'
rebuild={'ROOT.json','HUMAN_REPAIR_AUTHORIZATION.json','SOURCE_MODEL_RUNTIME_FREEZE.json','GPU_UUID_AND_PERMISSION_AUDIT.json','GROUP64_CPU_CODEC_QUALIFICATION.json','REPAIR_IMPLEMENTATION_AND_VALIDATION.md'}
external={'analysis/EXTERNAL_NATIVE_BINDINGS.json','analysis/EXTERNAL_PYTHON_BINDINGS.json','analysis/MODEL_TOKENIZER_FRESH_SHA.json'}
changed={'protocol/non_aime_fixture.json','analysis/EXTERNAL_NATIVE_BINDINGS.json','analysis/EXTERNAL_PYTHON_BINDINGS.json'}
for x in read(O/'SOURCE_FREEZE.json')['files']:
 name=x['path']
 if name in changed:continue
 preserve=name in rebuild or(name.startswith('analysis/')and name not in external)
 q=R/('provenance/NATIVE_ADMISSION_V2/SOURCE_CONTROL/'+name if preserve else name)
 assert q.is_file()and q.stat().st_size==x['bytes']and sha(q)==x['sha256'],name
assert sha(R/'protocol/non_aime_fixture.json')=='0b4c355f01b00d5958879218138944a9a71db43dec1ddfe1ca3d596f6eb605b3'
assert (R/'src/canonical_h128.py').read_bytes()==(O/'src/canonical_h128.py').read_bytes()
assert 'Path(__file__).resolve().parents[1]/"runtime/transformers-qwen35/src"'in(R/'src/canonical_h128.py').read_text()
p=R/'src/execution_control.py';s=p.read_text();old='qwen_group64_four_codec_aime26_seed1_v1_native_admission_v2';assert s.count(old)==1;p.write_text(s.replace(old,'qwen_group64_four_codec_aime26_seed1_v1_replay_binding_v3'))
write(R/'operations/CPU_PREPARATION_RELATIVE_LOADER_PATH_ASSERTION.json',dict(status='PRE_FREEZE_CPU_ASSERTION_PRESERVED_NO_GPU',error='Expected absolute old loader path, but actual loader already uses __file__ relative runtime/src. No loader edit needed.',resolution='Verified exact canonical bytes unchanged and retained correct relative provider; isolated only temporary reference namespace',GPU_dispatches=0))
'''
code=prefix+stage+original[original.index("root=read(O/'ROOT.json')"):]
compile(code,'<remote-pre-freeze-finish>','exec')
r=remote.run(code,timeout=240);L.mkdir(parents=True)
for n,s in r.pop('files').items():
 p=L/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
(L/'operations').mkdir(exist_ok=True);(L/'monitoring').mkdir(exist_ok=True)
(L/'operations/remote.py').write_bytes((P/'operations/remote.py').read_bytes())
remote.save('operations/REPLAY_BINDING_V3_APPLIED.json',r);print(json.dumps(r))
