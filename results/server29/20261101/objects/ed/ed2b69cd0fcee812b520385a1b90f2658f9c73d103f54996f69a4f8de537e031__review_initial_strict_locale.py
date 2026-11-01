"""Read-only CPU review of four actual CUDA qualifications and formal workers."""
import importlib.util,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'QWEN_GDN_PACE48_AIME26_SEED1_60_V1_4GPU_CONTINUE_CONTROL_V2_20261026'
s=importlib.util.spec_from_file_location('formal_review_transport',ROOT/'operations/remote.py');T=importlib.util.module_from_spec(s);s.loader.exec_module(T)
CODE=r'''
import sys,json,pathlib,os,stat
r=pathlib.Path('/data/zypan/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_4gpu_continue_control_v2_20261026');sys.path.insert(0,str(r/'src'));import pace_control as C
C.authorize(r);science=C.verify_science(r);mask=C.verify_head_mask(r);model=C.model_guard(r);external=C.read(r/'EXTERNAL_PRE_GPU_FRESH_SHA.json');assert external['status']=='PASS_FRESH_BOUND_EXTERNAL_SHA_BYTES' and external['includes_full_model17_SHA'] and external['files']==16244
ledger=C.read(r/'RESOURCE_LEDGER.json');ad=C.read(r/'FORMAL_ADMISSION.json');assert ad['status']=='PASS_ALL_FOUR_DEVICE_GATES_ADMIT_FIXED_REMAINING_ONCE' and ad['source_manifest_sha256']==C.sha(r/'source_manifest.json') and ad['science_freeze_sha256']==science
assert C.read(r/'INITIAL_REUSE_FRESH_VERIFICATION.json')['count']==15
native={};gates=[]
for gpu in range(4):
 unit='SCALE_REPAIR_GATES_GPU'+str(gpu);t=C.unit_target(r,'scale_repair_gates',unit);interval=C.read(t/'RESOURCE_INTERVAL.json');assert interval['returncode']==0 and interval['released'] and not C.same_process(interval['identity']);gate=C.verify_result(r,'scale_repair_gates',unit,interval['identity'])['result'];smoke=gate['paired_full_model_smoke'];assert set(smoke['configs'])=={'R2','R3'} and all(x['status']=='COMPLETE' and not x['nonfinite'] and x['runtime_error'] is None and x['head_mask_sha256']==mask['head_mask_sha256'] for x in smoke['configs'].values())
 assert smoke['actual_mask_gate']['status']=='PASS' and len(smoke['actual_mask_gate']['cases'])==48 and all(x['status']=='PASS' for x in smoke['actual_mask_gate']['cases'])
 B=gate['original_gates']['gates']['B']['native_first_chunk_subgate'];assert B['status']=='PASS' and C.sha(pathlib.Path(B['path']))==B['sha256'];chunk=C.read(B['path']);assert chunk['status']=='PASS' and chunk['fixture_count']==44 and chunk['token_comparisons']==440
 durable=C.read(t/'DURABLE29_RECEIPT.json');assert durable['status']=='PASS_DURABLE29_CLOSED_UNIT' and durable['actual_exit_returncode']==0 and durable['released']
 gates.append(dict(gpu=gpu,RESULT_sha256=C.sha(t/'RESULT.json'),interval_sha256=C.sha(t/'RESOURCE_INTERVAL.json'),native_first_chunk_sha256=B['sha256'],fixture_count=44,token_comparisons=440,actual_exit=0,released=True,durable29_manifest=durable['manifest_sha256']))
 for x in C.read(t/'POSTFLIGHT_LIBRARY.json')['selected']:native[x['path']]=x
allowed=C.bindings(r);allowed.update({str(pathlib.Path(x['path']).resolve()):x for x in C.read(r/'analysis/MODEL_TOKENIZER_FRESH_SHA.json')['files']});workers=[];unbound=[];fresh={};device_maps=[]
for unit,a in ledger['active_units'].items():
 assert a['stage']=='formal';i=a['identity'];assert C.same_process(i);p=pathlib.Path('/proc')/str(i['pid']);st=(p/'stat').read_text().rsplit(')',1)[1].split();assert int(st[1])==ledger['controller']['pid'];env=dict(x.split(b'=',1) for x in (p/'environ').read_bytes().split(b'\0') if b'=' in x);assert env[b'CUDA_VISIBLE_DEVICES'].decode()==a['gpu_uuid']==C.UUIDS[a['gpu']]
 t=C.unit_target(r,'formal',unit);driver=C.read(t/'DRIVER_IDENTITY.json');assert driver['actual_uuid']==driver['expected_uuid']==driver['CUDA_VISIBLE_DEVICES']==a['gpu_uuid'] and driver['logical_ordinal']==0
 proof=C.read(t/'LOADED_LIBRARY_1.json');assert proof['status']=='PASS_ACTUAL_LOADED_NATIVE_SHA_INODE';mapped={}
 for line in (p/'maps').read_text().splitlines():
  x=line.split(maxsplit=5)
  if len(x)!=6 or not x[5].startswith('/'):continue
  if x[5].startswith('/dev/'):
   device_maps.append(dict(pid=i['pid'],path=x[5],scope='DEVICE_OR_ANONYMOUS_SHARED_MEMORY_NOT_REGULAR_NATIVE_FILE'));continue
  path=str(pathlib.Path(x[5]).resolve());s=pathlib.Path(path).stat();assert stat.S_ISREG(s.st_mode) and int(x[4])==s.st_ino;expected=allowed.get(path)
  if expected is None:unbound.append(path);continue
  assert s.st_size==expected['bytes'];mapped[path]=dict(path=path,bytes=s.st_size,inode=s.st_ino,sha256=expected['sha256']);fresh[path]=mapped[path]
 progress=C.read(t/'PROGRESS.json');assert progress.get('input_forwards',0)>0 and progress['condition'] in ('R2','R3')
 workers.append(dict(unit=unit,gpu=a['gpu'],identity=i,PPID=int(st[1]),CVD=env[b'CUDA_VISIBLE_DEVICES'].decode(),DriverUUID=driver['actual_uuid'],actual_mapped_bound_files=len(mapped),loaded_library1_sha256=C.sha(t/'LOADED_LIBRARY_1.json'),progress=progress))
assert not unbound,sorted(set(unbound))
allfiles=dict(native);allfiles.update(fresh)
for path,x in allfiles.items():assert C.sha(pathlib.Path(path))==x['sha256'] and pathlib.Path(path).stat().st_size==x['bytes'] and pathlib.Path(path).stat().st_ino==x['inode']
assert len(workers)==4 and {x['gpu'] for x in workers}=={0,1,2,3}
obs=C.query_gpu()
for x in workers:assert any(a['pid']==x['identity']['pid'] and a['uuid']==x['DriverUUID'] for a in obs['apps']) and C.same_process(x['identity'])
assert not (r/'FIRST_FAILURE.json').exists()
print(json.dumps(dict(status='PASS_FOUR_GPU_FORMAL_START_FULL_INDEPENDENT_REVIEW',source_manifest_sha256=C.sha(r/'source_manifest.json'),permit_sha256=C.sha(r/'GPU_PERMIT.json'),source_files=C.read(r/'source_manifest.json')['count'],science_freeze_sha256=science,mask=mask,model17_stat=model,external_pre_GPU_full_SHA_proof=external,gates=gates,workers=workers,unique_native_and_mapped_files_fresh_SHA=len(allfiles),device_mmaps_excluded_from_regular_file_SHA=device_maps,NVML=obs,reused_formal=15,new_formal_planned=45,new_completed=ledger['current_formal_count'],answers_or_scores_read=False,GPU_calls=0,model_loads=0,signals=0,remote_mutations=0)))
'''

def main():
    v=T.run(CODE,role='55',python='/data/ydai/miniconda3/envs/bitdecode/bin/python',timeout=600)
    p=ROOT/'operations/FORMAL_START_FOUR_GPU_FULL_INDEPENDENT_REVIEW.json'
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps(dict(status=v['status'],source_files=v['source_files'],native=v['unique_native_and_mapped_files_fresh_SHA'],gates=len(v['gates']),workers=[dict(gpu=x['gpu'],unit=x['unit'],mapped=x['actual_mapped_bound_files'],generated=x['progress'].get('generated_tokens'),input_forwards=x['progress'].get('input_forwards')) for x in v['workers']],review_sha256=hashlib.sha256(p.read_bytes()).hexdigest()),ensure_ascii=False))

if __name__=='__main__':main()
