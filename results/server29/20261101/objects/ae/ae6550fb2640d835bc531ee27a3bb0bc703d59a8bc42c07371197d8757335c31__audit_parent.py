"""Read-only full content audit of the sealed pilot and its actual dependencies.

This new task's pre/post-run identity receipt never loads Torch or a model.
The parent seal, all parent tensors, frozen sources, runtime, tokenizer and
weights are streamed, never changed. Outputs are new-only in the new root.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

TASK = 'QWEN_GDN_DUAL_AXIS_PROSPECTIVE_REPLICATION_5DOC_V1'
PARENT_MANIFEST = 'b422ddf8359115448205b6afd2175e5d727446a02beb7e11281a6ac37f53c73a'
PARENT_RECEIPT = '94a5971c226e38647a107ad53070ad225c520606c94b6d8925ce5e1fe7c6c4a0'
PARENT_SOURCE = 'eaf0395728a0c57efbdbc28cb6d062803896c8c132cc4c5dd35687267f6eb1d4'
SCIENCE = dict(core='aa338b8b83cde62b97c94fd205d1c62999f783e7ec3242cbb813dc877e01f575',
 codec='ece859b901145c6d9d646b8befd01916037c9117ef4a80f7b801f27f79f8c23f',
 worker='e04ec85092ee4f2207699366b2aaa61e1e71038e728c9234477b5b2c21794773',
 reducer='7e94c95803acb7ae47f0e3ec44fa97faf7cc8d93c4ecc5791a56ea2cf305b84f',
 coordinator='28647e7d03b9d64f88c8e882bc27ae296053187874ee8431036dfeddbd99e065')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
 return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())
def need(ok,s):
 if not ok:raise ValueError(s)

def verify(p,expected,size=None):
 p=Path(p);a=p.stat();digest=sha(p);b=p.stat()
 need((a.st_size,a.st_mtime_ns,a.st_ino,a.st_dev)==(b.st_size,b.st_mtime_ns,b.st_ino,b.st_dev),'Changed during hash: '+str(p))
 need(digest==expected and (size is None or b.st_size==size),'Content mismatch: '+str(p))
 return dict(path=str(p),bytes=b.st_size,sha256=digest,fresh_stream_hash=True,unchanged_during_hash=True)

def run(root,parent,stage):
 start=time.time();root=Path(root);parent=Path(parent)
 need(root.name==TASK and root.resolve()!=parent.resolve(),'Wrong independent task root')
 output=root/('qualification_cpu/PARENT_IDENTITY_'+stage+'.json')
 need(not output.exists(),'Refuse overwrite of identity evidence')
 result=dict(task_id=TASK,status='FAIL',stage=stage,GPU_work=False,torch_imported=False,started_unix=start)
 try:
  seal=parent/'hashes/artifact_sha256.txt';receipt=parent/'hashes/DELIVERY_VERIFICATION.json'
  verify(seal,PARENT_MANIFEST);verify(receipt,PARENT_RECEIPT)
  delivery=read(receipt);need(delivery['status']=='PASS','Parent delivery was not PASS')
  lines=[x.split('  ',1) for x in seal.read_text().splitlines()]
  need(len(lines)==len({p for h,p in lines})==289,'Expected289 parent payloads')
  assets=[]
  for h,rel in lines:
   p=parent/rel;need(p.resolve().is_relative_to(parent.resolve()),'Parent entry escapes root')
   assets.append(verify(p,h))
  verify(parent/'source_manifest.json',PARENT_SOURCE)
  source=read(parent/'source_manifest.json')
  for e in source['execution_sources']:verify(parent/e['path'],e['sha256'],e['bytes'])
  for name,h in SCIENCE.items():
   verify(parent/('src/'+name+'.py'),h);verify(root/('src/'+name+'.py'),h)
  gate=read(parent/'GATE_STATUS_FINAL.json');ledger=read(parent/'gpu_resource_ledger.json')
  need(gate['status']=='COMPLETE_AND_STOP' and gate['GPU_release_verified'] is True,'Parent completion/release identity missing')
  need(ledger['GPU_HOURS_USED']==8.322145025332768 and ledger['final_GPU_release_verified'] is True,'Parent resource identity changed')
  initial=read(parent/'inherited_initial/FIRST_FAILURE.json')
  need('BrokenPipeError' in initial['error'],'Original BrokenPipe failure was not preserved')
  deps=read(parent/'analysis/FINAL_DEPENDENCY_VERIFICATION.json')
  need(deps['status']=='PASS','Parent final dependencies were not qualified')
  runtime=[verify(x['path'],x['sha256'],x['bytes']) for x in deps['runtime_source_files']]
  expected=read(parent/'qualification_cpu/MODEL_CONTENT_IDENTITY.json')
  weights=[verify(x['path'],x['sha256'],x['bytes']) for x in expected['model_weight_and_config_current_content_hashes']]
  need(len(weights)==14,'Model/tokenizer identity count changed')
  result.update(status='PASS',parent_root=str(parent),parent_manifest_sha256=PARENT_MANIFEST,parent_receipt_sha256=PARENT_RECEIPT,
   parent_source_manifest_sha256=PARENT_SOURCE,parent_payloads=assets,parent_frozen_source_files=len(source['execution_sources']),
   runtime_source_files=runtime,model_weight_and_config_current_content_hashes=weights,
   files_verified=len(weights),runtime_files_verified=len(runtime),science_sources=SCIENCE,
   old_FP_NATIVE_H_failure='PRESERVED_NOT_RECLASSIFIED',old_stdout_failure='PRESERVED_WITH_AUTHORIZED_COMPLETED_CONTINUATION',
   qualified_structure_and_codec_reused_by_exact_source_and_dependency_identity=True,
   fresh_model_or_GPU_forward=False)
 except Exception as exc:
  import traceback
  result.update(first_failure=dict(error=repr(exc),traceback=traceback.format_exc()))
 result.update(verified_unix=time.time(),CPU_wall_seconds=time.time()-start,script_sha256=sha(__file__))
 output.parent.mkdir(parents=True,exist_ok=True)
 with output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(status=result['status'],stage=stage,output=str(output),CPU_wall_seconds=result['CPU_wall_seconds'])),flush=True)
 need(result['status']=='PASS','Identity failed; inspect preserved receipt and STOP')
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--parent',required=True);p.add_argument('--stage',choices=['PRERUN','POSTRUN'],required=True);a=p.parse_args();run(a.root,a.parent,a.stage)
