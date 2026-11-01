"""Small control-plane primitives; no model or numerical implementation."""
from pathlib import Path
import hashlib,json,os,sys,time,tempfile,fcntl,subprocess
W=Path(__file__).resolve().parent
D=Path(os.environ['LING_PROSPECTIVE_RESULTS']).resolve()
TASK='LING_KDA_P1_L7_4DOC_CLEANROOM_FINAL_V1'
PYTHON='/data/zypan/envs/ling-kda/bin/python'

def read(path):return json.loads(Path(path).read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def publish(path,value,replace=False):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.partial')
 try:
  with os.fdopen(fd,'w') as f:json.dump(value,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  if replace:os.replace(tmp,path)
  else:os.link(tmp,path);os.unlink(tmp)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def stop(reason,**kw):
 if not (D/'STOP.json').exists():
  try:publish(D/'STOP.json',dict(status='STOP',task=TASK,reason=str(reason),epoch=time.time(),no_retry=True,**kw))
  except FileExistsError:pass
 return read(D/'STOP.json')
def event(state,**kw):
 row=dict(state=state,epoch=time.time(),**kw)
 with (D/'events.jsonl').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
 publish(D/'CURRENT_STATE.json',row,replace=True)
 print(json.dumps(row,ensure_ascii=False),flush=True)
def environment(uuid):
 from frozen_runtime_profile import profile_environment
 cfg=read(D/'EXECUTION_ROUTING.json');env=dict(os.environ)
 env.update(PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',LING_GATED_RMSNORM_FIXED_CONFIG='BT16_W8',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=str(cfg['authorized_devices'][uuid]),LING_TARGET_GPU_UUID=uuid,LING_PROSPECTIVE_RESULTS=str(D))
 env.update(profile_environment());env.pop('LING_EXECUTION_LANE',None)
 return env

def compare_exact(parent,fresh,uuid):
 assert parent['status']==fresh['status']=='PASS'
 assert parent['begin']==0 and fresh['begin']==8192 and parent['endpoint']==fresh['endpoint']==8256
 assert parent['observer_enabled'] is True and fresh['observer_enabled'] is False
 assert parent['document']==fresh['document'] and parent['condition']==fresh['condition']
 assert parent['host']['gpu_uuid']==fresh['host']['gpu_uuid']==uuid
 fields=['uninterrupted_suffix_full_logit_sha256','final_full_native_cache_sha256','audit_state_fingerprint','checkpoint_sha256']
 assert all(parent[x]==fresh[x] for x in fields),'EXACT_FRESH64_NUMERICAL_FAILURE'
 for key in ['kernel','driver','torch','cuda_runtime','selector']:assert parent['host'][key]==fresh['host'][key]
 if parent['condition']!='FP':
  for j in [parent,fresh]:
   q=j['integrity'];assert q['passed'] and q['every_token_layer_qdq_count']==8256*18 and q['actual_next_kernel_equal_post_qdq_count']==8255*18
 return True
