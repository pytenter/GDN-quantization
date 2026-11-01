"""Post-STOP frozen-source/dependency SHA audit, stdlib only, remote files stayremote."""
from pathlib import Path
import datetime,hashlib,json,os,sys,time
def main(cfg):
 root=Path(cfg['remote_root']);cache={};start=time.time()
 def read(p):return json.loads(p.read_text())
 def meta(s):return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
 def stable(p):
  p=Path(p);target=p.resolve(strict=True);before=meta(target.stat());key=tuple(before.values())
  if key not in cache:
   h=hashlib.sha256()
   with target.open('rb')as f:
    assert meta(os.fstat(f.fileno()))==before
    for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    assert meta(os.fstat(f.fileno()))==before
   cache[key]=h.hexdigest()
  after=meta(target.stat());assert before==after and p.resolve(strict=True)==target
  return dict(path=str(p),resolved_path=str(target),bytes=before['bytes'],sha256=cache[key],metadata_before=before,metadata_after=after)
 def rows(base,items):
  out=[]
  for row in items:
   path=Path(row['path']);path=path if path.is_absolute()else base/path
   try:a=stable(path);ok=a['bytes']==row['bytes']and a['sha256']==row['sha256'];out.append(dict(manifest_path=row['path'],expected_bytes=row['bytes'],expected_sha256=row['sha256'],pass_check=ok,actual=a))
   except Exception as exc:out.append(dict(manifest_path=row['path'],pass_check=False,error=type(exc).__name__+':'+str(exc)))
  return dict(count=len(out),passed=sum(x['pass_check']for x in out),rows=out)
 payload=read(root/'PAYLOAD_BUILD_MANIFEST.json');control=read(root/'CONTROL_SOURCE_MANIFEST.json');out=dict(task_id=cfg['task_id'],execution_run_id=cfg['execution_run_id'],audit_kind='POST_STOP_CURRENT_FROZEN_SOURCE_DEPENDENCY_IDENTITY',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),payload_manifest=stable(root/'PAYLOAD_BUILD_MANIFEST.json'),control_manifest=stable(root/'CONTROL_SOURCE_MANIFEST.json'),payload_files=rows(root,payload['files']),control_files=rows(root,control['files']),cards={},GPU_imports=False,GPU_queries=0,GPU_dispatches=0,raw_tensors_read=0,model_weights_downloaded_to_Mac=False,source_or_active_files_modified=False)
 out['payload_manifest']['expected_sha256']=cfg['payload_build_manifest_sha256'];out['control_manifest']['expected_sha256']=cfg['control_manifest_sha256'];external=[];library=[]
 for card in cfg['cards']:
  key=card['card_key'];child=Path(card['remote_root']);s=read(child/'SOURCE_MANIFEST.json');a=read(child/'AUTHORIZATION.json');o=read(child/'RUN_COPY_ORIGIN.json');c=read(child/'analysis/STRICT_BASELINE_CONTRACT.json')
  binds={}
  for k,name in [('source_manifest_sha256','SOURCE_MANIFEST.json'),('authorization_sha256','AUTHORIZATION.json'),('contract_sha256','analysis/STRICT_BASELINE_CONTRACT.json'),('preregistration_sha256','qualification/CUBLAS_PREREGISTRATION.json'),('archive_config_sha256','ARCHIVE_CONFIG.json')]:x=stable(child/name);binds[k]=dict(actual=x,expected=card[k],pass_check=x['sha256']==card[k])
  for k,name in [('run_copy_origin_sha256','RUN_COPY_ORIGIN.json'),('closure_discovery_sha256','CLOSURE_DISCOVERY_CPU_IMPORT.json'),('repair_provenance_sha256','REPAIR_PROVENANCE.json'),('repair_request_sha256','REPAIR_REQUEST_AND_PLAN.json'),('human_authorization_sha256','HUMAN_GPU_AUTHORIZATION.json'),('amendment_sha256','GPU_EXECUTION_AMENDMENT.json')]:x=stable(child/name);binds[k]=dict(actual=x,expected=a[k],pass_check=x['sha256']==a[k])
  out['cards'][key]=dict(manifest_bindings=binds,execution_files=rows(child,s['execution_sources']),device=c['execution_device_binding'],source_origin_device_consistent=(s['execution_run_id']==a['execution_run_id']==o['execution_run_id']==cfg['execution_run_id']and s['device']==a['device']==o['device']==c['execution_device_binding']and o['source_manifest_sha256']==a['source_manifest_sha256']))
  external.append(s['external_dependencies']);library.append(c['prospective_library_bindings'])
 out['all_external_rosters_equal']=all(x==external[0]for x in external);out['all_library_bindings_equal']=all(x==library[0]for x in library)
 out['fresh_external_dependencies']=rows(root,external[0]);out['fresh_prospective_library_paths']=rows(root,[x for x in library[0]if'path'in x and'sha256'in x]);out['model_weight_shards']=[x for x in out['fresh_external_dependencies']['rows']if x['manifest_path'].endswith('.safetensors')]
 good=(out['payload_manifest']['sha256']==cfg['payload_build_manifest_sha256']and out['control_manifest']['sha256']==cfg['control_manifest_sha256']and all(out[k]['count']==out[k]['passed']for k in ['payload_files','control_files','fresh_external_dependencies','fresh_prospective_library_paths'])and out['all_external_rosters_equal']and out['all_library_bindings_equal']and all(v['source_origin_device_consistent']and v['execution_files']['count']==v['execution_files']['passed']and all(x['pass_check']for x in v['manifest_bindings'].values())for v in out['cards'].values()))
 out.update(status='PASS_CURRENT_FROZEN_IDENTITIES_REAL_STOP_UNCHANGED'if good else 'CURRENT_SOURCE_IDENTITY_AUDIT_ATTENTION_REQUIRED',unique_physical_files_hashed=len(cache),elapsed_seconds=time.time()-start,current_disk_identity_not_retrospective_runtime_claim=True)
 target=root/'independent_delivery/source/FROZEN_SOURCE_DEPENDENCIES_POST_STOP.json';target.parent.mkdir(parents=True,exist_ok=True);encoded=json.dumps(out,sort_keys=True,ensure_ascii=False,indent=2).encode()+b'\n'
 with target.open('xb')as f:f.write(encoded);f.flush();os.fsync(f.fileno())
 sys.stdout.buffer.write(encoded)
