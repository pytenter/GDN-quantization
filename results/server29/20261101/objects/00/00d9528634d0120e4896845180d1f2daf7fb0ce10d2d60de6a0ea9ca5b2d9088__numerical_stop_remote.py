"""Post-STOP CPU diagnosis. Stream only failed token pair on55; never GPU."""
import base64,datetime,hashlib,importlib.util,json,math,os,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def main(cfg):
 root=Path(cfg['remote_root']);proofs={};checks=[]
 def read(path,keep=True):
  raw=path.read_bytes()
  if keep:
   rel=str(path.relative_to(root));proofs[rel]=dict(path=rel,bytes=len(raw),sha256=sha(raw),data_base64=base64.b64encode(raw).decode())
  return json.loads(raw)
 def check(label,ok):checks.append(dict(check=label,pass_check=bool(ok)))
 def bound(directory,row):
  raw=(directory/row['path']).read_bytes();return len(raw)==row['bytes']and sha(raw)==row['sha256']
 failure=read(root/'FIRST_FAILURE.json');session=read(root/'SESSION_RESULT.json')
 cards={}
 # Import exact frozen CPU reducer only. Neither it nor archive_io imports GPU packages.
 script=root/'gpu3/scripts/cublas_compare.py';check('frozen_compare_source_SHA',sha(script.read_bytes())=='f70d0e61f3252ffdc33c6282e8e630c668cbb0c47985e447d62f43686f0903f7')
 sys.path.insert(0,str(script.parent));spec=importlib.util.spec_from_file_location('frozen_failure_compare',script);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 for card in cfg['cards']:
  key=card['card_key'];child=root/key;auth=read(child/'AUTHORIZATION.json');source=read(child/'SOURCE_MANIFEST.json');contract=read(child/'analysis/STRICT_BASELINE_CONTRACT.json');read(child/'RUN_COPY_ORIGIN.json');read(child/'FIRST_FAILURE.json');units=[]
  check(key+'_auth_source_frozen',sha((child/'AUTHORIZATION.json').read_bytes())==card['authorization_sha256']and sha((child/'SOURCE_MANIFEST.json').read_bytes())==card['source_manifest_sha256']and sha((child/'analysis/STRICT_BASELINE_CONTRACT.json').read_bytes())==card['contract_sha256'])
  previous=None
  for u in ('B0_FP_Q0','B0_FP_Q1'):
   p=child/'execution'/u;r=read(p/'RESULT.json');start=read(p/'STARTED.json');ctrl=read(p/'CONTROLLER_RECEIPT.json');index=read(p/'numeric/NUMERICAL_INDEX.json');read(p/'ENVIRONMENT.json')
   check(key+'_'+u+'_worker_controller_complete_PASS',r['status']=='PASS'and ctrl['status']=='PASS'and ctrl['returncode']==0 and ctrl['result_sha256']==sha((p/'RESULT.json').read_bytes())and r['process_identity']==start['process_identity']and r['consumed_tokens']==256 and r['all_tokens_finite']is True)
   check(key+'_'+u+'_identity_gpu_source_exact',all(r[k]==card[k]for k in ['authorization_sha256','source_manifest_sha256','contract_sha256','preregistration_sha256'])and r['GPU_UUID']==auth['device']['uuid'])
   check(key+'_'+u+'_complete_index_bound',bound(p,r['numerical_index'])and index['tokens']==256 and len(index['token_receipts'])==256 and index['status']=='PASS'and index['reference']==(u=='B0_FP_Q0')and index['actual_bytes_compared']==(u=='B0_FP_Q1'))
   mod.verify_library_gates(p,r)
   for field in ['runtime_library_preflight','runtime_library_postflight']:
    for kind in ['maps_receipt','maps_final_receipt','identity_receipt']:readpath=p/r[field][kind]['path'];raw=readpath.read_bytes();rel=str(readpath.relative_to(root));proofs[rel]=dict(path=rel,bytes=len(raw),sha256=sha(raw),data_base64=base64.b64encode(raw).decode())
   aggregate=hashlib.sha256();aggregate.update(b'[');rows=[]
   for t,receipt in enumerate(index['token_receipts']):
    raw=(p/'numeric'/receipt['path']).read_bytes();x=json.loads(raw)
    if t:aggregate.update(b',')
    aggregate.update(canonical([x['token'],x['metadata'],x['tensors']]))
    rows.append(dict(token=t,sha256=sha(raw),index_bound=(receipt['token']==x['token']==t and sha(raw)==receipt['sha256']and len(raw)==receipt['bytes']),byte_equal=x['byte_equal'],actual_bytes_compared=x['actual_bytes_compared'],mismatched_names=x.get('mismatched_names',[])))
   aggregate.update(b']');check(key+'_'+u+'_all256_committed_metadata_consistent',all(x['index_bound']and x['byte_equal']is True and x['actual_bytes_compared']==(u=='B0_FP_Q1')and not x['mismatched_names']for x in rows)and aggregate.hexdigest()==index['all_tensor_digest'])
   if previous is not None:check(key+'_B0_fresh_process_distinct_same_descriptor_digest',previous['process_identity']!=r['process_identity']and previous['all_tensor_digest']==index['all_tensor_digest'])
   previous=dict(process_identity=r['process_identity'],all_tensor_digest=index['all_tensor_digest'])
   units.append(dict(unit_id=u,status='PASS',actual_bytes_compared_across_fresh=(u=='B0_FP_Q1'),process_identity=r['process_identity'],all_tensor_digest=index['all_tensor_digest'],runtime_library_digest=r['runtime_library_digest'],tokens=256,metadata_rows=rows))
  b1=child/'execution/B1_FP_Q0';state=dict(STARTED_present=(b1/'STARTED.json').exists(),RESULT_present=(b1/'RESULT.json').exists(),committed_token_receipts=len(list((b1/'numeric').glob('token_*.json'))),EARLIEST_NUMERICAL_FAILURE_present=(b1/'EARLIEST_NUMERICAL_FAILURE.json').exists())
  for name in ['STARTED.json','RESULT.json','DISPATCH.json','EARLIEST_NUMERICAL_FAILURE.json']:
   if(b1/name).exists():read(b1/name)
  cards[key]=dict(device=auth['device'],B0_completed_units=units,B1_status=state)
 refpath=root/'gpu3/execution/B0_FP_Q0/numeric/token_000.json';curpath=root/'gpu3/execution/B1_FP_Q0/numeric/token_000.json';ref=read(refpath);cur=read(curpath);earliest=read(root/'gpu3/execution/B1_FP_Q0/EARLIEST_NUMERICAL_FAILURE.json')
 check('recorded_earliest_failure_token0',earliest['token']==cur['token']==ref['token']==0 and cur['actual_bytes_compared']is True and cur['byte_equal']is False)
 # First pass is the unchanged frozen compare_failure, including immutable raw
 # stream SHA/bytes validation through frozen archive_io. No final catalog write.
 started=time.time();frozen=mod.compare_failure(refpath,curpath)
 # Second bounded pass: finite check all tensors and first differing scalar.
 finite=[];first_scalar=None;bytes_streamed=0
 with mod.open_receipt_raw(refpath.parent,ref)as a,mod.open_receipt_raw(curpath.parent,cur,True)as b:
  for r,c in zip(ref['tensors'],cur['tensors']):
   aa=a.read(r['bytes']);bb=b.read(c['bytes']);bytes_streamed+=len(aa)+len(bb);same_roster=all(r[k]==c[k]for k in ['name','dtype','shape','stride','bytes','offset'])
   for label,raw,row in [('reference',aa,r),('actual',bb,c)]:
    isfloat=row['dtype']in ['torch.bfloat16','torch.float16','torch.float32','torch.float64'];bad=sum(not math.isfinite(float(v))for v in mod.numeric_values(raw,row['dtype']))if isfloat else 0
    finite.append(dict(name=row['name'],side=label,nonfinite_elements=bad,bytes=len(raw),tensor_SHA_matches_metadata=sha(raw)==row['sha256']))
   if aa!=bb and first_scalar is None and same_roster:
    byte=next(i for i,(x,y)in enumerate(zip(aa,bb))if x!=y);nbytes={'torch.bfloat16':2,'torch.float16':2,'torch.float32':4,'torch.float64':8,'torch.int64':8,'torch.int32':4,'torch.int8':1,'torch.uint8':1,'torch.bool':1}[r['dtype']];index=byte//nbytes;left=next(mod.numeric_values(aa[index*nbytes:(index+1)*nbytes],r['dtype']));right=next(mod.numeric_values(bb[index*nbytes:(index+1)*nbytes],c['dtype']))
    first_scalar=dict(name=r['name'],tensor_ordinal=ref['tensors'].index(r),dtype=r['dtype'],shape=r['shape'],first_mismatched_byte_within_tensor=byte,first_mismatched_byte_in_raw=r['offset']+byte,flat_element_index=index,reference_value=left,actual_value=right,reference_element_hex=aa[index*nbytes:(index+1)*nbytes].hex(),actual_element_hex=bb[index*nbytes:(index+1)*nbytes].hex())
  for stream,total in [(a,ref['total_bytes']),(b,cur['total_bytes'])]:
   while stream.tell()<total:assert stream.read(min(1048576,total-stream.tell()))
   assert not stream.read(1)
 def fields(a,b,path='metadata'):
  if type(a)!=type(b):return [dict(path=path,reference=a,actual=b)]
  if isinstance(a,dict):
   out=[]
   for k in sorted(set(a)|set(b)):
    if k not in a or k not in b:out.append(dict(path=path+'/'+str(k),reference=a.get(k),actual=b.get(k)))
    else:out+=fields(a[k],b[k],path+'/'+str(k))
   return out
  if isinstance(a,list):
   if len(a)!=len(b):return [dict(path=path+'/length',reference=len(a),actual=len(b))]
   out=[]
   for i,(x,y)in enumerate(zip(a,b)):out+=fields(x,y,path+'/'+str(i))
   return out
  return []if a==b else [dict(path=path,reference=a,actual=b)]
 md=fields(ref['metadata'],cur['metadata']);check('all_pair_tensors_finite_and_SHA_bound',all(x['nonfinite_elements']==0 and x['tensor_SHA_matches_metadata']for x in finite))
 check('no_GPU_capable_module_imported',not any(x=='torch'or x.startswith('torch.')or x=='transformers'or x.startswith('transformers.')for x in sys.modules))
 out=dict(status='CPU_DIAGNOSIS_COMPLETE_REAL_FAIL_CLOSED_STOP_UNCHANGED'if all(x['pass_check']for x in checks)else 'CPU_AUDIT_ATTENTION_REQUIRED',task_id=cfg['task_id'],execution_run_id=cfg['execution_run_id'],created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),failure=failure,session_status=session['status'],cards=cards,frozen_compare_failure=frozen,earliest_actual_different_scalar=first_scalar,metadata_field_differences=md,metadata_input_identity_equal=all(ref['metadata'][k]==cur['metadata'][k]for k in ['token','input_id','next_input_id','cache_cursor','codec','GDN_layers']),tensor_roster_equal=all({k:r[k]for k in ['name','dtype','shape','stride','offset','bytes']}=={k:c[k]for k in ['name','dtype','shape','stride','offset','bytes']}for r,c in zip(ref['tensors'],cur['tensors']))and len(ref['tensors'])==len(cur['tensors']),pair_finite_and_tensor_SHA_checks=finite,pair_stream_passes=2,second_pass_streamed_bytes=bytes_streamed,diagnosis_elapsed_seconds=time.time()-started,checks=checks,checks_passed=sum(x['pass_check']for x in checks),GPU_imports=False,GPU_queries=False,GPU_dispatches=0,raw_download_to_Mac=False,all_raw_rehashed=False,archive_catalog_published=False,frozen_compare_source_SHA256=sha(script.read_bytes()),limitations=['Eight completed B0 units establish only same-card B0 repeatability on these256 inputs.','B1 other-card missing committed receipts are unobserved comparisons, not numerical PASS or FAIL.','Only this failed token pair raw was streamed twice on55; final full archive audit is separate.','No performance/H_C128/Dual-M/long-context conclusion or repair/retry is inferred.'])
 supplemental=root/'independent_delivery/source';supplemental.mkdir(parents=True,exist_ok=True);encoded=json.dumps(out,sort_keys=True,ensure_ascii=False,indent=2).encode()+b'\n'
 with(supplemental/'NUMERICAL_FAILURE_CPU_AUDIT.json').open('xb')as f:f.write(encoded);f.flush();os.fsync(f.fileno())
 print(json.dumps(dict(report=out,proofs=list(proofs.values())),sort_keys=True,separators=(',',':')))
