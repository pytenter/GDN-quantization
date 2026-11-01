#!/usr/bin/env python3
"""CPU-only raw-byte qualification reducer. Never imports torch/CUDA."""
import argparse,hashlib,json,math,os,struct
from pathlib import Path
from archive_io import store_for_unit,open_receipt_raw,archive_progress
TASK='QWEN_GDN_FAST_RUNTIME_QUALIFICATION_V1'
def require(ok,message):
 if not ok:raise RuntimeError(message)
def sha_bytes(x):return hashlib.sha256(x).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def atomic_new(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);raw=canonical(obj)+b'\n';tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
 with tmp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 try:os.link(tmp,p)
 finally:tmp.unlink()
 directory=os.open(p.parent,os.O_RDONLY)
 try:os.fsync(directory)
 finally:os.close(directory)
def read(path):return json.loads(Path(path).read_text())
def numeric_values(raw,dtype):
 fmt={'torch.float32':'f','torch.float64':'d','torch.float16':'e','torch.int8':'b','torch.uint8':'B','torch.int16':'h','torch.int32':'i','torch.int64':'q','torch.bool':'?'}
 if dtype=='torch.bfloat16':
  return (struct.unpack('<f',struct.pack('<I',x[0]<<16))[0] for x in struct.iter_unpack('<H',raw))
 require(dtype in fmt,'Unsupported diff dtype:'+dtype)
 return (x[0] for x in struct.iter_unpack('<'+fmt[dtype],raw))
def difference(a,b,dtype,shape):
 require(len(a)==len(b),'Difference byte length mismatch');first=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None)
 maxabs=maxrel=0.;nonfinite=[];n=0
 for n,(x,y) in enumerate(zip(numeric_values(a,dtype),numeric_values(b,dtype)),1):
  x=float(x);y=float(y)
  if not(math.isfinite(x) and math.isfinite(y)):
   if len(nonfinite)<16:nonfinite.append(n-1)
   continue
  delta=abs(y-x);maxabs=max(maxabs,delta);maxrel=max(maxrel,delta/max(abs(x),1e-300))
 return dict(byte_equal=a==b,first_mismatched_byte=first,elements=n,max_abs_error='UNDEFINED_NONFINITE' if nonfinite else ('Infinity' if math.isinf(maxabs) else maxabs),max_relative_error='UNDEFINED_NONFINITE' if nonfinite else ('Infinity' if math.isinf(maxrel) else maxrel),finite_pairs_max_abs_error='Infinity' if math.isinf(maxabs) else maxabs,relative_denominator='max(abs(reference),1e-300)',nonfinite_flat_indices_first16=nonfinite,shape=shape,dtype=dtype)
class SnapshotStore:
 """One immutable full reference; subsequent fresh runs compare every actual byte."""
 def __init__(self,unit_dir,reference_dir=None):
  self.root=Path(unit_dir);self.root.mkdir(parents=True,exist_ok=True);self.reference=Path(reference_dir) if reference_dir else None;self.rows=[];self.archive=store_for_unit(self.root)
 def record(self,token,tensors,metadata):
  # tensors=(name,dtype,shape,stride,rawbytes); order and tensor roster are strict.
  require(token==len(self.rows),'Nonsequential token snapshot');rows=[];offset=0
  for name,dtype,shape,stride,raw in tensors:
   require(type(raw) is bytes,'Snapshot raw bytes required');rows.append(dict(name=name,dtype=dtype,shape=list(shape),stride=list(stride),offset=offset,bytes=len(raw),sha256=sha_bytes(raw)));offset+=len(raw)
  require(len({x['name'] for x in rows})==len(rows),'Duplicate snapshot tensor names')
  item=dict(token=token,metadata=metadata,tensors=rows,total_bytes=offset,actual_bytes_compared=False)
  stem=f'token_{token:03d}';path=self.root/(stem+'.bin')
  if self.reference is None:
   if self.archive is not None:
    receipt=self.archive.put(path.name,[t[-1] for t in tensors]);item['raw_archive']=receipt;archive_progress(self.root,token,receipt,'FULL_REFERENCE')
   else:
    with path.open('xb') as f:
     for t in tensors:f.write(t[-1])
     f.flush();os.fsync(f.fileno())
    item['raw_file']=path.name
   item.update(role='FULL_REFERENCE',byte_equal=True)
  else:
   ref=read(self.reference/(stem+'.json'));keys=lambda rs:[{k:x[k] for k in ['name','dtype','shape','stride','offset','bytes']} for x in rs]
   bad=[]
   if ref['metadata']!=metadata or keys(ref['tensors'])!=keys(rows):bad.append('METADATA_OR_TENSOR_ROSTER')
   with open_receipt_raw(self.reference,ref) as f:
    for j,tr in enumerate(ref['tensors']):
     require(f.tell()==tr['offset'],'Reference tensor offset changed');expected=f.read(tr['bytes'])
     require(len(expected)==tr['bytes'] and sha_bytes(expected)==tr['sha256'],'Reference raw tensor SHA changed:'+tr['name'])
     if j<len(tensors) and expected!=tensors[j][-1]:bad.append(rows[j]['name'])
    if f.read(1):bad.append('REFERENCE_TRAILING_BYTES')
   item.update(role='FRESH_ACTUAL_BYTE_COMPARISON',reference=str(self.reference),actual_bytes_compared=True,byte_equal=not bad,mismatched_names=bad)
   if bad:
    if self.archive is not None:
     receipt=self.archive.put(path.name,[t[-1] for t in tensors]);item['failure_raw_archive']=receipt;archive_progress(self.root,token,receipt,'FIRST_DIFFERENCE')
    else:
     with path.open('xb') as f:
      for t in tensors:f.write(t[-1])
      f.flush();os.fsync(f.fileno())
     item['failure_raw_file']=path.name
  atomic_new(self.root/(stem+'.json'),item);self.rows.append(item)
  return item
 def finish(self):
  require(len(self.rows)==256,'Incomplete numerical token roster');require(all(x['byte_equal'] for x in self.rows),'Snapshot mismatch')
  index=dict(status='PASS',task_id=TASK,tokens=256,reference=self.reference is None,reference_directory=str(self.reference) if self.reference else None,actual_bytes_compared=self.reference is not None,all_tensor_digest=sha_bytes(canonical([[x['token'],x['metadata'],x['tensors']] for x in self.rows])),token_receipts=[dict(token=x['token'],path=f'token_{x["token"]:03d}.json',sha256=sha_bytes((self.root/f'token_{x["token"]:03d}.json').read_bytes()),bytes=(self.root/f'token_{x["token"]:03d}.json').stat().st_size) for x in self.rows])
  atomic_new(self.root/'NUMERICAL_INDEX.json',index);return index

def compare_failure(reference,actual):
 ref=read(reference);cur=read(actual);out=[]
 with open_receipt_raw(Path(reference).parent,ref) as a,open_receipt_raw(Path(actual).parent,cur,True) as b:
  for r,c in zip(ref['tensors'],cur['tensors']):
   aa=a.read(r['bytes']);bb=b.read(c['bytes'])
   if aa!=bb:
    out.append(dict(name=c['name'],result=difference(aa,bb,c['dtype'],c['shape']) if (r['dtype'],r['shape'],r['bytes'])==(c['dtype'],c['shape'],c['bytes']) else dict(status='METADATA_MISMATCH',reference=r,actual=c)))
  # A tensor-roster mismatch may leave unmatched bytes; validate both complete
  # immutable streams before reporting the same metadata/numeric difference.
  for stream,total in [(a,ref['total_bytes']),(b,cur['total_bytes'])]:
   while stream.tell()<total:require(stream.read(min(1048576,total-stream.tell())),'Failure raw stream truncated')
   require(not stream.read(1),'Failure raw stream trailing bytes')
 return dict(status='FAIL',earliest_divergence_token=cur['token'],non_tensor_metadata_equal=ref['metadata']==cur['metadata'],differences=out)

def reduce_qualification(root,codec='FP'):
 root=Path(root);rows=[];processes=[]
 for condition,repeat in [('B0',0),('B0',1),('B1',0),('B1',1)]:
  u=f'{condition}_{codec}_Q{repeat}';p=root/'execution'/u/'RESULT.json';r=read(p);require(r['status']=='PASS' and r['mode']=='qualify' and r['condition']==condition and r['codec']==codec and r['repeat']==repeat,'Missing or failed qualification:'+u)
  require(r['consumed_tokens']==256 and r['native_core_calls']==6144 and r['next_token_consumption_checks']==6120 and r['all_tokens_finite'] is True and r['strict_byte_pass'] is True,'Incomplete trajectory/safety qualification:'+u)
  start=read(root/'execution'/u/'STARTED.json');require(r['process_identity']==start['process_identity'],'Process binding mismatch:'+u);require(r['process_identity'] not in processes,'Not fresh process:'+u);processes.append(r['process_identity']);require(r['seed']==20261015 and r['unit_id']==u and r['workspace']=={'B0':':4096:8','B1':':16:8'}[condition],'Unit/seed/workspace changed:'+u)
  idir=root/'execution'/u/'numeric';ip=idir/'NUMERICAL_INDEX.json';index=read(ip);require(index['status']=='PASS' and index['tokens']==256,'Incomplete256 qualification:'+u)
  require(r['numerical_index']['sha256']==sha_bytes(ip.read_bytes()) and r['numerical_index']['bytes']==ip.stat().st_size,'Index receipt identity changed:'+u)
  actual=[];require(len(index['token_receipts'])==256,'Missing token receipts:'+u)
  for token,receipt in enumerate(index['token_receipts']):
   tp=idir/receipt['path'];raw=tp.read_bytes();require(receipt['token']==token and receipt['sha256']==sha_bytes(raw) and receipt['bytes']==len(raw),'Token receipt changed:'+u);x=json.loads(raw)
   require(x['token']==token and x['byte_equal'] is True,'Token byte qualification failed:'+u)
   require(x['actual_bytes_compared']==(u!=f'B0_{codec}_Q0'),'Missing actual byte comparison:'+u)
   if u==f'B0_{codec}_Q0':
    if 'raw_archive' in x:require(x['raw_archive']['bytes']==x['total_bytes'],'Reference archive bytes changed:'+u)
    else:
     rp=idir/x['raw_file'];require(rp.stat().st_size==x['total_bytes'],'Reference raw bytes changed:'+u)
    with open_receipt_raw(idir,x) as rawfile:
     for tr in x['tensors']:
      require(rawfile.tell()==tr['offset'] and sha_bytes(rawfile.read(tr['bytes']))==tr['sha256'],'Reference raw tensor SHA changed:'+u+':'+tr['name'])
     require(not rawfile.read(1),'Unexpected trailing reference raw bytes')
   actual.append([x['token'],x['metadata'],x['tensors']])
  require(sha_bytes(canonical(actual))==index['all_tensor_digest'],'Full tensor digest does not recompute:'+u);rows.append((u,r,index))
 baseline=rows[0][2]['all_tensor_digest'];require(all(x[2]['all_tensor_digest']==baseline for x in rows),'Fresh process tensor/metadata digest mismatch')
 require(not rows[0][2]['actual_bytes_compared'] and all(x[2]['actual_bytes_compared'] for x in rows[1:]),'Actual byte comparison proof missing')
 keys=['contract_sha256','source_manifest_sha256','preregistration_sha256','fixture_canonical_sha256','GPU_UUID','runtime_library_digest','runtime_override_env_digest','authorization_sha256']
 for key in keys:require(len({r[key] for _,r,_ in rows})==1,'Identity mismatch:'+key)
 return dict(status='PASS',codec=codec,scope='FULL256_INPUT_FP_OR_CODEC_SHORT_NUMERICAL_NOT_32K_OR_81K',B0_fresh_determinism='PASS_BYTE_EXACT',B1_fresh_determinism='PASS_BYTE_EXACT_BY_TWO_FRESH_RUNS_EQUAL_SAME_REFERENCE',B0_vs_B1='PASS_ACTUAL_BYTES_ALL_TENSORS_ALL256_TOKENS',shared_tensor_metadata_digest=baseline,units=[u for u,_,_ in rows],identities={k:rows[0][1][k] for k in keys})
def reduce_performance(root):
 import statistics
 root=Path(root);qualification=reduce_qualification(root,'FP');values={c:[] for c in ['B0','B1']};rows=[];processes=[]
 for repeat in range(3):
  for c in ['B0','B1']:
   u=f'{c}_FP_P{repeat}';directory=root/'execution'/u;r=read(directory/'RESULT.json');started=read(directory/'STARTED.json')
   require(r['status']=='PASS' and r['mode']=='benchmark' and r['unit_id']==u and r['codec']=='FP' and r['condition']==c and r['repeat']==repeat,'Incomplete/wrong performance:'+u)
   require(r['workspace']=={'B0':':4096:8','B1':':16:8'}[c] and r['seed']==20261015,'Workspace/seed mismatch:'+u)
   require(r['measured_tokens']==128 and r['warmup_tokens']==128,'Window changed')
   for k,v in qualification['identities'].items():require(r[k]==v,'Performance/numerical identity mismatch:'+u+':'+k)
   require(r['process_identity']==started['process_identity'],'Process start binding changed:'+u);require(r['process_identity'] not in processes,'Not a fresh process:'+u);processes.append(r['process_identity'])
   tps=r['tokens_per_second'];wall=r['decode_wall_seconds'];require(type(tps) in [int,float] and type(wall) in [int,float] and math.isfinite(tps) and math.isfinite(wall) and tps>0 and wall>0,'Invalid timing:'+u)
   require(abs(tps*wall-128)<=1e-6,'TPS/wall incoherent:'+u);require(math.isfinite(r['whole_process_wall_seconds']) and r['whole_process_wall_seconds']>=wall,'Invalid whole-process wall:'+u)
   raw=r['benchmark_reference_result'];require(raw['instrumented'] is False and raw['consumed_tokens']==256 and raw['native_core_calls']==6144 and raw['QDQ_calls']==0 and raw['condition']=='FP_NATIVE','Benchmark reference contract changed:'+u)
   values[c].append(tps);rows.append(r)
 med={c:statistics.median(v) for c,v in values.items()};gain=med['B1']/med['B0']-1;outside=min(values['B1'])>max(values['B0'])
 return dict(status='PASS_MEASUREMENT_COMPLETE',TPS=values,median_TPS=med,relative_gain=gain,at_least_5_percent=gain>=.05,all_B1_above_all_B0=outside,benefit_pass=gain>=.05 and outside,criterion='medianB1>=1.05*medianB0 AND minB1>maxB0; no selective repeats',identities=qualification['identities'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--codec',choices=['FP','H_C128'],default='FP');p.add_argument('--performance',action='store_true');p.add_argument('--failure-reference',type=Path);p.add_argument('--failure-actual',type=Path);a=p.parse_args()
 if a.failure_reference:r=compare_failure(a.failure_reference,a.failure_actual)
 elif a.performance:r=reduce_performance(a.root)
 else:r=reduce_qualification(a.root,a.codec)
 print(json.dumps(r,sort_keys=True,allow_nan=False))
if __name__=='__main__':main()
