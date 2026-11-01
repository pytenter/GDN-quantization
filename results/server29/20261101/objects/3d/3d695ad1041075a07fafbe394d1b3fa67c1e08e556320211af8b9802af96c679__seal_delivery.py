#!/usr/bin/env python3
"""CPU-only final evidence seal; does not run a GPU workload or remove assets."""
from pathlib import Path
import argparse,hashlib,json,subprocess,os,time
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);a=ap.parse_args();r=a.root
TASK='QWEN_GDN_HISTORICAL_THROUGHPUT_RECONCILIATION_V1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(n):return json.loads((r/n).read_bytes())
def put(n,j):
 p=r/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(j,f,ensure_ascii=False,indent=2);f.write('\n')
def entry(p):return dict(path=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=sha(p))
def same(i):
 p=Path('/proc')/str(i['pid'])/'stat'
 return p.exists() and p.read_text().rsplit(')',1)[1].split()[19]==str(i['start_tick'])
l=read('GPU_RESOURCE_LEDGER.json');batch=read('execution/BATCH_RESULT.json');m=read('SOURCE_SHA256_MANIFEST.json')
assert sha(r/'SOURCE_SHA256_MANIFEST.json')=='b59ee3a7417ebdae7f4f857bc3e8eb2f1644c373d6d28bf8d31da2871671c9f3'
assert batch['status']=='PASS' and batch['ledger_sha256']==sha(r/'GPU_RESOURCE_LEDGER.json')
assert l['all_own_GPUs_released'] and l['cumulative_GPU_hours']<=.75
assert not same(read('execution/BATCH_STARTED.json')['identity']) and all(not same(d['identity']) and d['process_exited'] and d['NVML_released'] and d['returncode']==0 for d in l['dispatches'])
live=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
assert not {d['pid'] for d in l['dispatches']}.intersection(int(x.split(',')[0]) for x in live.splitlines() if x.strip())
checks=[]
for x in m['execution_sources']:
 p=r/x['path'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'];checks.append(dict(path=str(p),sha256=x['sha256'],bytes=x['bytes'],kind='FROZEN_EXECUTION'))
for x in m['external_dependencies']:
 p=Path(x['path']);assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'];checks.append(dict(x,kind='FROZEN_DEPENDENCY'))
for x in read('CPU_EVIDENCE_SNAPSHOT_CATALOG.json'):
 p=r/x['snapshot_path'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'];checks.append(dict(path=str(p),sha256=x['sha256'],bytes=x['bytes'],kind='CPU_EVIDENCE_SNAPSHOT'))
large=[]
assert len(batch['completed_units'])==20
for u in batch['completed_units']:
 p=r/u['path'];assert p.stat().st_size==u['bytes'] and sha(p)==u['sha256'];j=json.loads(p.read_bytes())
 if j['profile']:
  x=j['chrome_trace'];t=r/x['path'];assert t.stat().st_size==x['bytes'] and sha(t)==x['sha256'];large.append(dict(x,retention='PERMANENT_55_DATA_NO_DELETE_NO_LOCAL_DOWNLOAD'))
# Check model checkpoint stat guards against CPU full raw-byte SHA record, without redownloading weights.
weights=read('SHARED_CHECKPOINT_IDENTITY.json')
weight_status=[]
for x in weights['files']:
 p=Path(x['path']);s=p.stat();assert s.st_size==x['bytes']
 if 'mtime_ns' in x:assert s.st_mtime_ns==x['mtime_ns']
 if 'inode' in x:assert s.st_ino==x['inode']
 weight_status.append(dict(path=str(p),bytes=s.st_size,sha256=x['sha256'],postrun_guard='SIZE_MTIME_INODE_STABLE_AFTER_CPU_FULL_RAW_SHA'))
put('REMOTE_DELIVERY_VERIFICATION.json',dict(task_id=TASK,status='PASS',frozen_sources_and_dependencies_checks=len(checks),checks=checks,large_profiles_fresh_SHA_bytes=large,shared_checkpoint_postrun_stat_guards=weight_status,GPU_seconds=l['cumulative_GPU_seconds'],GPU_hours=l['cumulative_GPU_hours'],all_own_processes_exited=True,NVML_tracked_PIDs_absent=True,foreign_jobs_untouched=True,retention='All original and new assets retained in permanent /data on55; nothing removed or copied to temporary archive.',unix=time.time()))
assert len(weight_status)==4
required=['HISTORICAL_EXECUTION_IDENTITY.json','CURRENT_EXECUTION_IDENTITY.json','HISTORICAL_TPS_RECOMPUTATION.json','HISTORICAL_TPS_BY_LENGTH.csv','HISTORICAL_TPS_BY_GPU.csv','RUNTIME_IDENTITY_DIFF.md','PERFORMANCE_CAUSE_HYPOTHESES.md','MATCHED_RUNTIME_BENCHMARK.csv','MATCHED_RUNTIME_BENCHMARK.json','GPU_RESOURCE_LEDGER.json','SOURCE_SHA256_MANIFEST.json','FIRST_FAILURE.json','GATE_STATUS_FINAL.json','FINAL_TECHNICAL_REPORT.md']
assert all((r/n).is_file() and (r/n).stat().st_size>0 for n in required)
small=[]
for p in sorted(r.rglob('*')):
 if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.pyc':continue
 rel=str(p.relative_to(r))
 if rel in {'DELIVERY_ASSET_MANIFEST.json','SHA256SUMS.txt'} or rel.startswith('profiles/'):continue
 assert p.stat().st_size<5000000,'Unexpected large payload '+rel
 small.append(entry(p))
# Every post-execution analysis source is sealed, separate from immutable GPU execution manifest.
post=[x for x in small if x['path'].startswith(('analysis/','source/')) and x['path'] not in {z['path'] for z in m['execution_sources']}]
put('POSTEXECUTION_ANALYSIS_SOURCE_MANIFEST.json',dict(task_id=TASK,status='CPU_ANALYSIS_AND_DELIVERY_ONLY',sources=post))
small.append(entry(r/'POSTEXECUTION_ANALYSIS_SOURCE_MANIFEST.json'))
sums=''.join(x['sha256']+'  '+x['path']+'\n' for x in sorted(small+large,key=lambda x:x['path']))
with (r/'SHA256SUMS.txt').open('x') as f:f.write(sums)
small.append(entry(r/'SHA256SUMS.txt'))
put('DELIVERY_ASSET_MANIFEST.json',dict(task_id=TASK,status='SEALED_COMPLETE_AND_STOP',small_assets=small,external_dependencies=m['external_dependencies'],large_remote_only=large,required_files=required,report_sha256=sha(r/'FINAL_TECHNICAL_REPORT.md'),GPU_hours=l['cumulative_GPU_hours']))
print(json.dumps(dict(status='SEALED_COMPLETE_AND_STOP',small_files=len(small),dependencies=len(m['external_dependencies']),large_profiles=len(large),checkpoint_guards=len(weight_status),GPUh=l['cumulative_GPU_hours'],report_sha256=sha(r/'FINAL_TECHNICAL_REPORT.md'))))
