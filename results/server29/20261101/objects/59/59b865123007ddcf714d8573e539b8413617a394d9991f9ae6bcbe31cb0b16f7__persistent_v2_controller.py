"""Offline parent: exact NVML reclaim, sequential fixed workers, fail-closed."""
import os,subprocess,time,traceback
import psutil
import persistent_v2_common as A
def main():
 A.verify_manifest();A.require(not A.cmd(['git','status','--porcelain']),'not clean after Commit A')
 A.require(A.cmd(['git','symbolic-ref','--short','HEAD'])==A.BRANCH,'wrong branch')
 A.require(A.cmd(['git','rev-parse','HEAD'])!=A.PARENT,'Commit A missing')
 A.save('provenance/v2_premeasurement_commit.json',dict(commit=A.cmd(['git','rev-parse','HEAD']),parent=A.PARENT,git_status='',time=time.time()))
 nv=A.Nvml();matrix=A.read(A.ROOT/'configs/v2_worker_matrix.json')['runs']
 for job in matrix:
  gpu=job['gpu'];key=job['key'];A.verify_manifest()
  pre=nv.snapshot(gpu);A.require(pre['uuid']==A.UUIDS[gpu],'UUID drift')
  A.require(not pre['processes'],'INCONCLUSIVE_EXTERNAL_GPU_ACTIVITY before spawn')
  A.require(os.statvfs('/data').f_bavail*os.statvfs('/data').f_frsize>=25*(1024**3),'free /data below 25GiB')
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
  descendants=set();history=[];external=set();started=time.time()
  with (A.ROOT/'logs'/f'{key}.log').open('x') as log:
   proc=subprocess.Popen([A.PY,'-u',str(A.ROOT/'scripts/persistent_v2_worker.py'),'--gpu',str(gpu),'--arm',job['arm'],'--key',key],cwd=A.REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
   print('SPAWN',key,proc.pid,flush=True)
   while proc.poll() is None:
    try:descendants.update(p.pid for p in psutil.Process(proc.pid).children(recursive=True))
    except psutil.NoSuchProcess:pass
    shot=nv.snapshot(gpu);external.update(p['pid'] for p in shot['processes'] if p['pid']!=proc.pid and p['pid'] not in descendants)
    time.sleep(1)
   rc=proc.wait()
  reclaim=False
  for i in range(11):
   post=nv.snapshot(gpu);gone=not psutil.pid_exists(proc.pid) and not any(psutil.pid_exists(p) for p in descendants)
   match=gone and not post['processes'] and post['used_bytes']==pre['used_bytes'] and post['uuid']==pre['uuid']
   history.append(dict(second=i,snapshot=post,worker_and_descendants_absent=gone,baseline_exact=match))
   if match:reclaim=True;break
   if i<10:time.sleep(1)
  gate='INCONCLUSIVE_EXTERNAL_GPU_ACTIVITY' if external else 'PASS' if reclaim else 'FAIL'
  path=A.ROOT/'analysis'/f'{key}_worker_result.json';result=A.read(path) if path.exists() else None
  ready=rc==0 and result is not None and result['status']=='PASS' and gate=='PASS'
  A.save('analysis/'+key+'_worker.json',dict(status='PASS' if ready else 'FAIL',ready='YES' if ready else 'NO',job=job,pid=proc.pid,descendants=sorted(descendants),exitcode=rc,before=pre,reclaim_history=history,PROCESS_EXIT_RECLAIM_GATE=gate,external_PIDs=sorted(external),worker_result=str(path) if result else None,latency_s=time.time()-started))
  (A.ROOT/'logs'/f'{key}.exitcode').write_text(str(rc)+'\n')
  print('CLOSED',key,'READY',ready,'RECLAIM',gate,flush=True)
  A.require(ready,'worker hard gate FAIL '+key+'; stop matrix, no retry')
 A.save('analysis/v2_matrix_completion.json',dict(status='PASS',runs=10,new_AIME_generation=0))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  A.save('analysis/v2_controller.error.json',dict(status='FAIL',error=str(e),traceback=traceback.format_exc(),new_launches_stopped=True,retry=False))
  raise
