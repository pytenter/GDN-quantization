"""Complete the same authorized P1 stop after recorded job-control STOP interference.
No experimental retry or original source/SESSION edit. Default is read-only.
"""
from pathlib import Path
import argparse,importlib.util,hashlib,json,os,signal,time,traceback
R=Path('/data/zypan/analyses/qwen_gdn_oscar_resident_rotation_efficient_qualification_v1_gpu_continuation_20261022')
TOOL_SHA='92d1d044603a806b3231ca0ec20a5a7cc12d1f98a19849232b11b8e22c595330'
AUTH_SHA='d00363479cef8ca7cdea8b288b5db4ca87bccbc0b0fd2d9bf00b21b3fd5fdf44'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--expected-self-sha256');a=ap.parse_args()
 p=R/'operations/stop_p1_authorized.py';assert hashlib.sha256(p.read_bytes()).hexdigest()==TOOL_SHA
 spec=importlib.util.spec_from_file_location('preserved_stop_tool',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 ah,auth=m.validate_authorization(R/'HUMAN_CONTINUATION_AUTHORIZATION.json',AUTH_SHA,a.execute)
 failure=R/'operations/P1_AUTHORIZED_STOP/STOP_FAILURE.json';assert failure.exists() and 'Coordinator must be STOPPED' in m.load(failure)['error']
 ci,pa,la=m.verify_frozen_p1();jobs=m.jobs_by_id();workers,unknown=m.scan_workers(ci,jobs);m.require(not unknown,'Unknown coordinator child')
 nv=m.query_nvml();m.check_nvml_ownership(workers,nv)
 preview=dict(status='READ_ONLY_RECOVERY_PREVIEW_NO_SIGNALS',human_authorization_sha256=ah,coordinator=ci,launcher=pa,workers=workers,NVML=nv,prior_stop_failure_sha256=m.sha(failure),source_freeze_sha256=m.PINS['SOURCE_FREEZE.json'])
 if not a.execute:print(json.dumps(preview));return
 assert a.expected_self_sha256==m.sha(Path(__file__))
 out=R/'operations/P1_AUTHORIZED_STOP_COMPLETION';m.require(not out.exists(),'Completion already attempted; inspect original evidence')
 out.mkdir();m.save(out/'STARTED.json',dict(status='SAME_HUMAN_STOP_AUTHORIZATION_ENGINEERING_COMPLETION',preview=preview,original_failure_preserved=True,coordinator_identity_method='Kernel3.10 fresh PID/starttick/argv; no pidfd claim'))
 events=m.SignalControl(out/'SIGNAL_EVENTS.jsonl',True)
 try:
  for i in [ci,pa]:events.bind(i)
  events.send(ci,signal.SIGSTOP);m.wait_stopped(ci)
  workers,unknown=m.scan_workers(ci,jobs);m.require(not unknown,'Unknown child while coordinator frozen')
  ids={x['identity']['pid']:x['identity'] for x in preview['workers']+workers}
  m.save(out/'PRE_REMOVAL.json',dict(coordinator=m.proc_info(ci['pid']),launcher=m.proc_info(pa['pid']),workers=workers,NVML=m.query_nvml()))
  # Parent is removed first, so it cannot chain its unconditional finisher.
  # Do not STOP the parent: observed terminal job control resumes coordinator.
  # The two exact signals are followed by a complete root-worker race rescan.
  pair=[]
  for expected in [pa,ci]:
   ob=m.exited_observation(expected)
   if ob['exact_identity_exited']:pair.append(dict(identity=expected,signal_sent=False,observation=ob));continue
   m.require(m.exact_live(expected),'Scheduler identity changed immediately before KILL')
   os.kill(expected['pid'],signal.SIGKILL)
   pair.append(dict(identity=expected,signal_sent=True,signal='SIGKILL',monotonic=time.monotonic()))
  m.save(out/'SCHEDULER_REMOVAL.json',dict(actual=pair,atomic_pair_claimed=False,manual_SIGCONT=False))
  end=time.monotonic()+20
  while True:
   try:
    workers,unknown=m.scan_workers(ci,jobs,allow_reparented=True);m.require(not unknown,'Unknown post-removal children');break
   except RuntimeError:
    m.require(time.monotonic()<end,'Worker ownership unresolved after removal');time.sleep(.1)
  m.check_nvml_ownership(workers,m.query_nvml())
  for w in workers:ids[w['identity']['pid']]=w['identity'];events.bind(w['identity'])
  m.save(out/'POST_SCHEDULER_WORKER_RESCAN.json',dict(workers=workers,all_original_coordinator_live=m.exact_live(ci),all_original_launcher_live=m.exact_live(pa)))
  for w in workers:events.send(w['identity'],signal.SIGTERM)
  release=m.observe_release(list(ids.values()),120,events);m.save(out/'WORKER_EXIT_AND_NVML_RELEASE.json',release)
  m.require(release['status']=='PASS_EXACT_IDENTITIES_EXITED_AND_OWN_NVML_RELEASED','Own worker release unresolved')
  sched=m.observe_release([pa,ci],30,events);m.save(out/'SCHEDULER_EXIT.json',sched)
  m.require(sched['status']=='PASS_EXACT_IDENTITIES_EXITED_AND_OWN_NVML_RELEASED','Own scheduler exit unresolved')
  remain,unknown=m.scan_workers(ci,jobs,allow_reparented=True);m.require(not remain and not unknown,'Own workers remain')
  other=[]
  for p in Path('/proc').glob('[0-9]*'):
   info=m.proc_info(int(p.name))
   if info and info['state'] not in ['Z','X','x']:
    args=info['identity']['cmdline']
    if any(str(m.P1/'scripts'/name) in args for name in ['gpu_controller.py','gpu_unit_worker.py','run_offline.sh','finalize_offline.py']):other.append(info)
  m.require(not other,'Original root scheduler/worker/finisher remains; do not admit newGPU')
  m.require(not (m.P1/'FINALIZER_STARTED.json').exists(),'Unexpected original finisher attempt; audit before GPU')
  for rel,expected in m.PINS.items():m.require(m.sha(m.P1/rel)==expected,'Old source changed')
  nv=m.query_nvml();complete=[];payloads=[]
  for job in jobs.values():
   if job['condition']!='P1':continue
   d=m.P1/'formal'/job['id']
   if (d/'RESULT.json').exists():payloads.append(job['id'])
   if (d/'RESULT.json').exists() and (d/'RESOURCE_INTERVAL.json').exists():
    x=m.load(d/'RESOURCE_INTERVAL.json')
    if x['exit_returncode']==0 and x['release']=='PASS':complete.append(dict(unit=job['id'],result_sha256=m.sha(d/'RESULT.json'),interval_sha256=m.sha(d/'RESOURCE_INTERVAL.json')))
  m.save(out/'POST_STOP_METADATA_CATALOG.json',m.metadata_snapshot())
  receipt=dict(status='PASS_HUMAN_AUTHORIZED_P1_STOP_EXACT_IDENTITIES_EXITED_OWN_GPU_RELEASED',terminal='USER_CANCELLED_P1_PARTIAL_DO_NOT_RESUME',human_authorization_sha256=ah,tool_sha256=m.sha(Path(__file__)),original_stop_tool_sha256=TOOL_SHA,prior_stop_failure_preserved_sha256=m.sha(failure),P1_root=str(m.P1),P1_source_freeze_sha256=m.PINS['SOURCE_FREEZE.json'],P1_permit_sha256=m.PINS['GPU_PERMIT.json'],original_launcher=pa,original_coordinator=ci,stopped_workers=list(ids.values()),no_future_original_scheduler_dispatch=True,all_exact_own_processes_exited=True,own_GPU_UUIDs_released=True,final_NVML=nv,selected_UUIDs_currently_idle=not any(x['uuid'] in m.UUIDS.values() for x in nv),original_completed_candidate_count=len(complete),original_completed_candidates=complete,actual_RESULT_payload_count=len(payloads),original_RESULT_units=payloads,completed_candidates_require_independent_payload_validation=True,in_flight_samples='HUMAN_CANCELLED_PARTIAL_NOT_SCORED_OR_CONCATENATED',old_C1_S3_failure_terminations_preserved=True,original_SESSION_RESULT_fabricated=False,original_ledger_rewritten=False,raw_Tensors_or_RAM_references_deleted=False,old_source_rewritten=False,original_finisher_absent_independently_verified=True,resident_GPU_dispatches=0,exact_physical_GPUh='UNRESOLVED',monotonic=time.monotonic())
  m.save(out/'USER_STOP_RECEIPT.json',receipt);print(json.dumps(receipt))
 except BaseException as e:
  if not (out/'FAILURE.json').exists():m.save(out/'FAILURE.json',dict(status='STOP_COMPLETION_UNRESOLVED_PRESERVE',error=repr(e),traceback=traceback.format_exc(),signals=events.sent,no_GPU_admission=True))
  raise
 finally:events.close()
if __name__=='__main__':main()
