"""Synthetic CPU qualification; zero GPU operations."""
from pathlib import Path
import hashlib, json
from resource_event_reducer import reduce_events
base=[dict(unit_id='u1',GPU_UUID='synthetic',monotonic_seconds=t,kind=k) for t,k in
 [(0.,'dispatch_before_spawn'),(.02,'spawn_observed'),(5.,'busy_observed'),(5.4,'first_process_exit_observed'),(6.,'first_exit_and_release_observed')]]
checks=[]
def check(name,fn):
 fn();checks.append({'name':name,'status':'PASS'})
def valid():
 x=reduce_events(base);u=x['units'][0]
 assert u['occupied_seconds_lower']==4.98 and u['occupied_seconds_upper']==6
 assert u['exact_GPU_hours']=='UNKNOWN'
def rejects(events):
 try:reduce_events(events)
 except ValueError:return
 raise AssertionError('invalid event sequence admitted')
check('observation_bounds_not_exact_physical_time',valid)
check('first_exit_cannot_be_overwritten',lambda:rejects(base+[dict(base[3],monotonic_seconds=7.)]))
check('first_release_cannot_be_overwritten',lambda:rejects(base+[dict(base[4],monotonic_seconds=7.)]))
check('busy_after_release_rejected',lambda:rejects(base+[dict(base[2],monotonic_seconds=7.)]))
check('duplicate_dispatch_rejected',lambda:rejects(base+[dict(base[0],monotonic_seconds=7.)]))
check('out_of_order_rejected',lambda:rejects([base[0],base[2],base[1]]))
check('different_device_rejected',lambda:rejects(base[:1]+[dict(base[1],GPU_UUID='other')]))
check('missing_exit_receipt_rejected',lambda:rejects(base[:3]+base[4:]))
check('incomplete_never_reported_as_zero_cost_complete',lambda: (None if reduce_events(base[:3])['has_incomplete_units'] else (_ for _ in ()).throw(AssertionError())))
check('incomplete_without_live_time_cost_UNKNOWN',lambda: (None if reduce_events(base[:3])['GPU_seconds_upper_bound']=='UNKNOWN' else (_ for _ in ()).throw(AssertionError())))
check('live_budget_counts_entire_dispatch_so_far',lambda: (None if reduce_events(base[:3],as_of_monotonic=7.)['GPU_seconds_upper_bound']==7. else (_ for _ in ()).throw(AssertionError())))
root=Path(__file__).resolve().parent
report={'status':'CPU_SYNTHETIC_PASS','GPU_dispatches':0,'checks':checks,'not_GPU_runtime_qualification':True,
 'reducer_sha256':hashlib.sha256((root/'resource_event_reducer.py').read_bytes()).hexdigest()}
(root/'RESOURCE_EVENT_CPU_QUALIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'checks':len(checks),'GPU_dispatches':0}))
