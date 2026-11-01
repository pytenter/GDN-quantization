"""CPU-only proposed finite roster; does not authorize, freeze or dispatch."""
import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--publish-plans',action='store_true');a=p.parse_args();r=a.root.resolve();sys.path.insert(0,str(r/'src'))
import execution_control as C
from control_plan import build_formal_plan,build_qualification_plan
C.require(not (r/'GPU_RUN_STARTED.json').exists() and not (r/'SOURCE_FREEZE.json').exists(),'Cannot change plans after freeze/STARTED')
dataset=C.read(a.dataset);assignment,formal=build_formal_plan(dataset,C.UUIDS,C.AUTHORIZED_GPUS);qualification=build_qualification_plan(C.UUIDS,C.AUTHORIZED_GPUS)
if a.publish_plans:
 for name,v in [('QUESTION_GPU_ASSIGNMENT.json',assignment),('FORMAL_JOB_MANIFEST.json',formal),('GPU_EXECUTION_PLAN.json',qualification)]:C.atomic_new(r/name,v)
print(json.dumps(dict(status='CPU_FINITE_ROSTER_CHECK_PASS_NO_GPU_DISPATCH',questions=30,formal_new_samples=90,qualification_units=len(qualification['qualification_units']),physical_GPU_indices=list(C.AUTHORIZED_GPUS),UUIDs=[C.UUIDS[g] for g in C.AUTHORIZED_GPUS],published_plans=a.publish_plans,source_frozen=False,GPU_dispatches=0)))
