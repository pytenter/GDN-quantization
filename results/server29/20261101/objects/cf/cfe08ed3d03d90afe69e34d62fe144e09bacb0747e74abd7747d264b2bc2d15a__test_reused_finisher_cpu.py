"""Exercise the real finisher's mixed-origin path using external temporary CPU fixtures."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    r=args.root.resolve();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    sys.path.insert(0,str(r/'src'))
    import resident_execution_control as C,qualification_reuse as Q
    spec=importlib.util.spec_from_file_location('actual_continuation_finisher',r/'scripts/resident_finalize_offline.py')
    F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)
    def write(p,v):p.write_text(json.dumps(v))
    checks=[]
    with tempfile.TemporaryDirectory(prefix='resident-reuse-finisher-cpu-') as temp:
        p=Path(temp).resolve();assert not p.is_relative_to(r)
        for name in ['QUALIFICATION_JOB_MANIFEST.json','FORMAL_JOB_MANIFEST.json','QUALIFICATION_REUSE_ADMISSION.json','HUMAN_QUALIFICATION_REUSE_AUTHORIZATION.json']:
            shutil.copyfile(r/name,p/name)
        shutil.copytree(r/'provenance/REUSED_QUALIFICATION_48',p/'provenance/REUSED_QUALIFICATION_48',copy_function=shutil.copy2)
        shutil.copytree(r/'provenance/REUSED_QUALIFICATION_58',p/'provenance/REUSED_QUALIFICATION_58',copy_function=shutil.copy2)
        write(p/'SOURCE_FREEZE.json',dict(files=[]))
        identity=dict(pid=999999999,starttick=-1,cmdline=[str(p/'scripts/resident_controller.py'),'--root',str(p)])
        receipts=list(Q.seed(C,p).values())
        session=dict(task_id=C.TASK,execution_attempt=p.name,controller_identity=identity,
            all_own_processes_released=True,own_release_all_PASS=True,unresolved_launches=[],
            status='FAIL_CLOSED_STOP',qualification_receipts=receipts,formal_receipts=[],
            qualification_completed=58,formal_completed=0,resource_intervals=0,attempted_units=0,first_failure=None)
        write(p/'SESSION_RESULT.json',session)
        write(p/'CONTROLLER_STARTED.json',dict(task_id=C.TASK,execution_attempt=p.name,identity=identity))
        write(p/'FINALIZER_CHAIN_STARTED.json',dict(controller_process_really_returned=True,controller_returncode=1))
        write(p/'FORMAL_GENERATIONS_COMPLETE.json',dict(count=0,task_id=C.TASK,execution_attempt=p.name,planned=60,
            formal_receipts=[],source_freeze_sha256='CPU_FIXTURE_ONLY',all_own_processes_released=True))
        control=SimpleNamespace(**{n:getattr(C,n) for n in ['TASK','read','sha','same_process','_interval_ok','validate_qualification_result','validate_formal_result','require']})
        control.verify_sources=lambda root:'CPU_FIXTURE_ONLY'
        control.verify_model_and_external=lambda *a,**k:[]
        control.query_gpu=lambda:dict(apps=[])
        admitted=F.admission(control,p)
        assert admitted['session']['qualification_completed']==58 and admitted['actual_intervals']==[] and not admitted['complete60']
        checks.append('real_finisher_accepts58_actual_original_receipts_and_zero_current_intervals_without_complete_claim')
        for name,edit in [
            ('wrong_origin',lambda v:v['qualification_receipts'][0].update(origin_root='/wrong')),
            ('wrong_source',lambda v:v['qualification_receipts'][0].update(origin_source_freeze_sha256='0'*64)),
            ('wrong_validation_SHA',lambda v:v['qualification_receipts'][0].update(validation_sha256='0'*64)),
            ('old_result_relabelled_current',lambda v:v['qualification_receipts'][0].update(reused_qualification=False)),
        ]:
            value=copy.deepcopy(session);edit(value);write(p/'SESSION_RESULT.json',value)
            try:F.admission(control,p)
            except (RuntimeError,KeyError,FileNotFoundError):pass
            else:raise AssertionError('Finisher accepted '+name)
            checks.append('reject_'+name)
    assert 'torch' not in sys.modules
    with args.output.open('x') as f:json.dump(dict(status='PASS_REAL_FINISHER_REUSED_RECEIPT_PATH_AND_NEGATIVES',
        check_count=len(checks),checks=checks,finisher_sha256=C.sha(r/'scripts/resident_finalize_offline.py'),
        GPU_dispatches=0,Torch_imported=False,fixtures_outside_science_root=True),f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps(dict(status='PASS',check_count=len(checks))))


if __name__=='__main__':main()
