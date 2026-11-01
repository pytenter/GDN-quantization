"""Actual archived48 revalidation plus in-memory provenance corruption negatives. No GPU dispatch."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();r=args.root.resolve()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    sys.path[:0]=[str(r/'src')]
    import resident_execution_control as C,qualification_reuse as Q
    assert 'torch' not in sys.modules
    original=C.read(r/'QUALIFICATION_REUSE_ADMISSION.json')
    receipts=Q.seed(C,r)
    assert len(receipts)==48 and all(x['reused_qualification'] for x in receipts.values())
    unit=original['reused'][0]['unit'];checks=[dict(name='all_actual48_original_and_cached_SHA_exit_release_verified',status='PASS')]
    cases=[
        ('human_authorization',lambda v:v.update(human_authorization_sha256='0'*64)),
        ('count47',lambda v:v.update(reused_count=47)),
        ('remaining11',lambda v:v.update(remaining_count=11)),
        ('duplicate_unit',lambda v:v['reused'].__setitem__(1,copy.deepcopy(v['reused'][0]))),
        ('wrong_source',lambda v:v['reused'][0].update(origin_source_freeze_sha256='0'*64)),
        ('wrong_seal',lambda v:v['reused'][0].update(origin_seal_sha256='0'*64)),
        ('wrong_permit',lambda v:v['reused'][0].update(origin_permit_sha256='0'*64)),
        ('wrong_UUID',lambda v:v['reused'][0]['unit'].update(gpu_uuid='GPU_WRONG')),
        ('profiler_reuse',lambda v:v['reused'][0]['unit'].update(kind='profile_short')),
        ('origin_is_current',lambda v:v['reused'][0].update(origin_root=str(r))),
        ('cached_path_escape',lambda v:v['reused'][0].update(cached_evidence_prefix='../outside')),
        ('cached_file_path_escape',lambda v:v['reused'][0]['files'][0].update(path='../outside')),
        ('cached_file_SHA',lambda v:v['reused'][0]['files'][0].update(sha256='0'*64)),
        ('cached_file_size',lambda v:v['reused'][0]['files'][0].update(bytes=-1)),
        ('result_receipt_SHA',lambda v:v['reused'][0]['controller_receipt'].update(result_sha256='0'*64)),
        ('exit_receipt_SHA',lambda v:v['reused'][0]['controller_receipt'].update(resource_interval_sha256='0'*64)),
        ('validation_receipt_SHA',lambda v:v['reused'][0]['controller_receipt'].update(validation_sha256='0'*64)),
    ]
    read=C.read
    for name,mutate in cases:
        value=copy.deepcopy(original);mutate(value)
        def altered(path):
            return value if Path(path)==r/'QUALIFICATION_REUSE_ADMISSION.json' else read(path)
        C.read=altered
        try:
            try:Q.validate(C,r,unit)
            except (RuntimeError,AssertionError,FileNotFoundError,KeyError,StopIteration):pass
            else:raise AssertionError('Corrupted provenance accepted: '+name)
        finally:C.read=read
        checks.append(dict(name='reject_'+name,status='PASS'))
    remaining=original['remaining']
    assert len(remaining)==10 and not {x['id'] for x in remaining}&set(receipts)
    assert {x['id'] for x in C.read(r/'QUALIFICATION_JOB_MANIFEST.json')['units']}==set(receipts)|{x['id'] for x in remaining}
    checks.append(dict(name='all_same58_requirements_exactly48_plus10_without_duplicates',status='PASS'))
    assert 'torch' not in sys.modules
    pins={p:hashlib.sha256((r/p).read_bytes()).hexdigest() for p in [
        'src/qualification_reuse.py','src/resident_execution_control.py','scripts/resident_controller.py',
        'scripts/resident_finalize_offline.py','QUALIFICATION_REUSE_ADMISSION.json','HUMAN_QUALIFICATION_REUSE_AUTHORIZATION.json']}
    with args.output.open('x') as f:json.dump(dict(status='PASS_ACTUAL48_READONLY_REVALIDATION_AND_PROVENANCE_NEGATIVES',check_count=len(checks),checks=checks,
        source_pins=pins,real_old_qualification_count=48,new_GPU_dispatches=0,new_AIME_generations=0,
        negatives_only_in_memory_no_science_fixture_paths=True,Torch_imported=False),f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps(dict(status='PASS',check_count=len(checks),old48_revalidated=True,new_GPU_dispatches=0)))


if __name__=='__main__':main()
