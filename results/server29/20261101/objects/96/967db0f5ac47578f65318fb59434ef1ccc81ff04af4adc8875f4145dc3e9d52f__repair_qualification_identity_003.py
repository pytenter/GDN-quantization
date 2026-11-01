from common import EXP, canonical_hash, read_json, sha256, write_json


EXCLUDED = ['elapsed_seconds','peak_gpu_allocated_bytes','mode','run_index',
            'deterministic_identity_sha256','paired_clean_scale_schedule']


def identity(value):
    value=dict(value)
    for key in EXCLUDED:
        value.pop(key,None)
    return canonical_hash(value)


def main():
    old_log=EXP/'discovery/logs/qualify.log'
    old_fail=EXP/'FAIL_AND_STOP_RECOVERY_002.json'
    if not old_log.exists() or not old_fail.exists():
        raise RuntimeError('historical qualification identity failure missing')
    a=read_json(EXP/'qualification/smoke_run101_H.json')
    b=read_json(EXP/'qualification/smoke_run102_H.json')
    arms={}
    for arm in a['arms']:
        ia=identity(a['arms'][arm]); ib=identity(b['arms'][arm])
        if ia!=ib:
            raise RuntimeError(f'scientific smoke identity really differs: {arm}')
        arms[arm]={'run101_scientific_identity':ia,
                   'run102_scientific_identity':ib,
                   'exact_match_after_excluding_schedule_scope_metadata':True,
                   'state_hashes_exact':a['arms'][arm]['state_hashes_final']==b['arms'][arm]['state_hashes_final'],
                   'clamp_count_exact':a['arms'][arm]['clamp_count']==b['arms'][arm]['clamp_count'],
                   'QDQ_count_exact':a['arms'][arm]['qdq_calls_after_anchor']==b['arms'][arm]['qdq_calls_after_anchor']}
    write_json(EXP/'engineering/qualification_identity_repair_003.json',{
        'classification':'ENGINEERING_ONLY_QUALIFICATION_IDENTITY_SCOPE',
        'failure':'The smoke identity included paired_clean_scale_schedule metadata even though run101 intentionally materialized through 32768 and run102 intentionally stopped at 12288.',
        'repair':'Validate full clean schedule identity separately, and compare the two 8192->12288 scientific smoke payloads after excluding schedule-scope metadata plus runtime-only fields.',
        'excluded_fields':EXCLUDED,
        'scientific_trajectory_changed':False,'formal_measurements_before_fix':0,
        'GPU_rerun_required':False,'old_failure_preserved':True,
        'old_qualify_log_sha256':sha256(old_log),'old_failure_sha256':sha256(old_fail),
        'arms':arms})
    print('QUALIFICATION_IDENTITY_SCOPE_REPAIR_PASS',flush=True)


if __name__=='__main__':
    main()
