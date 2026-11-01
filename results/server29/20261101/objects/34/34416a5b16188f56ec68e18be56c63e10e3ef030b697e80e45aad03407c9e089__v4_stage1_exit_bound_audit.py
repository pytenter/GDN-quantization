"""Append retrospective natural-exit upper bounds from frozen controller events."""
from v4_common import OUT,RUNS,checked,digest,emit

def main():
    assert checked(OUT/'CROSS_DEVICE_FP_REPEATABILITY.json')['status']=='PASS'
    tags=[f'fp512_pg19-test-5956_gpu6_fresh{i}' for i in (0,1)]
    next_receipt=RUNS/(tags[1]+'.LAUNCH_AUTHORIZATION.json')
    completion=OUT/'V4_STAGE1_COMPLETION.json'
    # Controller wait() required bound PID gone before the next launch or
    # Stage1 completion. These are conservative observation bounds, not exit times.
    bounds=[(checked(next_receipt)['epoch'],next_receipt),(checked(completion)['epoch'],completion)]
    rows=[]
    for t,(bound,source) in zip(tags,bounds):
        result=checked(RUNS/(t+'.json')); assert bound>=result['completed_epoch']
        p=RUNS/(t+'.EXIT_OBSERVED.json'); assert not p.exists(),'NO_DUPLICATE_EXIT_BOUND_AUDIT'
        row=dict(observed_epoch=bound,natural_exit=True,no_signals=True,retrospective_upper_bound=True,
            evidence_path=str(source),evidence_sha256=digest(source),canonical_completed_epoch=result['completed_epoch'],
            basis='Frozen Stage1 controller required bound PROCESS_IDENTITY absent before next launch/completion; not exact exit timestamp')
        emit(p,row); rows.append(row)
    emit(OUT/'STAGE1_NATURAL_EXIT_BOUND_AUDIT.json',dict(status='APPENDED_BOUND_NOT_ORIGINAL_LEDGER_REWRITE',rows=rows,
        initial_Stage1_ledger_unchanged=True,original_canonical_timestamps_unchanged=True))
    print('STAGE1_NATURAL_EXIT_UPPER_BOUNDS_APPENDED')

if __name__=='__main__': main()
