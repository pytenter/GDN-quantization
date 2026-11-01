"""Predeclared exact repeat and original contrast-derived portability gate."""
import shlex
import subprocess
import time
from v4_common import *

def launch(rep):
    cfg,sel,freeze=verify(); t=tag('portability','pg19-test-5956',6,rep)
    assert not any((RUNS/(t+s)).exists() for s in ('.json','.STOP.json','.LAUNCH_AUTHORIZATION.json','.STARTED.json','.log'))
    hw=device(6,cfg)
    busy={'UNAUTHORIZED_TARGET_COMPUTE_PID','UNEXPLAINED_TARGET_MEMORY_OR_UTILIZATION'}
    while not hw['exclusive'] and set(hw['failure_conditions']).issubset(busy):
        print('AWAITING_NATURAL_GPU6_RELEASE',flush=True); time.sleep(30); cfg,sel,freeze=verify(); hw=device(6,cfg)
    assert hw['exclusive'],('GPU6_PRELAUNCH_FAIL',hw)
    session=f'ling_p1l7_v4_port5956_fresh{rep}'; worker=CODE/'v4_portability_worker.py'
    emit(RUNS/(t+'.LAUNCH_AUTHORIZATION.json'),dict(epoch=time.time(),gpu=6,gpu_uuid=UUID,mode='portability',repeat=rep,
        document='pg19-test-5956',runner_sha256=digest(worker),freeze_sha256=digest(FREEZE),session=session,hardware=hw))
    args=['env','PYTHONDONTWRITEBYTECODE=1','CUDA_VISIBLE_DEVICES=6','PYTHONHASHSEED=0','LING_GATED_RMSNORM_FIXED_CONFIG=BT16_W8',
        'CUBLAS_WORKSPACE_CONFIG=:4096:8','/data/zypan/envs/ling-kda/bin/python','-B','-u',str(worker),'--repeat',str(rep)]
    subprocess.run(['tmux','new-session','-d','-s',session,'cd '+shlex.quote(str(CODE))+' && exec '+shlex.join(args)+' > '+shlex.quote(str(RUNS/(t+'.log')))+' 2>&1'],check=True)
    while True:
        assert not LATCH.exists(),'V4_STOP_NO_RETRY'
        path=RUNS/(t+'.json')
        if path.exists() and path.with_suffix('.sha256.json').exists() and not alive(t):
            v=checked(path); assert v['status']=='PASS_SINGLE_PROCESS_NOT_PORTABILITY_APPROVAL'
            emit(RUNS/(t+'.EXIT_OBSERVED.json'),dict(observed_epoch=time.time(),natural_exit=True,no_signals=True))
            return v,path
        if (RUNS/(t+'.STOP.json')).exists(): raise RuntimeError('PORTABILITY_WORKER_STOP')
        log=RUNS/(t+'.log')
        if time.time()-max(checked(RUNS/(t+'.LAUNCH_AUTHORIZATION.json'))['epoch'],log.stat().st_mtime if log.exists() else 0)>1800:
            stop('NO_PROGRESS','PORTABILITY'); raise RuntimeError('NO_PROGRESS_SAFE_TOKEN_STOP')
        time.sleep(5)

def gate(a,b,paths):
    cfg,sel,freeze=verify(); e=next(e for e in sel['documents'] if e['document']=='pg19-test-5956')
    old=checked(e['calibration_identity']['canonical_path'])
    assert digest(e['calibration_identity']['canonical_path'])==e['calibration_identity']['canonical_sha256']
    exact=a['numeric_fingerprint']==b['numeric_fingerprint'] and a['numerical_payload']==b['numerical_payload']
    conditions=[e['pair']['condition_a'],e['pair']['condition_b']]; rows=[]; directions={}
    current={r['condition']:r for r in a['results']}; prior={r['condition']:r for r in old['results']}
    for metric in ('P1','L7_TOTAL_ORIGINAL'):
        old_gap=abs(prior[conditions[0]]['aggregate'][metric]-prior[conditions[1]]['aggregate'][metric])
        for c in conditions:
            ov,nv=prior[c]['aggregate'][metric],current[c]['aggregate'][metric]
            rows.append(dict(condition=c,metric=metric,old_GPU3_score=ov,new_GPU6_score=nv,absolute_drift=abs(nv-ov),
                predeclared_strict_bound=old_gap/2,passed=abs(nv-ov)<old_gap/2,
                original_windows=prior[c]['windows'],new_windows=current[c]['windows']))
        directions[metric]=min(conditions,key=lambda c:current[c]['aggregate'][metric])
    expected={'P1':e['predicted_winners']['P1'],'L7_TOTAL_ORIGINAL':e['predicted_winners']['L7_TOTAL_ORIGINAL']}
    preserved=directions==expected and len(set(directions.values()))==2
    passed=exact and preserved and all(x['passed'] for x in rows)
    result=dict(status='PASS' if passed else 'FAIL',fresh_process_exact_repeat=exact,opposite_predictions_preserved=preserved,
        expected_winners=expected,actual_winners=directions,score_drift_checks=rows,selector_not_run=True,
        original_pair_only=True,source_sha256=[digest(p) for p in paths],old_calibration_sha256=digest(e['calibration_identity']['canonical_path']),
        criteria_prefrozen_plan_sha256=digest(OUT/'V4_EXECUTION_PLAN.md'),new_threshold_chosen_after_scores=False,
        math='original P1 and original L7 functional/state/total; equal mean of 256/1024 H128',epoch=time.time())
    emit(OUT/'P1_L7_PORTABILITY_GATE.json',result)
    if passed:
        emit(OUT/'HARDWARE_MAPPING_AMENDMENT.json',dict(status='APPROVED_PROSPECTIVE_V4_ONLY',epoch=time.time(),
            original_map={'pg19-test-860':6,'pg19-test-5956':3},V4_map={'pg19-test-860':6,'pg19-test-5956':6},
            execution='SERIAL_GPU6',document_order=['pg19-test-860','pg19-test-5956'],
            original_selection_sha256=digest(OLD/'PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json'),
            portability_gate_sha256=digest(OUT/'P1_L7_PORTABILITY_GATE.json'),
            freeze_constraint_review='Original mapping frozen for original run; no perpetually immutable device constraint found. V4 user explicitly authorizes prospective qualified amendment.',
            only_device_map_changes=True,original_V3_FAIL_remains_FAIL=True,cross_run_qualification_inheritance_not_explicitly_allowed=True,
            both_documents_full_512x2_will_be_rerun=True,documents_pairs_predictions_weights_tokens_quantizer_KL_writeback_unchanged=True))
    else:
        stop('PORTABILITY_QUALIFICATION_FAIL','PORTABILITY',fresh_repeat=exact,predictions_preserved=preserved,
            no_formal=True,no_candidate_reselection=True)
    return result

def main():
    try:
        assert not (OUT/'V4_PORTABILITY_SUPERVISOR_STARTED.json').exists(),'NO_DUPLICATE_SUPERVISOR'
        verify(); emit(OUT/'V4_PORTABILITY_SUPERVISOR_STARTED.json',dict(pid=__import__('os').getpid(),epoch=time.time()))
        a,p=launch(0); b,q=launch(1); result=gate(a,b,[p,q])
        emit(OUT/'RESOURCE_USAGE_LEDGER_PORTABILITY.json',ledger())
        print('V4_PORTABILITY_COMPLETE',result['status'],flush=True)
    except Exception as exc:
        stop(repr(exc),'PORTABILITY_SUPERVISION'); raise

if __name__=='__main__': main()
