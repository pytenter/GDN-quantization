"""Each stage is gated independently; no duplicate launches or science peeking."""
import shlex
import subprocess
import time
from v4_persistent_common import *

def launch(mode,doc,rep=0):
    cfg,sel,freeze=verify(); t=tag(mode,doc,6,rep)
    assert not any((RUNS/(t+s)).exists() for s in ('.json','.STOP.json','.STARTED.json','.LAUNCH_AUTHORIZATION.json','.log'))
    if mode in ('micro','formal'): assert checked(OUT/'GPU_512X2_QUALIFICATION.json')['status']=='PASS'
    if mode=='formal':
        assert checked(OUT/'PHYSICAL_RESOURCE_FEASIBILITY.json')['status']=='PASS'
        assert checked(OUT/'FROZEN_FORMAL_32K_SPEC.json')['status']=='FROZEN'
    hw=device(6,cfg); busy={'UNAUTHORIZED_TARGET_COMPUTE_PID','UNEXPLAINED_TARGET_MEMORY_OR_UTILIZATION'}
    while not hw['exclusive'] and set(hw['failure_conditions']).issubset(busy):
        print('AWAITING_NATURAL_GPU6_RELEASE',flush=True); time.sleep(30); cfg,sel,freeze=verify(); hw=device(6,cfg)
    assert hw['exclusive'],('GPU6_PRELAUNCH_QUERY_OR_IDENTITY_FAIL',hw)
    session=f'ling_p1l7_v4_{mode}_{doc.removeprefix("pg19-test-")}_fresh{rep}'
    assert subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode!=0
    watchdog=1800 if mode!='formal' else checked(OUT/'FROZEN_FORMAL_32K_SPEC.json')['no_progress_seconds'][doc]
    emit(RUNS/(t+'.LAUNCH_AUTHORIZATION.json'),dict(epoch=time.time(),gpu=6,gpu_uuid=UUID,document=doc,mode=mode,repeat=rep,
        endpoint=MODES[mode]['endpoint'],session=session,hardware=hw,runner_sha256=digest(CODE/'v2_worker.py'),
        wrapper_sha256=digest(CODE/'v4_persistent_worker.py'),freeze_sha256=digest(FREEZE),source_commit=freeze['code_commit'],
        hardware_mapping_amendment_sha256=digest(OUT/'HARDWARE_MAPPING_AMENDMENT.json'),no_progress_seconds=watchdog,
        no_retry=True,formal_spec_sha256=digest(OUT/'FROZEN_FORMAL_32K_SPEC.json') if mode=='formal' else None))
    args=['env','PYTHONDONTWRITEBYTECODE=1','CUDA_VISIBLE_DEVICES=6','PYTHONHASHSEED=0','LING_GATED_RMSNORM_FIXED_CONFIG=BT16_W8',
        'CUBLAS_WORKSPACE_CONFIG=:4096:8','/data/zypan/envs/ling-kda/bin/python','-B','-u',str(CODE/'v4_persistent_worker.py'),
        '--gpu','6','--document',doc,'--mode',mode,'--repeat',str(rep)]
    subprocess.run(['tmux','new-session','-d','-s',session,'cd '+shlex.quote(str(CODE))+' && exec '+shlex.join(args)+' > '+shlex.quote(str(RUNS/(t+'.log')))+' 2>&1'],check=True)
    print('V4_LAUNCHED',t,flush=True); return t

def wait(t):
    while True:
        assert not LATCH.exists(),'V4_STOP_NO_RETRY'
        p=RUNS/(t+'.json'); prefix=RUNS/(t+'.PREFIX_FINGERPRINTS.json')
        if p.exists() and p.with_suffix('.sha256.json').exists() and prefix.exists() and prefix.with_suffix('.sha256.json').exists() and not alive(t):
            assert checked(p)['status']=='PASS'; assert checked(prefix)['status']=='PASS'
            emit(RUNS/(t+'.EXIT_OBSERVED.json'),dict(observed_epoch=time.time(),natural_exit=True,no_signals=True))
            print('V4_WORKER_COMPLETE',t,flush=True); return
        if (RUNS/(t+'.STOP.json')).exists(): raise RuntimeError('V4_WORKER_STOP')
        r=checked(RUNS/(t+'.LAUNCH_AUTHORIZATION.json')); log=RUNS/(t+'.log')
        if time.time()-max(r['epoch'],log.stat().st_mtime if log.exists() else r['epoch'])>r['no_progress_seconds']:
            stop('NO_PROGRESS',r['mode'],worker_not_signaled=True); raise RuntimeError('NO_PROGRESS_SAFE_TOKEN_LATCH')
        time.sleep(5)

def qualification_document_gate(doc):
    cfg,sel,freeze=verify()
    a,p=complete('qualification',doc,6,0); b,q=complete('qualification',doc,6,1)
    exact=a['numeric_fingerprint']==b['numeric_fingerprint'] and fingerprint(a['stream'])==fingerprint(b['stream'])
    pa=checked(RUNS/(tag('qualification',doc,6,0)+'.PREFIX_FINGERPRINTS.json'))
    pb=checked(RUNS/(tag('qualification',doc,6,1)+'.PREFIX_FINGERPRINTS.json'))
    exact=exact and pa['prefixes']==pb['prefixes']
    if doc=='pg19-test-5956':
        exact=exact and pa['prefixes']['512']['FP']==checked(OUT/'CROSS_DEVICE_FP_REPEATABILITY.json')['FP_logits_sha256'][0]
    row=dict(document=doc,gpu=6,gpu_uuid=UUID,status='PASS' if exact else 'FAIL',
        canonical_sha256=[digest(p),digest(q)],fingerprints=[a['numeric_fingerprint'],b['numeric_fingerprint']],
        FP_full_logit_sha256=[a['stream']['fp_reference_full_logit_sha256'],b['stream']['fp_reference_full_logit_sha256']],
        real_endpoint=512,independent_fresh_processes=2,all_integrity_pass=True,source_cross_run_PASS_not_inherited=True)
    emit(OUT/(doc+'_512X2_GATE.json'),row)
    if not exact:
        emit(OUT/'GPU_512X2_QUALIFICATION.json',dict(status='FAIL',first_failed_document=doc,
            documents=[checked(OUT/(e['document']+'_512X2_GATE.json')) if (OUT/(e['document']+'_512X2_GATE.json')).exists()
                else dict(document=e['document'],status='NOT_RUN') for e in sel['documents']],
            original_exact_repeat_rule_unchanged=True))
        emit(OUT/'V4_DIAGNOSTIC_REQUIRED.json',dict(status='SCIENCE_FAIL_CLOSED_TARGETED_DIAGNOSIS_ONLY',
            reason='FULL_FP_STUDENT_FRESH_EXACT_REPEAT_FAIL',failed_document=doc,no_micro_formal=True,maximum_operator_repairs=1))
        raise RuntimeError('QUALIFICATION_FRESH_EXACT_REPEAT_FAIL')
    print('V4_DOCUMENT_512X2_PASS',doc,flush=True)

def qualification_gate():
    cfg,sel,freeze=verify(); docs=[]; diagnostics=[]
    for e in sel['documents']:
        doc=e['document']; row=checked(OUT/(doc+'_512X2_GATE.json')); assert row['status']=='PASS'
        docs.append(row)
        for rep in (0,1):
            diagnostics.extend(checked(RUNS/(tag('qualification',doc,6,rep)+'.REDUCER_DIAGNOSTIC.json'))['samples'])
    emit(OUT/'GPU_512X2_QUALIFICATION.json',dict(status='PASS',documents=docs,
        failed_documents=[],reducer_diagnostics=diagnostics,original_exact_repeat_rule_unchanged=True,
        full18KDA6MLAconv_real_QDQ_next_kernel_storage_finite=True,epoch=time.time()))
    print('V4_BOTH_DOCUMENT_512X2_PASS',flush=True)
