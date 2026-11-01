"""Exactly two serial fresh FP processes; no retries or stage expansion."""
import json
import math
import shlex
import subprocess
import time
from v4_stage1_common import *


def launch(rep):
    cfg,freeze=verify(); t=name(rep)
    assert not any((RUNS/(t+s)).exists() for s in ('.LAUNCH_AUTHORIZATION.json','.STARTED.json','.json','.STOP.json','.log'))
    hw=device(cfg)
    busy={'UNAUTHORIZED_TARGET_COMPUTE_PID','UNEXPLAINED_TARGET_MEMORY_OR_UTILIZATION'}
    while not hw['exclusive'] and set(hw['failure_conditions']).issubset(busy):
        print('V4_AWAITING_NATURAL_GPU6_RELEASE',flush=True); time.sleep(30)
        cfg,freeze=verify(); hw=device(cfg)
    assert hw['exclusive'],('PRELAUNCH_UUID_QUERY_OR_IDENTITY_FAIL',hw)
    session=f'ling_p1l7_v4_fp512_gpu6_fresh{rep}'
    assert subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode!=0
    worker=CODE/'v4_fp512_worker.py'
    emit(RUNS/(t+'.LAUNCH_AUTHORIZATION.json'),dict(epoch=time.time(),gpu=6,gpu_uuid=UUID,
        document='pg19-test-5956',repeat=rep,session=session,runner_sha256=digest(worker),
        freeze_sha256=digest(FREEZE),source_commit=freeze['code_commit'],hardware=hw,endpoint=512,
        stage='FP512_CROSS_DEVICE_ONLY_NO_HARDWARE_AMENDMENT_YET',no_progress_seconds=1800))
    args=['env','PYTHONDONTWRITEBYTECODE=1','CUDA_VISIBLE_DEVICES=6','PYTHONHASHSEED=0',
          'LING_GATED_RMSNORM_FIXED_CONFIG=BT16_W8','CUBLAS_WORKSPACE_CONFIG=:4096:8',
          '/data/zypan/envs/ling-kda/bin/python','-B','-u',str(worker),'--repeat',str(rep)]
    command='cd '+shlex.quote(str(CODE))+' && exec '+shlex.join(args)+' > '+shlex.quote(str(RUNS/(t+'.log')))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,command],check=True)
    print('V4_FP512_LAUNCHED',rep,flush=True)


def wait(rep):
    t=name(rep); canonical=RUNS/(t+'.json')
    while True:
        assert not LATCH.exists(),'V4_STOP_NO_RETRY'
        if canonical.exists() and canonical.with_suffix('.sha256.json').exists() and not alive(rep):
            value=checked(canonical); assert value['status']=='PASS_SINGLE_PROCESS_NOT_512X2'
            return value
        fail=RUNS/(t+'.STOP.json')
        if fail.exists(): raise RuntimeError('V4_FP_WORKER_STOP')
        receipt=checked(RUNS/(t+'.LAUNCH_AUTHORIZATION.json')); log=RUNS/(t+'.log')
        updated=max(receipt['epoch'],log.stat().st_mtime if log.exists() else receipt['epoch'])
        if time.time()-updated>receipt['no_progress_seconds']:
            stop('NO_PROGRESS',stage='FP512',worker_not_signaled=True)
            raise RuntimeError('V4_NO_PROGRESS_SAFE_TOKEN_LATCH')
        time.sleep(5)


def compare(a,b):
    pa,pb=a['numerical_payload'],b['numerical_payload']
    changed=[i for i,(x,y) in enumerate(zip(pa['token_logit_sha256'],pb['token_logit_sha256'])) if x!=y]
    max_abs=0.; diff_sq=0.; norm_sq=0.; first=None; first_stats=None
    if changed:
        import numpy as np
        assert pa['raw_tensor_layout']==pb['raw_tensor_layout']
        size=pa['raw_tensor_layout']['tensor_bytes']; dtype=pa['raw_tensor_layout']['dtype']
        def decode(raw):
            if dtype=='torch.bfloat16':
                return (np.frombuffer(raw,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32).astype(np.float64)
            return np.frombuffer(raw,dtype={'torch.float32':np.float32,'torch.float16':np.float16}[dtype]).astype(np.float64)
        with Path(a['raw_FP_logits']['path']).open('rb') as fa, Path(b['raw_FP_logits']['path']).open('rb') as fb:
            for position in range(512):
                x,y=decode(fa.read(size)),decode(fb.read(size)); d=x-y
                assert np.isfinite(x).all() and np.isfinite(y).all()
                da=float(np.max(np.abs(d))); ds=float(np.sum(d*d)); ns=float(np.sum(x*x))
                max_abs=max(max_abs,da); diff_sq+=ds; norm_sq+=ns
                if position in changed and first is None:
                    first=position; first_stats=dict(max_abs=da,relative_L2=math.sqrt(ds/ns) if ns else None)
    exact=a['numeric_fingerprint']==b['numeric_fingerprint'] and pa==pb
    result=dict(status='PASS' if exact else 'FAIL',document='pg19-test-5956',gpu=6,gpu_uuid=UUID,
        actual_fresh_process_count=2,endpoint=512,original_exact_repeat_rule_unchanged=True,
        canonical_sha256=[digest(RUNS/(name(i)+'.json')) for i in (0,1)],
        FP_logits_sha256=[pa['FP_full_logit_sha256'],pb['FP_full_logit_sha256']],
        numerical_fingerprints=[a['numeric_fingerprint'],b['numeric_fingerprint']],
        first_inconsistent_zero_based_token=first,changed_token_count=len(changed),
        max_abs=max_abs,relative_L2=math.sqrt(diff_sq/norm_sq) if norm_sq else 0. if exact else None,
        first_token_stats=first_stats,raw_FP_logit_archives=[a['raw_FP_logits'],b['raw_FP_logits']],
        runtime_identities=[a['runtime_telemetry'],b['runtime_telemetry']],
        hardware_mapping_amendment_NOT_approved_by_FP_test_alone=True,
        no_student_or_formal_scientific_outcome=True,epoch=time.time())
    emit(OUT/'CROSS_DEVICE_FP_REPEATABILITY.json',result)
    if not exact:
        emit(OUT/'V4_DIAGNOSTIC_REQUIRED.json',dict(status='SCIENCE_FAIL_CLOSED_TARGETED_DIAGNOSIS_ONLY',
            reason='GPU6_DOC5956_FP_EXACT_REPEAT_FAIL',source_sha256=digest(OUT/'CROSS_DEVICE_FP_REPEATABILITY.json'),
            no_formal_or_portability_launch=True,maximum_evidence_based_operator_repairs=1))
    return result


def main():
    try:
        assert not (OUT/'STAGE1_SUPERVISOR_STARTED.json').exists(),'NO_DUPLICATE_CONTROLLER'
        cfg,freeze=verify()
        emit(OUT/'STAGE1_SUPERVISOR_STARTED.json',dict(pid=__import__('os').getpid(),epoch=time.time(),code_commit=freeze['code_commit']))
        values=[]
        for rep in (0,1):
            launch(rep); values.append(wait(rep))
        result=compare(*values)
        emit(OUT/'RESOURCE_USAGE_LEDGER_STAGE1.json',ledger())
        emit(OUT/'V4_STAGE1_COMPLETION.json',dict(status=result['status'],epoch=time.time(),
            next_stage='PORTABILITY_GATE' if result['status']=='PASS' else 'TARGETED_512_GEOMETRY_FORWARD_DIAGNOSIS',
            no_automatic_long_run_without_remaining_gates=True))
        print('V4_STAGE1_COMPLETE',result['status'],flush=True)
    except Exception as exc:
        stop(repr(exc),stage='STAGE1_SUPERVISION',no_signal=True); raise


if __name__=='__main__': main()
