"""CPU-only explicit stage gates; no scientific outcomes used for resources."""
import argparse
import fcntl
import shlex
import subprocess
import time
from v2_common import *

def launch(gpu,mode,rep):
    cfg,sel,freeze=verify(); entry=next(x for x in sel['documents'] if x['gpu']==gpu)
    doc=entry['document']; t=tag(mode,doc,gpu,rep)
    with (RUNS/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert not any((RUNS/(t+s)).exists() for s in ('.LAUNCH_AUTHORIZATION.json','.STARTED.json','.json','.STOP.json','.log')),'NO_DUPLICATE_OR_RETRY'
        if mode=='qualification' and rep==1:
            for x in sel['documents']: complete('qualification',x['document'],x['gpu'],0)
        if mode in ('micro','formal'):
            assert checked(OUT/'GPU_QUALIFICATION_RESULTS.json')['status']=='PASS'
        if mode=='formal':
            assert checked(OUT/'RESOURCE_FEASIBILITY_REPORT.json')['status']=='PASS'
            assert checked(OUT/'FROZEN_PREMEASUREMENT_SPEC.json')['status']=='FROZEN'
            raise RuntimeError('FORMAL_CONTROLLER_NOT_YET_QUALIFIED_NO_LAUNCH')
        assert rep==0 or mode=='qualification'
        hw=device(gpu,cfg)
        if not hw['exclusive']:
            stop('EXTERNAL_DEVICE_OCCUPANCY_STOP_RESOURCE_QUALIFICATION',mode,hardware=hw)
            raise RuntimeError('DEVICE_BUSY_STOP_NO_WAIT_NO_SIGNAL_NO_ALTERNATE')
        session=f'ling_p1l7_v2_{mode}_gpu{gpu}_fresh{rep}'
        assert subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode!=0
        receipt=dict(epoch=time.time(),gpu=gpu,gpu_uuid=hw['gpu_uuid'],document=doc,mode=mode,repeat=rep,
            session=session,hardware=hw,runner_sha256=digest(CODE/'v2_worker.py'),freeze_sha256=digest(FREEZE),
            pair_selection_sha256=digest(OLD/'PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json'),
            no_progress_seconds=1800,administrative_total_wall_cap=None,complete_stage_roster_required=True)
        emit(RUNS/(t+'.LAUNCH_AUTHORIZATION.json'),receipt)
        args=['env','PYTHONDONTWRITEBYTECODE=1',f'CUDA_VISIBLE_DEVICES={gpu}','PYTHONHASHSEED=0',
            'LING_GATED_RMSNORM_FIXED_CONFIG=BT16_W8','CUBLAS_WORKSPACE_CONFIG=:4096:8',
            '/data/zypan/envs/ling-kda/bin/python','-B','-u',str(CODE/'v2_worker.py'),
            '--document',doc,'--gpu',str(gpu),'--mode',mode,'--repeat',str(rep)]
        command='cd '+shlex.quote(str(CODE))+' && exec '+shlex.join(args)+' > '+shlex.quote(str(RUNS/(t+'.log')))+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',session,command],check=True)
        print('V2_STAGE_LAUNCHED',mode,rep,doc,hw['gpu_uuid'])

def aggregate():
    cfg,sel,_=verify(); rows=[]
    for x in sel['documents']:
        a,p=complete('qualification',x['document'],x['gpu'],0)
        b,q=complete('qualification',x['document'],x['gpu'],1)
        assert a['numeric_fingerprint']==b['numeric_fingerprint'],'FRESH_PROCESS_NUMERIC_REPEAT_FAIL'
        assert a['stream']['fp_replay_parity_tokens']==b['stream']['fp_replay_parity_tokens']==512
        assert read(p.with_name(p.stem+'.STARTED.json'))['pid']!=read(q.with_name(q.stem+'.STARTED.json'))['pid']
        rows.append(dict(document=x['document'],gpu=x['gpu'],gpu_uuid=a['gpu_uuid'],status='PASS',
            fresh_process_repeat_exact=True,canonical_sha256=[digest(p),digest(q)],
            canonical_paths=[str(p),str(q)],numeric_fingerprint=a['numeric_fingerprint'],
            persistent_integrity=a['persistent_integrity'],runtime_telemetry=a['runtime_telemetry']+b['runtime_telemetry']))
    emit(OUT/'GPU_QUALIFICATION_RESULTS.json',dict(status='PASS',documents=rows,all18_KDA_all6_MLA_conv=True,
        same_GPU_FP_replay_exact=True,fresh_process_512x2_exact=True,cross_GPU_identity_not_required=True))
    emit(OUT/'SAME_GPU_FP_PARITY_REPORT.json',dict(status='PASS',stage='512x2 qualification only',documents=rows,
        full_length_32k_FP_status='NOT_RUN',fp_full_logit_and_native_recurrence_repeat_bitwise=True))
    emit(OUT/'PERSISTENT_QDQ_SEMANTICS_REPORT.json',dict(status='PASS',stage='512x2 qualification only',documents=rows,
        carrier='FP32 dequantized INT8 semantics; not physical resident INT8 storage',anchor_continuation_micro='NOT_RUN'))
    emit(OUT/'DETERMINISTIC_RUNTIME_INIT_REPORT.json',dict(status='PASS',actual_child_runtime=True,documents=rows,
        strict=True,warn_only=False,flag_verified_before_helpers_after_Capture_after_stream=True))
    print('BOTH_GPU_QUALIFICATION_PASS_FRESH_PROCESS_EXACT')

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    RUNS.mkdir(exist_ok=True)
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=('launch','aggregate','status'))
    ap.add_argument('--gpu',type=int,choices=(3,6)); ap.add_argument('--mode',choices=tuple(MODES))
    ap.add_argument('--repeat',type=int,choices=(0,1),default=0); a=ap.parse_args()
    if a.action=='launch': launch(a.gpu,a.mode,a.repeat)
    elif a.action=='aggregate':
        try: aggregate()
        except Exception as e: stop(repr(e),'qualification_aggregation'); raise
    else:
        account=ledger(); p=OUT/f'GPU_COST_LEDGER_SNAPSHOT_{time.time_ns()}.json'; emit(p,account)
        print(json.dumps(dict(ledger_path=str(p),v2_gpu_hours=account['v2_gpu_hours'],jobs=account['jobs'],
            latch=read(LATCH) if LATCH.exists() else None)))

if __name__=='__main__': main()
