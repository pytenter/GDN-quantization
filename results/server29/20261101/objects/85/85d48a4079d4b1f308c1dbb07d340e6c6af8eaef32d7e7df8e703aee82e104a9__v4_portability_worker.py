"""Only the two original conditions and two original matched H128 windows."""
import argparse
import gc
import hashlib
import json
import os
import time
import traceback
from v4_common import *
from v2_runtime import initialize,snapshot
from objective import original_helpers
from run_matched_l7 import TRAINER,OutputTap,ObservedRunner,ObservedCore,augment

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repeat',type=int,choices=(0,1),required=True)
    rep=ap.parse_args().repeat; t=tag('portability','pg19-test-5956',6,rep)
    started=time.time(); capture=tap=None; path=RUNS/(t+'.json'); failure=RUNS/(t+'.STOP.json')
    assert not path.exists() and not failure.exists(),'NO_RETRY'
    try:
        cfg,sel,freeze=verify(); receipt=checked(RUNS/(t+'.LAUNCH_AUTHORIZATION.json'))
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='6'
        assert receipt['runner_sha256']==digest(__file__) and receipt['freeze_sha256']==digest(FREEZE)
        emit(RUNS/(t+'.PROCESS_IDENTITY.json'),dict(identity=proc_identity(os.getpid()),gpu=6,gpu_uuid=UUID))
        emit(RUNS/(t+'.STARTED.json'),dict(pid=os.getpid(),started_epoch=started,mode='portability',repeat=rep))
        import torch
        runtime=[initialize(torch,cfg['seed'])]
        base=load_module(PRIOR/'run_short_horizon_proxy_v2.py','v4_port_original_proxy')
        # The old time-only budget guard is replaced with V4 safety guards. No
        # score/model/quantizer call is replaced; no administrative GPUh cap.
        last_check=time.monotonic(); steps=0
        def guard():
            assert not LATCH.exists(),'SAFE_TOKEN_STOP'
            assert torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
            assert time.monotonic()-last_check<1800,'NO_PROGRESS'
        base.budget_guard=guard
        direction=load_module(base.DIRECTION_RUNNER,'v4_port_original_direction')
        host=direction.gpu_guard(torch,6)
        assert host['gpu_uuid']==UUID and all(host[k]==cfg['runtime'][k] for k in ('kernel','driver','torch','cuda_runtime','selector'))
        _,_,capture_module,core=direction.load_assets(6)
        import run_phase_b_calibration_v2 as adapter
        adapter.ROOT=OLD
        entry=next(e for e in sel['documents'] if e['document']=='pg19-test-5956')
        conditions=[entry['pair']['condition_a'],entry['pair']['condition_b']]
        document,tokens,token_path,rotations,rotation_ids=adapter.load_inputs(torch,cfg,entry['document'],conditions)
        tick=time.perf_counter(); capture=capture_module.Capture(); capture.model.eval()
        assert not any(p.requires_grad for p in capture.model.parameters())
        tap=OutputTap(torch,capture,cfg['layers']); helpers=original_helpers(torch,TRAINER)
        observer=ObservedRunner(base,direction,torch,core,capture,tap,helpers,cfg['windows'],cfg['layers'])
        observed_core=ObservedCore(core,observer); loading=time.perf_counter()-tick
        original_step=observer.model_step
        def observed_step(*args):
            nonlocal steps,last_check
            guard(); value=original_step(*args); steps+=1
            if steps%128==0:
                hw=device(6,cfg,os.getpid()); assert hw['exclusive'],'MIDRUN_GPU_IDENTITY_FAIL'
                print(json.dumps(dict(event='portability_progress',calls=steps,elapsed_seconds=time.time()-started,
                    allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())),flush=True)
                last_check=time.monotonic()
            return value
        observer.model_step=observed_step
        runtime.append(snapshot(torch,'before_portability_forward',cfg['seed']))
        torch.cuda.reset_peak_memory_stats()
        records,fp_logits,fp_states,fp_seconds=base.extract_fp(torch,observer,capture,tokens,1152,cfg['windows'],cfg['layers'])
        parity=base.replay_parity(torch,core,records,fp_states); assert parity['passed'],'FP_REPLAY_PARITY_FAIL'
        identity=dict(post_o_proj=observer.fp_output_hashes,full_logits=observer.fp_logit_hashes,
            drivers={k:direction.tensor_sha(v) for k,v in records.items()},
            checkpoint_states={p:direction.tensor_sha(v) for p,v in fp_states.items()})
        covariances=base.build_covariances(torch,records,cfg['windows']); observer.mode='student'; results=[]
        for condition in conditions:
            observer.scores,observer.reference_states={},None
            result=base.run_condition(torch,observer,observed_core,capture,tokens,records,fp_logits,covariances,
                rotations[condition].to(capture.device),condition,1,cfg['windows'],[128],cfg['layers'],cfg['layers'].index(20))
            augment(result,observer,cfg['windows']); assert result['mechanical_gates']['passed'],'REAL_QDQ_CONSUMPTION_FAIL'
            results.append(result)
        assert len(observer.helper_forward_checks)==72 and all(x['bitwise_equal'] and x['identity_loss']==0 for x in observer.helper_forward_checks)
        payload=dict(fp_reference_identity=identity,FP_content_sha256=base.canonical_hash(identity),
            results=[dict(condition=r['condition'],windows=r['windows'],aggregate=r['aggregate'],mechanical_gates=r['mechanical_gates']) for r in results])
        fp=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        memory=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),total_bytes=torch.cuda.get_device_properties(0).total_memory)
        runtime.append(snapshot(torch,'after_portability_forward',cfg['seed']))
        tap.close(); tap=None; capture.close(); capture=None; gc.collect(); torch.cuda.empty_cache()
        emit(path,dict(status='PASS_SINGLE_PROCESS_NOT_PORTABILITY_APPROVAL',document=entry['document'],gpu=6,gpu_uuid=UUID,
            endpoint=1152,repeat=rep,runner_sha256=digest(__file__),freeze_sha256=digest(FREEZE),code_commit=freeze['code_commit'],
            historical_pair_selection_sha256=digest(OLD/'PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json'),
            token_file=dict(path=str(token_path),sha256=digest(token_path)),rotation_artifacts=rotation_ids,
            numerical_payload=payload,numeric_fingerprint=fp,results=results,fp_replay_parity=parity,
            original_helper_forward_checks=observer.helper_forward_checks,runtime_telemetry=runtime,host=host,
            started_epoch=started,completed_epoch=time.time(),loading_seconds=loading,fp_seconds=fp_seconds,gpu_memory=memory,
            selector_not_run=True,original_two_conditions_only=True,original_loss_math_unchanged=True,
            historical_512_training_sampling_replayed=False,no_long_horizon_outcome_used=True))
        print('V4_PORTABILITY_PROCESS_COMPLETE',rep,flush=True)
    except Exception as exc:
        if tap is not None: tap.close()
        if capture is not None: capture.close()
        emit(failure,dict(status='STOP',reason=repr(exc),traceback=traceback.format_exc(),started_epoch=started,stopped_epoch=time.time()))
        stop(repr(exc),'PORTABILITY_WORKER',run_stop=str(failure)); raise

if __name__=='__main__': main()
