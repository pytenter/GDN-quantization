"""New V2 worker, immutable V1 numerical engine and inputs; no retries."""
import argparse
import gc
import importlib.metadata
import json
import os
import shutil
import time
import traceback
from pathlib import Path
from v2_common import *
from v2_runtime import initialize,snapshot
from persistent_engine_v1 import stream

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--gpu',type=int,choices=(3,6),required=True)
    ap.add_argument('--document',required=True); ap.add_argument('--mode',choices=tuple(MODES),required=True)
    ap.add_argument('--repeat',type=int,choices=(0,1),default=0)
    args=ap.parse_args(); start=time.time(); capture=None; gate='WORKER_IDENTITY'; runtime=[]; host={}
    t=tag(args.mode,args.document,args.gpu,args.repeat); path=RUNS/(t+'.json'); fail=RUNS/(t+'.STOP.json')
    assert not path.exists() and not fail.exists(),'ALREADY_EXECUTED_NO_RETRY'
    try:
        cfg,sel,freeze=verify(); entry=next(x for x in sel['documents'] if x['document']==args.document)
        assert entry['gpu']==args.gpu and os.environ.get('CUDA_VISIBLE_DEVICES')==str(args.gpu)
        receipt=checked(RUNS/(t+'.LAUNCH_AUTHORIZATION.json'))
        assert receipt['runner_sha256']==digest(__file__) and receipt['freeze_sha256']==digest(FREEZE)
        assert receipt['repeat']==args.repeat and receipt['mode']==args.mode
        emit(RUNS/(t+'.STARTED.json'),dict(pid=os.getpid(),started_epoch=start,gpu_uuid=receipt['gpu_uuid'],
            document=args.document,mode=args.mode,repeat=args.repeat,runner_sha256=digest(__file__)))
        gate='STRICT_DETERMINISTIC_INITIALIZATION'
        import torch
        runtime.append(initialize(torch,cfg['seed'])); print(json.dumps(dict(event='runtime_init',**runtime[-1])),flush=True)
        gate='LOAD_INHERITED_HELPERS'
        base=load_module(PRIOR/'run_short_horizon_proxy_v2.py','v2_verified_proxy')
        direction=load_module(base.DIRECTION_RUNNER,'v2_verified_direction')
        host=direction.gpu_guard(torch,args.gpu)
        assert all(host[k]==cfg['runtime'][k] for k in ('kernel','driver','torch','cuda_runtime','selector'))
        _,_,capture_module,core=direction.load_assets(args.gpu)
        runtime.append(snapshot(torch,'after_inherited_helpers',cfg['seed']))
        import run_phase_b_calibration_v2 as input_adapter
        input_adapter.ROOT=OLD  # namespace-only read path; never execute old main/verify/guard
        conditions=[entry['pair']['condition_a'],entry['pair']['condition_b']]
        document,tokens,token_path,rotations,rotation_ids=input_adapter.load_inputs(torch,cfg,args.document,conditions)
        gate='MODEL_LOAD'; tick=time.perf_counter(); capture=capture_module.Capture()
        assert torch.are_deterministic_algorithms_enabled()  # retained original gate
        runtime.append(snapshot(torch,'after_Capture_before_forward',cfg['seed']))
        print(json.dumps(dict(event='runtime_post_capture',**runtime[-1])),flush=True)
        capture.model.eval(); assert len(capture.model.model.layers)==24
        assert not any(p.requires_grad for p in capture.model.parameters())
        rotations={c:r.to(capture.device) for c,r in rotations.items()}
        loading=time.perf_counter()-tick; samples=[]; last_progress=time.monotonic()
        def guard():
            assert not LATCH.exists(),'V2_PEER_STOP_NO_RETRY'
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            assert time.monotonic()-last_progress<receipt['no_progress_seconds'],'NO_PROGRESS_WATCHDOG'
        def progress(pos,end,elapsed):
            nonlocal last_progress
            last_progress=time.monotonic()
            hw=device(args.gpu,cfg,os.getpid()); assert hw['exclusive'],'MIDRUN_EXCLUSIVITY_FAIL'
            assert not Path('/var/run/reboot-required').exists() and not Path('/run/systemd/shutdown/scheduled').exists()
            assert digest(__file__)==receipt['runner_sha256'] and digest(FREEZE)==receipt['freeze_sha256']
            sample=dict(position=pos,endpoint=end,elapsed_seconds=elapsed,
                allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                disk_free_bytes=shutil.disk_usage(OUT).free,tmp_free_bytes=shutil.disk_usage('/tmp').free,
                deterministic_algorithms_enabled=torch.are_deterministic_algorithms_enabled(),
                pid=os.getpid(),gpu_uuid=receipt['gpu_uuid'])
            assert sample['disk_free_bytes']>=2*1024**3,'PHYSICAL_STORAGE_EXHAUSTION'
            samples.append(sample); print(json.dumps(dict(event='persistent_progress',**sample)),flush=True)
        gate='FULL_PERSISTENT_STREAM'; mode=MODES[args.mode]; guard(); torch.cuda.reset_peak_memory_stats()
        result=stream(torch,direction,core,capture,tokens,rotations,cfg['layers'],mode['endpoint'],mode['primary'],
            mode['secondary'],guard,fp_replay=args.mode=='qualification',progress=progress)
        numeric_fingerprint=fingerprint(result)
        runtime.append(snapshot(torch,'after_full_stream',cfg['seed']))
        memory=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            total_bytes=torch.cuda.get_device_properties(0).total_memory)
        integrity={x['condition']:x['persistent_integrity'] for x in result['results']}
        assert all(x['all_finite'] for x in result['results'])
        for a in rotation_ids.values(): assert digest(a['path'])==a['sha256'],'ROTATION_MUTATION'
        capture.close(); capture=None; gc.collect(); torch.cuda.empty_cache(); end=time.time()
        packages={n:importlib.metadata.version(n) for n in ('torch','fla-core','triton','transformers')}
        # Micro output deliberately excludes KL values/trace/ranking. The unchanged engine
        # evaluates them, but controller decisions can access only mechanical/performance evidence.
        if args.mode=='micro':
            result={k:v for k,v in result.items() if k!='results'}
        out=dict(status='PASS',valid=True,document=args.document,gpu=args.gpu,gpu_uuid=receipt['gpu_uuid'],
            mode=args.mode,repeat=args.repeat,endpoint=mode['endpoint'],host=host,packages=packages,
            runner_sha256=digest(__file__),engine_sha256=digest(CODE/'persistent_engine_v1.py'),freeze_sha256=digest(FREEZE),
            code_commit=freeze['code_commit'],pair_selection_sha256=digest(OLD/'PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json'),
            token_file=dict(path=str(token_path),sha256=digest(token_path)),document_identity=document,rotation_artifacts=rotation_ids,
            conditions=conditions,predicted_winners=entry['predicted_winners'],stream=result,persistent_integrity=integrity,
            numeric_fingerprint=numeric_fingerprint,runtime_telemetry=runtime,progress_memory_storage=samples,
            gpu_memory=memory,model_loading_seconds=loading,started_epoch=start,completed_epoch=end,total_wall_seconds=end-start,
            accounted_gpu_hours=(end-start)/3600,micro_KL_not_published_or_used=args.mode=='micro',
            qdq_carrier='FP32 tensor containing dequantized INT8 semantics; NOT physical resident INT8',
            new_rotation_training=False,optimizer_step=False,scientific_primary_result=args.mode=='formal')
        assert base.finite_tree(out),'NONFINITE_CANONICAL'
        emit(path,out); print(json.dumps(dict(event='COMPLETE',mode=args.mode,path=str(path),sha256=digest(path))),flush=True)
    except Exception as exc:
        if capture is not None:
            try: capture.close()
            except Exception: pass
        stopped=time.time()
        evidence=dict(status='STOP',classification='ENGINEERING_OR_NUMERICAL_STOP',first_failure_gate=gate,
            error=repr(exc),traceback=traceback.format_exc(),document=args.document,gpu=args.gpu,
            gpu_uuid=receipt['gpu_uuid'] if 'receipt' in locals() else None,mode=args.mode,repeat=args.repeat,
            actual_runtime=host,runtime_telemetry=runtime,pid=os.getpid(),started_epoch=start,stopped_epoch=stopped,
            accounted_gpu_hours=(stopped-start)/3600,runner_sha256=digest(__file__),no_retry=True)
        emit(fail,evidence); stop(repr(exc),args.mode,run_stop=str(fail)); raise

if __name__=='__main__': main()
