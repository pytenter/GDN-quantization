"""Original production FP step at 512 geometry; students omitted, no math edits."""
import argparse
import gc
import hashlib
import json
import os
import shutil
import time
import traceback
from v4_stage1_common import *
from v2_runtime import initialize,snapshot
from run_matched_l7 import PRIOR,load_module
import persistent_engine_v1 as engine


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repeat',type=int,choices=(0,1),required=True)
    rep=ap.parse_args().repeat; t=name(rep); started=time.time(); capture=None; audit=None
    path=RUNS/(t+'.json'); failure=RUNS/(t+'.STOP.json'); runtime=[]
    assert not path.exists() and not failure.exists(),'ALREADY_EXECUTED_NO_RETRY'
    try:
        cfg,freeze=verify(); receipt=checked(RUNS/(t+'.LAUNCH_AUTHORIZATION.json'))
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='6'
        assert receipt['runner_sha256']==digest(__file__) and receipt['freeze_sha256']==digest(FREEZE)
        emit(RUNS/(t+'.PROCESS_IDENTITY.json'),dict(identity=proc_identity(os.getpid()),gpu=6,
            gpu_uuid=UUID,runner_sha256=digest(__file__),receipt_sha256=digest(RUNS/(t+'.LAUNCH_AUTHORIZATION.json'))))
        emit(RUNS/(t+'.STARTED.json'),dict(pid=os.getpid(),started_epoch=started,gpu_uuid=UUID,repeat=rep,stage='FP512'))
        import torch
        runtime.append(initialize(torch,cfg['seed']))
        base=load_module(PRIOR/'run_short_horizon_proxy_v2.py','v4_fp_verified_proxy')
        direction=load_module(base.DIRECTION_RUNNER,'v4_fp_verified_direction')
        host=direction.gpu_guard(torch,6)
        assert host['gpu_uuid']==UUID
        assert all(host[k]==cfg['runtime'][k] for k in ('kernel','driver','torch','cuda_runtime','selector'))
        _,_,capture_module,core=direction.load_assets(6)
        runtime.append(snapshot(torch,'after_original_helpers',cfg['seed']))
        import run_phase_b_calibration_v2 as adapter
        adapter.ROOT=OLD
        document,tokens,token_path,rotations,rotation_ids=adapter.load_inputs(torch,cfg,'pg19-test-5956',['H_to_L6_a50','H_to_L7_a50'])
        tick=time.perf_counter(); capture=capture_module.Capture(); capture.model.eval()
        assert len(capture.model.model.layers)==24 and not any(p.requires_grad for p in capture.model.parameters())
        loading=time.perf_counter()-tick
        runtime.append(snapshot(torch,'before_FP_forward',cfg['seed']))
        # Original production FP leg already installs this wrapper with active=None.
        # Retain that wrapper; original model_step and native cache shapes are unchanged.
        audit=engine.PersistentAudit(torch,direction,core,capture,cfg['layers'],sorted(rotations))
        audit.active=None
        cache=None; parity_state=None; h=hashlib.sha256(); token_hashes=[]; inventories={}; samples=[]
        raw=RUNS/(t+'.FP_LOGITS.bin'); dtype=None; shape=None; tensor_nbytes=None
        torch.cuda.reset_peak_memory_stats(); forward_start=time.perf_counter()
        with raw.open('xb') as saved:
            for position in range(512):
                assert not LATCH.exists(),'V4_SAFE_TOKEN_STOP'
                assert torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
                assert time.perf_counter()-forward_start<1800,'NO_PROGRESS_OR_UNKNOWN_RUNTIME'
                cache,logits,records=direction.model_step(torch,capture,cache,tokens[position],position)
                assert bool(torch.isfinite(logits).all()),'NONFINITE_FP_LOGITS'
                states=torch.cat([direction.cache_state(capture,cache,layer) for layer in cfg['layers']])
                if parity_state is None: parity_state=torch.zeros_like(states)
                record={key:torch.cat([records[layer][key] for layer in cfg['layers']]) for key in ('q','k','v','beta','log_decay')}
                with torch.inference_mode(): parity_state=core.update_pre_q(parity_state,record)
                assert torch.equal(parity_state,states),'ORIGINAL_FP_REPLAY_PARITY_FAIL'
                data=engine.tensor_bytes(torch,logits)
                if dtype is None: dtype=str(logits.dtype); shape=list(logits.shape); tensor_nbytes=len(data)
                assert dtype==str(logits.dtype) and shape==list(logits.shape) and tensor_nbytes==len(data)
                saved.write(data); h.update(position.to_bytes(8,'little')); h.update(data)
                token_hashes.append(hashlib.sha256(data).hexdigest())
                if position in (0,255,256,511):
                    inventories[str(position)]=engine.hybrid_inventory(torch,cache,position,cfg['layers'],24)
                if (position+1)%128==0:
                    hw=device(cfg,rep,os.getpid()); assert hw['exclusive'],'MIDRUN_UUID_COMPUTE_EXCLUSIVITY_FAIL'
                    sample=dict(position=position+1,elapsed_seconds=time.perf_counter()-forward_start,
                        allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        disk_free_bytes=shutil.disk_usage(OUT).free)
                    assert sample['disk_free_bytes']>=2*1024**3,'PHYSICAL_DISK_EXHAUSTION'
                    samples.append(sample); print(json.dumps(dict(event='FP512_progress',**sample)),flush=True)
                del logits
            saved.flush(); os.fsync(saved.fileno())
        forward_seconds=time.perf_counter()-forward_start
        runtime.append(snapshot(torch,'after_FP512',cfg['seed']))
        payload=dict(endpoint=512,FP_full_logit_sha256=h.hexdigest(),token_logit_sha256=token_hashes,
            hybrid_cache_inventories=inventories,fp_replay_parity_tokens=512,
            raw_tensor_layout=dict(dtype=dtype,shape=shape,tensor_bytes=tensor_nbytes),
            token_sha256=digest(token_path),original_engine_sha256=ENGINE_SHA,
            original_single_token_model_step=True,student_calls_omitted_only=True)
        fp=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        memory=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            total_bytes=torch.cuda.get_device_properties(0).total_memory)
        audit.close(); audit=None; capture.close(); capture=None; cache=None; gc.collect(); torch.cuda.empty_cache()
        end=time.time()
        emit(path,dict(status='PASS_SINGLE_PROCESS_NOT_512X2',stage='FP512',document='pg19-test-5956',gpu=6,gpu_uuid=UUID,
            repeat=rep,source_commit=freeze['code_commit'],runner_sha256=digest(__file__),freeze_sha256=digest(FREEZE),
            host=host,runtime_telemetry=runtime,numerical_payload=payload,numeric_fingerprint=fp,
            raw_FP_logits=dict(path=str(raw),sha256=digest(raw),bytes=raw.stat().st_size),
            document_identity=document,model_and_source_identity_sha256=digest(OUT/'V4_REUSED_SOURCE_IDENTITY.json'),
            loading_seconds=loading,forward_seconds=forward_seconds,gpu_memory=memory,progress=samples,
            started_epoch=started,completed_epoch=end,wall_seconds=end-started,
            no_student_no_scientific_P1_L7_outcome=True,no_optimizer_step=True))
        print('V4_FP512_COMPLETE',rep,digest(path),flush=True)
    except Exception as exc:
        if audit is not None: audit.close()
        if capture is not None: capture.close()
        emit(failure,dict(status='STOP',reason=repr(exc),traceback=traceback.format_exc(),
            pid=os.getpid(),repeat=rep,started_epoch=started,stopped_epoch=time.time()))
        stop(repr(exc),stage='FP512',failure_path=str(failure)); raise


if __name__=='__main__': main()
