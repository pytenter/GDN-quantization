"""Full16 short screening using unchanged original P1 and L7 helper calls."""
import argparse
import json
import os
import shutil
import sys
import time
import traceback
sys.dont_write_bytecode=True
from prospective_common import OUT,OLD,WT,sha,read,publish,verify,load_inputs
from objective import original_helpers
from run_matched_l7 import PRIOR,TRAINER,load_module,OutputTap,ObservedRunner,ObservedCore,augment
from v2_runtime import initialize,snapshot
from gpu_admission import device,proc_identity,route

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--document',required=True);a=ap.parse_args()
    folder=OUT/'screening';folder.mkdir(exist_ok=True);tag=a.document
    start=time.time();capture=tap=None
    try:
        cfg=verify();gpu=route(cfg);assert a.document in cfg['documents']
        assert not (OUT/'FROZEN_4DOC_SELECTION.json').exists(),'SCREENING_ALREADY_CLOSED'
        assert not (folder/(tag+'.json')).exists(),'NO_RETRY_NO_OVERWRITE'
        publish(folder/(tag+'.STARTED.json'),dict(document=a.document,pid=os.getpid(),started_epoch=start,
            source_commit=read(OUT/'SOURCE_CODE_FREEZE.json')['source_commit'],source_sha256=sha(__file__)))
        import torch
        runtime=[initialize(torch,cfg['seed'])];binding=proc_identity(os.getpid())
        assert device(gpu,cfg)['exclusive'],'GPU_ADMISSION_FAILED'
        base=load_module(PRIOR/'run_short_horizon_proxy_v2.py','prospective_original_proxy')
        assert base._DEADLINE==float('inf'),'NO_ARTIFICIAL_WALL_BUDGET'
        direction=load_module(base.DIRECTION_RUNNER,'prospective_original_direction')
        host=direction.gpu_guard(torch,gpu)
        assert host['gpu_uuid']==cfg['gpu_uuids'][str(gpu)]
        assert all(host[k]==cfg['runtime'][k] for k in ('kernel','driver','torch','cuda_runtime','selector'))
        _,_,capture_module,core=direction.load_assets(gpu)
        entry,tokens,token_path,rotations,rotation_ids=load_inputs(torch,cfg,a.document,cfg['conditions'])
        load_start=time.perf_counter();capture=capture_module.Capture();capture.model.eval()
        assert not any(p.requires_grad for p in capture.model.parameters())
        tap=OutputTap(torch,capture,cfg['layers']);helpers=original_helpers(torch,TRAINER)
        observer=ObservedRunner(base,direction,torch,core,capture,tap,helpers,cfg['windows'],cfg['layers'])
        observed_core=ObservedCore(core,observer);load_seconds=time.perf_counter()-load_start
        calls=[0]
        def progress_guard():
            assert not (OUT/'STOP.json').exists(),'SAFE_STEP_FIRST_FAILURE_STOP'
            calls[0]+=1
            if calls[0]%512==0:
                guard=device(gpu,cfg,os.getpid(),binding=binding,require_context=True,
                    evidence=folder/(tag+'.GPU_EXCLUSIVITY.jsonl'))
                assert guard['exclusive'],guard['failure_conditions']
                assert shutil.disk_usage(OUT).free>2*1024**3
                print(json.dumps(dict(event='short_progress',position=calls[0],document=a.document,
                    elapsed=time.time()-start,mode=observer.mode)),flush=True)
        base.budget_guard=progress_guard
        records,fp_logits,fp_states,fp_seconds=base.extract_fp(torch,observer,capture,tokens,1152,cfg['windows'],cfg['layers'])
        parity=base.replay_parity(torch,core,records,fp_states);assert parity['passed'],'FP_REPLAY_PARITY_FAIL'
        fp_identity=dict(post_o_proj=observer.fp_output_hashes,full_logits=observer.fp_logit_hashes,
            drivers={k:direction.tensor_sha(v) for k,v in records.items()},
            checkpoint_states={p:direction.tensor_sha(v) for p,v in fp_states.items()})
        covariances=base.build_covariances(torch,records,cfg['windows']);observer.mode='student'
        marks=[dict(stage='FP_teacher',peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())]
        results=[]
        for condition in cfg['conditions']:
            base.budget_guard();observer.scores,observer.reference_states={},None
            result=base.run_condition(torch,observer,observed_core,capture,tokens,records,fp_logits,covariances,
                rotations[condition].to(capture.device),condition,1,cfg['windows'],[128],cfg['layers'],cfg['layers'].index(cfg['auxiliary_layer']))
            augment(result,observer,cfg['windows'])
            assert result['mechanical_gates']['passed'],'REAL_QDQ_WRITEBACK_GATE_FAIL'
            assert result['tokens_processed']==1152 and result['l7_scored_token_count']==256
            marks.append(dict(stage=condition,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
            publish(folder/(tag+'.'+condition+'.json'),result);results.append(result)
            print(json.dumps(dict(event='condition_complete',document=a.document,condition=condition,
                position=calls[0],elapsed=time.time()-start)),flush=True)
        runtime.append(snapshot(torch,'after_full16_screen',cfg['seed']))
        assert [x['condition'] for x in results]==cfg['conditions']
        for item in rotation_ids.values():assert sha(item['path'])==item['sha256']
        output=dict(status='PASS',valid=True,document=a.document,endpoint=1152,results=results,
            document_identity=entry,token_file=dict(path=str(token_path),sha256=sha(token_path)),rotation_artifacts=rotation_ids,
            source_commit=read(OUT/'SOURCE_CODE_FREEZE.json')['source_commit'],runtime=runtime,host=host,
            runner_sha256=sha(__file__),model_loading_seconds=load_seconds,
            fp_reference=dict(source='SHARED_CURRENT_RUNTIME_SAME_GPU_REFERENCE',tokens=1152,
                tensor_identities=fp_identity,content_sha256=base.canonical_hash(fp_identity),replay_parity=parity,
                helper_real_forward_checks=observer.helper_forward_checks,elapsed_seconds=fp_seconds),
            memory=marks,all_finite=True,started_epoch=start,completed_epoch=time.time(),
            long_horizon_outcome_generated_or_used=False,gradients_or_optimizer_used=False)
        assert base.finite_tree(output),'NONFINITE_SCREENING_OUTPUT'
        publish(folder/(tag+'.json'),output)
        print(json.dumps(dict(event='full16_screen_complete',document=a.document,result_sha256=sha(folder/(tag+'.json')))),flush=True)
    except BaseException as exc:
        failure=dict(status='STOP',stage='SHORT_SCREEN',document=a.document,reason=repr(exc),
            traceback=traceback.format_exc(),started_epoch=start,stopped_epoch=time.time(),no_retry=True)
        if not (folder/(tag+'.STOP.json')).exists():publish(folder/(tag+'.STOP.json'),failure)
        if not (OUT/'STOP.json').exists():publish(OUT/'STOP.json',failure)
        raise
    finally:
        if tap is not None:tap.close()
        if capture is not None:capture.close()

if __name__=='__main__':main()
