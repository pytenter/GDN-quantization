"""One fresh claimed HMMT trajectory. No gold, scorer, retry or prefix resume."""
import argparse,gc,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import execution_common as C

def run(a):
    target=ROOT/'formal'/a.unit
    assert target.is_dir() and not (target/'WORKER_IDENTITY.json').exists() and not (target/'RESULT.json').exists()
    C.source(ROOT,a.source_sha);assert C.sha(ROOT/'GPU_PERMIT.json')==a.permit_sha
    permit=json.loads((ROOT/'GPU_PERMIT.json').read_text());assert permit['formal_target']==99 and permit['reuse']==0
    job=next(x for x in json.loads((ROOT/'FORMAL_JOBS.json').read_text())['jobs'] if x['id']==a.unit)
    assert os.environ['CUDA_VISIBLE_DEVICES']==C.UUIDS[a.gpu]
    before=C.model_pins(ROOT);ident=C.identity(os.getpid())
    C.atomic(target/'WORKER_IDENTITY.json',dict(identity=ident,job=job,gpu=a.gpu,CVD=os.environ['CUDA_VISIBLE_DEVICES'],source_sha256=a.source_sha,permit_sha256=a.permit_sha))
    def progress(phase,**kw):print(json.dumps(dict(phase=phase,monotonic=time.monotonic(),**kw)),flush=True)
    try:
        import pace_runtime as R
        from hmmt_engine import HMMTEngine
        from model_engineering import runtime_identity
        progress('SOURCE_MODEL_STAT_ADMISSION_AND_LOAD')
        t=time.monotonic();model,tokenizer,native=R._load_frozen_model(ROOT);load=time.monotonic()-t
        runtime=runtime_identity();assert runtime['driver_uuid']==C.UUIDS[a.gpu]
        C.atomic(target/'LOADED_REGULAR_MAPPED_SHA.json',dict(files=C.regular_maps(),runtime=runtime))
        item=next(x for x in map(json.loads,(ROOT/'protocol/HMMT_REQUESTS.jsonl').read_text().splitlines()) if x['problem_idx']==job['problem_idx'])
        assert set(item)=={'input_ids','input_ids_sha256','input_tokens','messages','problem_idx','problem_sha256','problem_type','rendered_prompt','rendered_prompt_sha256'}
        rendered=tokenizer.apply_chat_template(item['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
        assert rendered==item['rendered_prompt']
        assert tokenizer(rendered,add_special_tokens=False)['input_ids']==item['input_ids']
        assert item['input_ids_sha256']==job['input_ids_sha256']
        engine=HMMTEngine(ROOT,job['config_id'],model,tokenizer,native,progress)
        result=engine.generate(item['input_ids'],problem_id=item['problem_idx'])
        assert result['config_id']==job['config_id'] and result['actual_decode_forwards']==result['generated_token_count']-1
        assert result['input_ids_sha256']==job['input_ids_sha256'] and result['head_mask_sha256']==C.sha(ROOT/'calibration/head_mask.json')
        assert result['deterministic_algorithms'] is False and not result['diagnostic_only']
        assert C.model_pins(ROOT)==before and C.sha(native.__file__)==R.NATIVE_SHA256
        C.source(ROOT,a.source_sha);C.atomic(target/'POSTFLIGHT_REGULAR_MAPPED_SHA.json',dict(files=C.regular_maps()))
        value=dict(status='PASS',task_id='QWEN_GDN_HMMT26_FP_LQ8_HLQ8_LQ16_V1',unit=a.unit,job=job,
            identity=ident,runtime=runtime,gpu=a.gpu,source_manifest_sha256=a.source_sha,permit_sha256=a.permit_sha,
            model_load_wall_s=load,result=result,input_problem_sha256=item['problem_sha256'],prompt_text_sha256=item['rendered_prompt_sha256'])
        value['payload_sha256']=C.objsha(value);C.atomic(target/'RESULT.json',value)
        progress('UNIT_RESULT_SAVED',sha256=C.sha(target/'RESULT.json'));del engine,model;gc.collect();return 0
    except BaseException as exc:
        C.atomic(target/'FIRST_FAILURE.json',dict(status='FAIL_CLOSED_STOP',error=repr(exc),traceback=traceback.format_exc(),identity=ident,source_sha256=a.source_sha))
        traceback.print_exc();return 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--unit',required=True);p.add_argument('--gpu',required=True,type=int);p.add_argument('--source-sha',required=True);p.add_argument('--permit-sha',required=True)
    raise SystemExit(run(p.parse_args()))
