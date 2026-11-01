"""One fresh authorized trajectory; no retry, resume prefix or answer access."""
import argparse,gc,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import execution_common as C
def run(a):
    target=ROOT/'formal'/a.unit;target.mkdir(parents=True,exist_ok=True)
    assert not (target/'WORKER_IDENTITY.json').exists() and not (target/'RESULT.json').exists(),'No worker overwrite/retry'
    C.source(ROOT,a.source_sha)
    assert C.sha(ROOT/'GPU_PERMIT.json')==a.permit_sha
    permit=json.loads((ROOT/'GPU_PERMIT.json').read_text());assert permit['formal_target']==60 and permit['reuse']==0
    job=next(x for x in json.loads((ROOT/'FORMAL_JOBS.json').read_text())['jobs'] if x['id']==a.unit)
    assert os.environ['CUDA_VISIBLE_DEVICES']==C.UUIDS[job['gpu']]
    before=C.model_pins(ROOT)
    ident=C.identity(os.getpid());C.atomic(target/'WORKER_IDENTITY.json',dict(identity=ident,job=job,CVD=os.environ['CUDA_VISIBLE_DEVICES'],source_sha256=a.source_sha,permit_sha256=a.permit_sha))
    def progress(phase,**values):
        print(json.dumps(dict(phase=phase,monotonic=time.monotonic(),**values)),flush=True)
    try:
        from unroll_runtime import UnrollEngine
        sys.path.insert(0,str(ROOT/'scripts'))
        from model_engineering import runtime_identity
        progress('SOURCE_AND_MODEL_STAT_ADMISSION')
        load_start=time.monotonic()
        engine=UnrollEngine.from_loader(ROOT,job['period'],progress=progress)
        model_load_wall_s=time.monotonic()-load_start
        runtime=runtime_identity();assert runtime['driver_uuid']==C.UUIDS[job['gpu']]
        C.atomic(target/'LOADED_REGULAR_MAPPED_SHA.json',dict(files=C.regular_maps(),runtime=runtime))
        data=json.loads((ROOT/'AIME26_DATASET_FREEZE.json').read_text())
        item=next(x for x in data['records'] if x['problem_id']==job['problem_id'])
        _,rendered=engine.f.render_input(engine.tokenizer,item['problem'])
        assert rendered[0].tolist()==item['input_ids']
        assert item['input_ids_sha256']==job['input_ids_sha256']
        result=engine.generate(item['input_ids'],problem_id=item['problem_id'],max_new_tokens=81920)
        assert result['status']=='COMPLETE' and result['seed']==1 and result['actual_model_forward_calls']==item['input_tokens']+result['generated_token_count']-1
        assert result['head_mask_sha256']==C.sha(ROOT/'calibration/head_mask.json')
        assert result['input_ids_sha256']==job['input_ids_sha256']
        assert result['configuration']=='PACE48_LQUNROLL_DP'+str(job['period'])+'_V1'
        assert result['deterministic_algorithms'] is False
        assert result['quantizer_audit']['actual_layer_coverage']==24 and len(result['quantizer_audit']['storage_by_layer'])==24
        for row in result['quantizer_audit']['storage_by_layer'].values():
            assert row['decode_period']==job['period'] and row['validation_dense_materializations']==0
            assert not row['dense_state_persistent'] and row['decode_dense_materializations']==row['decode_tokens']//job['period']
        assert C.model_pins(ROOT)==before,'Model stat identity changed during trajectory'
        import pace_runtime as R
        assert C.sha(engine.native.__file__)==R.NATIVE_SHA256,'Actual native Python source changed'
        C.source(ROOT,a.source_sha)
        C.atomic(target/'POSTFLIGHT_REGULAR_MAPPED_SHA.json',dict(files=C.regular_maps()))
        value=dict(status='PASS',task_id='PACE48_LEAP_UNROLL_DP8_DP16_V1',unit=a.unit,job=job,
            identity=ident,runtime=runtime,source_manifest_sha256=a.source_sha,permit_sha256=a.permit_sha,
            model_load_wall_s=model_load_wall_s,
            result=result,input_problem_sha256=item['problem_sha256'],prompt_text_sha256=item['prompt_text_sha256'])
        value['payload_sha256']=C.objsha(value)
        C.atomic(target/'RESULT.json',value)
        progress('UNIT_RESULT_SAVED',sha256=C.sha(target/'RESULT.json'))
        del engine;gc.collect();return 0
    except BaseException as e:
        C.atomic(target/'FIRST_FAILURE.json',dict(status='FAIL_CLOSED_STOP',error=repr(e),traceback=traceback.format_exc(),identity=ident,source_sha256=a.source_sha))
        traceback.print_exc();return 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--unit',required=True);p.add_argument('--source-sha',required=True);p.add_argument('--permit-sha',required=True);raise SystemExit(run(p.parse_args()))
