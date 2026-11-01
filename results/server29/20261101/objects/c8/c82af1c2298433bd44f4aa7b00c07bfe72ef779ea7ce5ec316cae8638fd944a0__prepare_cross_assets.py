"""CPU-only, task-specific reuse audit; no historical weight re-hashing."""
import ast
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
import functional_worker as w

D=w.OUT
OLD=Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_FUNCTIONAL_VALIDATION_V1')
WT=Path(__file__).resolve().parent


def same_function_ast(a,b,name):
    x=next(n for n in ast.walk(ast.parse(a)) if getattr(n,'name',None)==name)
    y=next(n for n in ast.walk(ast.parse(b)) if getattr(n,'name',None)==name)
    assert ast.dump(x,include_attributes=False)==ast.dump(y,include_attributes=False),name


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    assert not (D/'STOP.json').exists()
    assert w.read(OLD/'FORMAL_COMPLETE.json')['status']=='PASS'
    assert w.read(OLD/'COMPLETED_ARTIFACT_AUDIT.json')['status']=='PASS'
    assert w.read(OLD/'delivery/DOCUMENT_VERDICT.json')['overall']=='MIXED'
    base=w.read(OLD/'FROZEN_SPEC.json');keys=w.read(D/'CROSS_SCIENTIFIC_KEYS_FROZEN.json')
    assert [(u['document'],u['rotation']) for u in keys['units']]==[
        ('pg19-test-860','H_to_L7_a50'),('pg19-test-5956','H_to_L7_a25')]
    assert keys['original_spec_sha256']==w.sha(OLD/'FROZEN_SPEC.json')
    assert keys['primary']==base['primary'] and keys['resolution']==base['resolution']
    previous_freeze=w.read(OLD/'FORMAL_CODE_FREEZE.json')
    for name,h in previous_freeze['files'].items():assert w.sha(OLD/name)==h,name
    model_reuse=[];fresh=[]
    for name,row in base['source_assets'].items():
        s=Path(name).stat()
        assert s.st_size==row['bytes'] and s.st_mtime_ns==row['mtime_ns'],name
        if name.endswith('.safetensors'):
            model_reuse.append(dict(path=name,verified_SHA_reused=row['sha256'],unchanged_size_mtime=True))
        else:
            assert w.sha(name)==row['sha256'],name;fresh.append(name)
    for name,h in base['historical_STOP_sha256'].items():assert w.sha(name)==h,name
    ast_checks=[]
    for filename,names in [('functional_worker.py',['transfer','object_fingerprint','audit_state','restore_audit','HiddenObserver','checkpoint']),
        ('formal_worker.py',['original_state_diagnostic','observe_qdq','collect_metrics'])]:
        for name in names:
            same_function_ast((OLD/filename).read_text(),(WT/filename).read_text(),name)
            ast_checks.append(filename+':'+name)
    assert subprocess.check_output(['git','-C',str(WT),'status','--porcelain'],text=True)==''
    commit=subprocess.check_output(['git','-C',str(WT),'rev-parse','HEAD'],text=True).strip()
    spec=dict(base)
    spec.update(experiment=keys['experiment'],status='FROZEN_POSTHOC_EXPLORATORY_CROSS_PANEL',
        new_units=keys['units'],maximum_new_student_32K_units=2,
        reference_experiment=str(OLD),reference_primary_spec_sha256=w.sha(OLD/'FROZEN_SPEC.json'),
        scientific_keys_sha256=w.sha(D/'CROSS_SCIENTIFIC_KEYS_FROZEN.json'),
        original_formal_verdict='MIXED_PERMANENTLY_PRESERVED',
        source_worktree=str(WT),source_commit=commit,no_GPUh_hard_cap=True,
        no_fixed_VRAM_percent_gate=True,no_progress_timeout_seconds=900,
        model_weight_SHA_reuse_basis='Previously verified SHA bound to unchanged size/mtime; no complete historical audit repeated',
        scoring_contract_sha256=hashlib.sha256(json.dumps(base['primary'],sort_keys=True).encode()).hexdigest())
    w.publish(D/'FROZEN_SPEC.json',spec);w.publish(D/'CROSS_ROTATION_FROZEN_SPEC.json',spec)
    files={name:w.sha(WT/name) for name in ('functional_worker.py','formal_worker.py','cross_pipeline.py','prepare_cross_assets.py')}
    for name in files:
        (D/name).symlink_to(WT/name)
    w.publish(D/'CODE_FREEZE.json',{'functional_worker.py':files['functional_worker.py']})
    w.publish(D/'FORMAL_CODE_FREEZE.json',dict(files={**files,'FROZEN_SPEC.json':w.sha(D/'FROZEN_SPEC.json'),
        'SECONDARY_SPEC.json':w.sha(D/'SECONDARY_SPEC.json')},dependencies=previous_freeze['dependencies'],
        engine_sha256=base['engine_sha256'],source_commit=commit,frozen_before_new_functional_outcomes=True))
    evidence=D/'existing_evidence';evidence.mkdir()
    records=[];availability=[]
    for entry in base['documents']:
        doc=entry['document']
        for rotation in (entry['P1_predicted_winner'],entry['L7_predicted_winner']):
            f=OLD/'formal'/f'{doc}__{rotation}__formal.json';v=w.read(f)
            points=v.get('primary_samples')
            available=points is not None and len(points)==384
            if available:
                assert [x['position'] for x in points]==base['primary']['positions']
                assert all(x['vocabulary_dimension']==157184 for x in points)
                assert sum(x['KL'] for x in points)/384==v['primary_mean_KL']
            shutil.copyfile(f,evidence/f.name)
            row=dict(document=doc,rotation=rotation,path=str(f),sha256=w.sha(f),
                status=v['status'],per_position_KL_available=available,source_commit_of_engine='9badc4b4744a10aba4b737f49daa52ba74775015',
                actual_successful_adapter_freeze_SHA=w.sha(OLD/'FORMAL_CODE_FREEZE.json'),
                reducer='original persistent_engine_v1.full_kl FP32, Python sum/384',
                scalar_count=len(points) if points is not None else None)
            records.append(row);availability.append(row)
    w.publish(D/'PER_POSITION_KL_AVAILABILITY.json',dict(status='AVAILABLE' if all(x['per_position_KL_available'] for x in availability)
        else 'PER_POSITION_KL_NOT_AVAILABLE',trajectories=availability,not_reconstructed_from_aggregates=True))
    import torch
    torch.set_num_threads(1)
    refs=[]
    for entry in base['documents']:
        doc=entry['document'];fp_path=OLD/'formal'/f'{doc}__FP__formal.json';fp=w.read(fp_path)
        assert fp['status']=='PASS' and fp['endpoint']==32768
        assert fp['sample_positions']==base['primary']['positions']
        assert fp['gpu_uuid']==base['execution']['gpu_uuid']
        assert fp['primary_spec_sha256']==w.sha(OLD/'FROZEN_SPEC.json')
        token=w.read(entry['token_file']['path']);digest=hashlib.sha256();positions=[]
        for r in fp['references']:
            assert Path(r['path']).is_file() and w.sha(r['path'])==r['sha256'],r['path']
            payload=torch.load(r['path'],map_location='cpu',weights_only=False)
            assert payload['position']==r['position'] and payload['token']==token[r['position']]
            if r['logits']:
                value=payload['logits'];assert tuple(value.shape)==(1,1,157184) and value.dtype==torch.float32
                assert bool(torch.isfinite(value).all())
                digest.update(r['position'].to_bytes(8,'little'));digest.update(w.engine.tensor_bytes(torch,value))
                positions.append(r['position'])
        assert positions==base['primary']['positions'] and digest.hexdigest()==fp['sampled_logits_content_sha256']
        refs.append(dict(document=doc,manifest_path=str(fp_path),manifest_sha256=w.sha(fp_path),
            real_sampled_logits_available=True,actual_sample_count=384,sampled_logits_content_SHA=digest.hexdigest(),
            GPU_UUID=fp['gpu_uuid'],runtime=fp['runtime'],all_reference_payload_SHA_PASS=True))
        print('FP_REFERENCE_REUSE_VERIFIED',doc,flush=True)
    assert not torch.cuda.is_initialized()
    cfg=w.read(w.OLD/'PHASE_B_CALIBRATION_CONFIG.json');guard=w.device(6,cfg)
    assert guard['exclusive'],guard['failure_conditions']
    resources=w.read(OLD/'delivery/RESOURCE_LEDGER.json')
    assert shutil.disk_usage(D).free>64*1024**3
    w.publish(D/'FP_REFERENCE_REUSE_GATE.json',dict(status='PASS',spec_sha256=w.sha(D/'FROZEN_SPEC.json'),
        old_and_new_full_vocabulary_reducer_identical=True,same_device_required=True,FP_reruns=0,references=refs,
        qualification_reuse_from_completed_pipeline=True,changed_items=['scientific-key whitelist','result and FP manifest routing'],
        numerical_engine_checkpoint_restore_device_mapping_KL_unchanged=True,exact_AST_checks=ast_checks,
        own_rotation_parent_and_resume_qualification_required=True))
    w.publish(D/'EXISTING_RESULTS_ASSET_AUDIT.json',dict(status='PASS',latest_COMPLETE_experiment=str(OLD),
        original_verdict='MIXED',successful_execution='SINGLE_CACHE_SERIAL_EXECUTION',
        prior_full_asset_audit_path=str(OLD/'COMPLETED_ARTIFACT_AUDIT.json'),prior_full_asset_audit_SHA=w.sha(OLD/'COMPLETED_ARTIFACT_AUDIT.json'),
        old_student_records=records,real_FP_references=refs,unchanged_weight_SHA_reuse=model_reuse,
        fresh_task_specific_nonweight_SHA_checks=fresh,source_worktree=str(WT),source_commit=commit,
        unchanged_numerical_helper_AST=ast_checks,historical_STOP_preserved=True,GPU_guard=guard,
        past_complete_resource_receipt=dict(peak_allocated=resources['peak_allocated_during_recorded_token_execution'],
            peak_reserved=resources['peak_reserved_during_recorded_token_execution']),CUDA_initialized=False,
        no_old_student_reruns=True,maximum_new_32K_students=2))
    print('CPU_REUSE_GATES_PASS_NO_NEW_GPU_MEASUREMENT',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as e:
        if not (D/'STOP.json').exists():w.publish(D/'STOP.json',dict(status='STOP',stage='cross_pairing_reuse',
            reason=repr(e),classification='CROSS_PANEL_PAIRING_INVALID_STOP',epoch=time.time(),first_failure=True))
        raise
