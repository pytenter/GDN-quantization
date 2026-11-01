"""Fresh-process A/B controller. Any scientific/runtime failure stops, never retries."""
from common import *
from recovery_control import active_stop,scientific_stop
import argparse

def freeze():
    assert not active_stop()
    attempt=load(ROOT/'provenance/PREMEASUREMENT_ENGINEERING_RECOVERY_ENVELOPE_V1.json')['CURRENT_SUCCESSFUL_PREPARATION_ATTEMPT']
    complete=load(ROOT/f'provenance/preparation_attempts/attempt{attempt}/CPU_PREPARATION_COMPLETE.json')
    assert all(v=='PASS' for v in complete['gates'].values())
    assert load(ROOT/'provenance/ENGINEERING_RECOVERY_CLOSURE.json')['gate']=='PASS'
    assert not (ROOT/'runtime/phase_a_source_freeze.json').exists()
    paths=[p for p in (ROOT/'scripts').glob('*.py')]
    paths+=[p for p in (ROOT/'protocol').glob('*.json')]+list((ROOT/'configs').glob('*.json'))
    paths+=[ROOT/'runtime/v3_runtime_identity.json',Path(load(ROOT/'runtime/v3_runtime_identity.json')['current_observation_path']),
        ROOT/'provenance/PREMEASUREMENT_ENGINEERING_RECOVERY_ENVELOPE_V1.json',
        ROOT/'provenance/ENGINEERING_RECOVERY_CLOSURE.json',
        ROOT/'provenance/v3_scientific_objects_before_engineering_recovery.json',
        ROOT/'inputs/historical_trainer.py',TRACES/'wikitext2raw-train-000.pt',
        PARENT/'scripts/phase2_final_hidden_512_smoke.py']
    paths+=[p for p in LEGACY.rglob('*.py') if '__pycache__' not in p.parts]
    paths+=[p for p in MODEL.glob('*') if p.is_file() and (p.suffix=='.safetensors' or p.name in ['config.json',
        'model.safetensors.index.json','configuration_bailing_moe_v3.py','modeling_bailing_moe_v3.py','tokenizer.json','tokenizer_config.json'])]
    paths=sorted(set(paths))
    rows=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in paths]
    emit(ROOT/'runtime/phase_a_source_freeze.json',dict(gate='PASS',files=rows,
        numerical_comparison='EXACT within same GPU only',no_tolerance=True,model_or_GPU_measurement_started=False,
        all_CPU_preparation_gates=complete['gates'],first_STOP_sha256=sha(ROOT/'provenance/STOP_PREPARATION.json')))
    print('PHASE_A_SOURCE_FREEZE=PASS',flush=True)

def run_worker(gpu,label,condition='L7R'):
    assert not active_stop()
    apps=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'])
    assert not apps,'GPU_NOT_RECLAIMED_BEFORE_FRESH_PROCESS'
    h=hardware();assert h['gpus'][gpu]['used_mib']<1024
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    emit(ROOT/f'runtime/gpu{gpu}_{label}_{condition}_launch.json',dict(physical_gpu=gpu,gpu_uuid=UUIDS[gpu],label=label,
        condition=condition,source_freeze_sha256=sha(ROOT/'runtime/phase_a_source_freeze.json'),
        SCIENTIFIC_GPU_MEASUREMENT_STARTED='YES',time_unix=time.time(),Fresh_B_forwards=0,AIME_generation=0,formal_updates=0))
    log=ROOT/f'logs/gpu{gpu}_{label}_{condition}.txt';log.parent.mkdir(exist_ok=True)
    with log.open('x') as f:
        p=subprocess.run([str(PYTHON),str(ROOT/'scripts/contemporary_smoke.py'),'--gpu',str(gpu),'--label',label,'--condition',condition],
                         env=env,stdout=f,stderr=subprocess.STDOUT,cwd=REPO)
    assert p.returncode==0,('SMOKE_PROCESS_FAILED',gpu,label,condition,p.returncode,str(log))
    output=ROOT/f'runtime/gpu{gpu}_{label}_{condition}_smoke.json';assert load(output)['gate']=='PASS'
    assert not command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'])
    emit(ROOT/f'runtime/gpu{gpu}_{label}_{condition}_exit.json',dict(gate='PASS',returncode=0,workers_exited=True,
        GPU_reclaim=True,output_sha256=sha(output),time_unix=time.time()))

def compare(gpu):
    a=load(ROOT/f'runtime/gpu{gpu}_A_L7R_smoke.json');b=load(ROOT/f'runtime/gpu{gpu}_B_L7R_smoke.json')
    assert a['physical_gpu']==b['physical_gpu']==gpu and a['gpu_uuid']==b['gpu_uuid']==UUIDS[gpu]
    assert a['pid']!=b['pid']
    # JSON round-trip canonicalizes integer dictionary keys before this exact comparison.
    same=a['numeric']==b['numeric'] and a['numeric_sha256']==b['numeric_sha256']
    differences=[k for k in a['numeric'] if a['numeric'][k]!=b['numeric'].get(k)]
    key=f'GPU{gpu}_CONTEMPORARY_DETERMINISM_GATE'
    receipt=dict(gate='PASS' if same else 'FAIL',**{key:'PASS' if same else 'FAIL'},physical_gpu=gpu,
        independent_fresh_processes=True,process_A_pid=a['pid'],process_B_pid=b['pid'],
        process_A_numeric_sha256=a['numeric_sha256'],process_B_numeric_sha256=b['numeric_sha256'],
        comparison='EXACT',tolerance=None,cross_GPU_bitwise_required=False,different_numeric_fields=differences)
    emit(ROOT/f'runtime/gpu{gpu}_contemporary_determinism.json',receipt)
    assert same,('SAME_DEVICE_A_B_DETERMINISM_FAIL',gpu,differences)
    print(key+'=PASS',flush=True)

def matched():
    a=load(ROOT/'runtime/gpu0_A_L7R_smoke.json')['numeric']
    b=load(ROOT/'runtime/gpu0_scientific_L8_smoke.json')['numeric']
    assert a['model_base']==b['model_base'] and a['input_token_ids']==b['input_token_ids']
    assert a['trace_sha256']==b['trace_sha256'] and a['flags']==b['flags'] and a['H128']==b['H128']
    for l in LAYERS:
        aa=a['parameter_before'][str(l)];bb=b['parameter_before'][str(l)]
        assert aa['shape']==bb['shape']==[8128] and aa['sha256']==bb['sha256']
        assert a['initial_rotations'][str(l)]==b['initial_rotations'][str(l)]
        if l in REGIONS['early']:
            assert aa['requires_grad'] and aa['in_optimizer_list'] and not bb['requires_grad'] and not bb['in_optimizer_list']
        else:assert aa==bb
    result=dict(gate='PASS',V3_UPDATE0_MATCHED_IDENTITY_GATE='PASS',V3_L7R_L8_SINGLE_FACTOR_GATE='PASS',
        L7R_SCIENTIFIC_PATH_GATE='PASS',L8_SCIENTIFIC_PATH_GATE='PASS',L8_EARLY_FIXED_H_GATE='PASS',
        L8_MIDDLE_LATE_TRAINABLE_GATE='PASS',BASE_WEIGHT_FREEZE_GATE='PASS',
        same_physical_gpu=0,base_model_exact=True,tokenizer_config_exact=True,data_exact=True,RNG_config_exact=True,
        middle_late_init_shape_bytes_mapping_exact=True,only_difference='early learnable Cayley versus exact fixed H',
        L7R_evidence_sha256=sha(ROOT/'runtime/gpu0_A_L7R_smoke.json'),L8_evidence_sha256=sha(ROOT/'runtime/gpu0_scientific_L8_smoke.json'),
        formal_optimizer_updates=0,disposable_optimizer_updates=5,Fresh_B_treatment_forwards=0,AIME_generation=0)
    emit(ROOT/'provenance/v3_update0_identity.json',result)
    emit(ROOT/'runtime/PHASE_A_COMPLETE.json',dict(gate='PASS',DUAL_4090_CONTEMPORARY_READINESS_GATE='PASS',
        GPU0_CONTEMPORARY_DETERMINISM_GATE='PASS',GPU1_CONTEMPORARY_DETERMINISM_GATE='PASS',matched_gates=result,
        PREMEASUREMENT_COMMIT_GATE='NOT_YET_CREATED',formal_optimizer_updates=0,Fresh_B_treatment_forwards=0,AIME_generation=0))
    print('DUAL_4090_CONTEMPORARY_READINESS_GATE=PASS; PHASE_A_COMPLETE=PASS',flush=True)

def main():
    for gpu in [0,1]:
        run_worker(gpu,'A');run_worker(gpu,'B');compare(gpu)
    run_worker(0,'scientific','L8');matched()

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['freeze','run']);x=a.parse_args()
    try:freeze() if x.mode=='freeze' else main()
    except BaseException as e:scientific_stop('CONTEMPORARY_DETERMINISM_CLOSURE',e);raise
