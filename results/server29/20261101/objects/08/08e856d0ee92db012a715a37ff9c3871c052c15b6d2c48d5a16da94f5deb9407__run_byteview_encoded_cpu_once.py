"""Bounded real Torch CPU entry tests through stdin; no remote files or GPU."""
from pathlib import Path
import hashlib,importlib.util,json,os,subprocess,time
R=Path(__file__).resolve().parents[1]
P=R.parent/'QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1_REDUCTION_REPAIR_20261025'
D=P/'operations/NEXT_CPU_ONLY_DIAGNOSTIC_PATCH'
OUT=R/'operations/BYTEVIEW_REAL_TORCH_ENCODED_CPU_TWO_FRESH.json'
assert not OUT.exists(),'Preserve every actual attempt; do not overwrite receipts'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(R/'src/gpu_codec_fields.py')=='83c153a992220e27e71567679a8d7524082fb2f100395b4453c98176a4a41211'
assert sha(P/'src/gpu_codec_fields.py')=='7e0430b04c354988b0ac66c2334236c976b0987055b9acf1d7f210d1815ec891'
assert sha(D/'CPU_CHECKS_TWO_FRESH.json')=='54bb49ca0270e7358fd3f84d66dd793404c126a051a02af0aecc18ea768c7b18'
inherit=json.loads((D/'CPU_CHECKS_TWO_FRESH.json').read_text())
assert inherit['status']=='PASS_TWO_FRESH_REAL_TORCH_CPU_AND_ORIGINAL_NINE_FIXTURES'
assert len(inherit['runs'])==4 and all(x['returncode']==0 for x in inherit['runs'])
assert inherit['source_sha256_before']['operations/NEXT_CPU_ONLY_DIAGNOSTIC_PATCH/gpu_codec_fields_candidate.py']==sha(R/'src/gpu_codec_fields.py')
sources=['src/gpu_codec_fields.py','src/unified_codec.py','reference/math_reference.py','inherited/group32/src/group_codecs.py']
payload={p:(R/p).read_text() for p in sources}
pins={p:sha(R/p) for p in sources+['tests/byteview_encoded_cpu.py']}
protected={str(p):sha(p) for p in [P/'SOURCE_FREEZE.json',P/'GPU_PERMIT.json',P/'src/gpu_codec_fields.py',D/'CPU_CHECKS_TWO_FRESH.json',R/'operations/HUMAN_BYTEVIEW_REPAIR_AUTHORIZATION.json']}
spec=importlib.util.spec_from_file_location('existing_byteview_readonly_transport',R.parent/'QWEN_GDN_GROUP64_FOUR_CODEC_AIME26_SEED1_V1/operations/remote.py')
T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
code='CPU_PAYLOAD='+repr(payload)+'\n'+(R/'tests/byteview_encoded_cpu.py').read_text()
runs=[]
for fresh in (1,2):
    start=time.monotonic()
    p=subprocess.run(T.command('55','/data/ydai/miniconda3/envs/bitdecode/bin/python'),input=code,text=True,capture_output=True,
        timeout=60,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
    report=None
    try:report=json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError,IndexError):pass
    runs.append(dict(fresh=fresh,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,report=report,elapsed_local_monotonic=time.monotonic()-start))
    if p.returncode:break
assert all(sha(R/p)==v for p,v in pins.items()),'new source changed during tests'
assert all(sha(Path(p))==v for p,v in protected.items()),'protected prior source/receipts changed'
passed=len(runs)==2 and all(x['returncode']==0 and x['report'] and x['report']['status'].startswith('PASS_') for x in runs)
receipt=dict(status='PASS_TWO_FRESH_REAL_TORCH_COMPLETE_ENCODED_AND_PANEL_CPU_ENTRY' if passed else 'FAIL_BYTEVIEW_CPU_ENTRY',
    source_sha256=pins,protected_prior_receipt_authorization_sha256=protected,runs=runs,
    inherited_evidence=dict(path=str(D/'CPU_CHECKS_TWO_FRESH.json'),sha256=sha(D/'CPU_CHECKS_TWO_FRESH.json'),
        real_Torch_CPU_checks=12,original_nine_fixtures=9,fresh_processes_each_suite=2,
        inherited_not_fresh_for_this_revision=True,exact_approved_candidate_matches_current_production_source=True),
    old_true_STOP_preserved=True,old_SOURCE_PERMIT_and_source_unchanged=True,
    GPU_calls=0,driver_calls=0,kernel_calls=0,model_loads=0,compile_calls=0,remote_file_writes=0,
    GPU_qualification_credit=0,AIME_samples=0,
    scope='Production Encoded/cpu/summary/exact/verify_panel plumbing with real Torch CPU tensors and explicit encode CPU fixtures; no GPU numerical proof')
with OUT.open('x') as f:json.dump(receipt,f,sort_keys=True,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(status=receipt['status'],receipt=str(OUT),sha256=sha(OUT),runs=[dict(fresh=x['fresh'],returncode=x['returncode'],report=x['report']) for x in runs]),ensure_ascii=False))
raise SystemExit(not passed)
