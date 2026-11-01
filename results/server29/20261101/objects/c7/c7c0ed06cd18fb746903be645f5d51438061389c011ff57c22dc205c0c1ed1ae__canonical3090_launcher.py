#!/usr/bin/env python3
"""Additive operational gate before the already-frozen endpoint coordinator."""
import json
from pathlib import Path
import subprocess
import time
import traceback
import canonical3090_offline as O
P=O.P
ROOT=O.ROOT


def evidence_copy():
    for name in ('provenance','qualification','teacher','readiness','training','logs','analysis'):
        O.copy(O.A800+'/'+name+'/',ROOT/'a800_stop_evidence'/name,('--exclude=targets_data/','--exclude=*.pt','--exclude=*.log'))


def failure_report(error):
    try:evidence_copy()
    except Exception as e:error['evidence_copy_error']=str(e)
    base=ROOT/'a800_stop_evidence';trust=base/'provenance/trusted_environment_freeze.json'
    status={}
    if trust.exists():status.update(P.read(trust))
    trusted=status.get('A800_TRAINING_ENVIRONMENT_TRUSTED')=='YES'
    required=('A800_HOST_QUALIFICATION','LM_HEAD_CANARY_3X','FULL_MODEL_GPU_PARAMETER_INTEGRITY','FULL_MODEL_FRESH_PROCESS_REPEATABILITY','FP_RUNTIME_STABILITY')
    s={k:status.get(k,'PASS' if k=='LM_HEAD_CANARY_3X' and trusted else 'FAIL') for k in required}
    s['A800_TRAINING_ENVIRONMENT_TRUSTED']='YES' if trusted else 'NO'
    gates={'CLEAN_TEACHER_TARGET_PARITY':'teacher/teacher_cache_parity.json','C5R_C6R_C7_UPDATE0_FORWARD_PARITY':'readiness/update0_forward_parity.json'}
    for arm in P.ARMS:
        gates[arm+'_GRADIENT_SMOKE']=f'readiness/{arm.lower()}_gradient_smoke.json'
        gates[arm+('_H64_DIRECT_READY' if arm=='C7' else '_H64_READY')]=f'readiness/{arm.lower()}_h64_readiness.json'
        summary=base/f'training/{arm}/summary.json';log=base/f'training/{arm}/train_log.jsonl'
        s[arm+'_FORMAL_TRAINING']='COMPLETE' if summary.exists() and P.read(summary).get('status')=='PASS' else ('PARTIAL' if log.exists() else 'NOT_STARTED')
        s[arm+'_SELECTED_UPDATE']=P.read(summary).get('selected_update','NA') if summary.exists() else 'NA'
        smoke=ROOT/'deployment/canonical_3090_smoke.json'
        s[arm+'_DEPLOYMENT']='PASS' if smoke.exists() and P.read(smoke).get('status')=='PASS' else ('FAIL' if (ROOT/'logs/endpoint_smoke.error.json').exists() else 'NOT_RUN')
        s[arm+'_AIME26']='NA'
    for key,rel in gates.items():
        p=base/rel;s[key]=P.read(p).get('status','FAIL') if p.exists() else 'NOT_RUN'
    verdict=('INCONCLUSIVE' if all(s[a+'_FORMAL_TRAINING']=='COMPLETE' for a in P.ARMS) else 'TRAINING_BLOCKED') if trusted else 'A800_ENVIRONMENT_REJECTED'
    s.update(PRIMARY_VERDICT=verdict,C7_VS_C6R_NET='NA',C7_VS_HADAMARD_NET='NA',C6R_VS_HISTORICAL_C6='NA',FULL_DENSITY_FULL60='CANCELLED_BY_USER')
    P.save(ROOT/'analysis/blocked_status.json',s)
    with (ROOT/'FINAL_REPORT.md').open('x') as f:
        f.write('# Clean-room task stopped at a hard gate\n\nNo protocol downgrade, scientific retry or tensor repair was performed. No new downstream stage is authorized after this stop. Existing endpoint files, if any, remain unscored unless the full output freeze passed. Historical C5/C6 endpoints remain valid archived observations.\n\n```text\n')
        for k,v in s.items():f.write(k+' = '+str(v)+'\n')
        f.write('```\n\nExact stop evidence:\n\n```json\n'+json.dumps(error,indent=2)+'\n```\n\nReview preserved A800 error records; do not restart automatically.\n')
    with (ROOT/'NEXT_EXPERIMENT_DECISION.md').open('x') as f:f.write('# Manual decision required\n\nInvestigate the preserved blocking evidence. No alternative protocol or follow-on experiment is authorized by this report.\n')
    with (ROOT/'SHA256SUMS').open('x') as f:
        for p in sorted(ROOT.rglob('*')):
            if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts and p.suffix not in ('.log','.exitcode'):f.write(P.sha(p)+'  '+str(p.relative_to(ROOT))+'\n')


def main():
    P.check_freeze()
    P.save(ROOT/'protocol/operational_launcher_identity.json',{'launcher_sha256':P.sha(Path(__file__)),'purpose':'CPU-only inventory completion gate; frozen endpoint source unmodified'})
    while True:
        code=O.remote('if test -f '+O.A800+'/logs/archive_inventory.exitcode; then cat '+O.A800+'/logs/archive_inventory.exitcode; else echo WAIT; fi')
        if code=='WAIT':time.sleep(60);continue
        if code!='0':raise RuntimeError('A800 formal/posttraining/inventory hard gate stopped; no 3090 stage')
        break
    with (ROOT/'logs/offline_endpoint_coordinator.log').open('x') as f:
        subprocess.run([O.PY,str(ROOT/'scripts/canonical3090_offline.py')],stdout=f,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':
    try:main()
    except BaseException as e:
        error={'status':'FAIL','error':str(e),'traceback':traceback.format_exc(),'no_GPU_process_killed':True,'no_semantic_retry':True}
        P.save(ROOT/'logs/offline_launcher.error.json',error)
        if not (ROOT/'FINAL_REPORT.md').exists():failure_report(error)
        raise
