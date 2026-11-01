"""Administrative output routing only; every mathematical module is unchanged.

Set the original utility module's output namespace before importing the same
qualified implementations. All model/input/source paths stay unchanged.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import capture_original_drive as C
BUG=C.ROOT/'amendments/CAPTURE_KEY_SHADOWING_BUGFIX_V1'
C.AMEND=BUG
import isolated_replay_engineering as E
import persistent_anchor_replay as P
import qualify_long_and_streaming as Q
import scientific_lifecycle as S
import capture_extended_drive as M

def status(action):
    deterministic=E.GATES/'determinism.json';streaming=E.GATES/'streaming.json'
    auth=BUG/'scientific_execution_authorization.json'
    results=list((BUG/'scientific').rglob('replay_result.json'))
    return dict(BUG_CLASSIFICATION='IMPLEMENTATION_VARIABLE_SHADOWING',
        BUGFIX_SCOPE='VARIABLE_RENAME_OR_SCOPE_CORRECTION_ONLY',FAILED_ATTEMPT_PRESERVED='YES',
        SCIENTIFIC_MEASUREMENT_BEFORE_FIX=0,G0_G10_PREVIOUS_PASS_PRESERVED='YES',G11_256_TOKEN_STATUS='PASS',
        G11_2048_TOKEN_PREVIOUS_STATUS='FAIL_IMPLEMENTATION_BUG',
        G11_2048_TOKEN_RETEST_STATUS=C.read(deterministic)['gate'] if deterministic.exists() else 'NOT_COMPLETED',
        G12_STATUS=C.read(streaming)['gate'] if streaming.exists() else 'NOT_RUN',
        PREMEASUREMENT_COMMIT_SHA=C.read(auth)['PREMEASUREMENT_COMMIT_SHA'] if auth.exists() else None,
        DISCOVERY_COMPLETE=str(len(results))+'/16',FINAL_ACTION=action,
        bugfix_commit_sha=C.read(BUG/'bugfix_commit_receipt.json')['bugfix_commit_sha'])

def final_bugfix_report(action):
    v=status(action)
    if action=='FAIL_AND_STOP':
        error_files=list(BUG.glob('*.error.json'))
        stages=[C.read(p).get('stage','') for p in error_files]
        if (BUG/'final_verdict.json').exists():stages.append(C.read(BUG/'final_verdict.json').get('failed_phase',''))
        if any(s.startswith('G11') for s in stages):v['G11_2048_TOKEN_RETEST_STATUS']='FAIL'
        if any(s.startswith('G12') for s in stages):v['G12_STATUS']='FAIL'
    C.save(BUG/'BUGFIX_FINAL_STATUS.json',v)
    report='# Surgical key-binding bugfix outcome\n\n'+json.dumps(v,indent=2)+'\n\n'
    report+='See static_source_audit.json, source_diff.patch, BUGFIX_REPORT.md, failed_attempt_snapshot.json and the scientific FINAL_REPORT.md. No additional fix or retry was authorized or executed.\n'
    S.write_text(BUG/'BUGFIX_FINAL_REPORT.md',report)
    files=[p for p in sorted(BUG.rglob('*')) if p.is_file() and p.name!='BUGFIX_FINAL_SHA256SUMS' and '__pycache__' not in p.parts]
    S.write_text(BUG/'BUGFIX_FINAL_SHA256SUMS',''.join(C.sha(p)+'  '+str(p.relative_to(BUG))+'\n' for p in files))
    for line in (BUG/'BUGFIX_FINAL_SHA256SUMS').read_text().splitlines():
        expected,rel=line.split(None,1);assert C.sha(BUG/rel)==expected

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['capture2048','replay2048','compare2048','streaming',
        'preregister','capture_discovery','replay_discovery','release','finalize','stop'],required=True)
    p.add_argument('--run');p.add_argument('--document',type=int);p.add_argument('--condition');p.add_argument('--failed-stage');a=p.parse_args()
    receipt=C.read(BUG/'bugfix_commit_receipt.json')
    for rel,expected in receipt['code_sha256'].items():assert C.sha(BUG/rel)==expected,'Frozen bugfix code drift'
    try:
        if a.stage=='capture2048':M.execute('engineering2048',None)
        elif a.stage=='replay2048':P.execute('engineering',2048,a.run,None,None)
        elif a.stage=='compare2048':Q.determinism(2048)
        elif a.stage=='streaming':Q.streaming()
        elif a.stage=='preregister':S.preregister()
        elif a.stage=='capture_discovery':M.execute('discovery',a.document)
        elif a.stage=='replay_discovery':P.execute('discovery',32768,None,a.document,a.condition)
        elif a.stage=='release':S.seal_release(a.document)
        elif a.stage=='finalize':
            S.finalize();final_bugfix_report('DISCOVERY_COMPLETE_REPORT_ONLY')
        else:
            S.stop(a.failed_stage);final_bugfix_report('FAIL_AND_STOP')
    except Exception as e:
        path=BUG/(a.stage+'.error.json')
        if not path.exists():C.save(path,dict(status='FAIL_AND_STOP',stage=a.failed_stage or a.stage,
            exception=type(e).__name__,message=str(e),traceback=traceback.format_exc(),retry_authorized=False))
        raise

if __name__=='__main__':main()
