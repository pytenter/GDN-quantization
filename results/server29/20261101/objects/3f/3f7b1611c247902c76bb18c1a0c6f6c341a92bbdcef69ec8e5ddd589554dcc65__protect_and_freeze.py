"""Read-only historical audit; write only new append-only correction receipts."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from capsule_v2 import sha, write

C=Path(__file__).resolve().parent; A=C.parent; E=A.parents[1]; R=E.parents[1]


def main():
    mode=__import__('sys').argv[1]
    os.environ['GIT_OPTIONAL_LOCKS']='0'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()=='c9c8e3199e0d017b9712809b77d8f16cbd07f165'
    assert not subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=no'],cwd=R,text=True).strip()
    if mode=='before':
        files=[dict(path=str(p.relative_to(A)),size=p.stat().st_size,sha256=sha(p))
            for p in sorted(A.rglob('*')) if p.is_file() and not p.is_relative_to(C)]
        assert json.loads((A/'FINAL_REPORT.json').read_text())['SUPERVISOR_CLEANUP_LEDGER_REMAINING']==1
        assert sha(A/'FINAL_REPORT.md')=='ea112dc1382e05af369bd2626b5acc27e112c92b60322112021ac48d9dbf6fa5'
        assert 'absent' in (A/'FINAL_REPORT.md').read_text()
        write(C/'prior_v2_protection.json',dict(files=files,old_v2_cleanup_ledger_remaining=1,
            old_fail_stop_remains_valid=True,formal_measurements=0))
        print('PRIOR_V2_PROTECTED',len(files),flush=True)
    elif mode=='freeze':
        tests=json.loads((C/'engineering_tests/results.json').read_text())
        assert all(tests['V2_TEST_'+str(i)].startswith('PASS') for i in range(1,11))
        assert sha(C/'test_v2.py')==sha(A/'test_v2.py')
        semantics=json.loads((C/'semantic_tests/results.json').read_text());assert semantics['gate']=='PASS'
        write(C/'candidate_architecture_freeze.json',dict(source_files=[dict(name=p.name,sha256=sha(p))
            for p in sorted(C.iterdir()) if p.is_file() and p.suffix in ('.py','.md')],
            original_10_tests=tests,semantic_tests=semantics,formal_measurements=0,
            amendment_sha256=sha(C/'AMENDMENT_CLASSIFIER_CORRECTION_V1.md'),
            qualification_started=False,GPU6_FORMAL_AUTHORIZED='NO',new_premeasurement_commit='NOT_CREATED'))
        print('CANDIDATE_ARCHITECTURE_FROZEN_NOT_QUALIFIED',flush=True)
    elif mode=='after':
        before=json.loads((C/'prior_v2_protection.json').read_text())
        for row in before['files']:
            p=A/row['path'];assert p.stat().st_size==row['size'] and sha(p)==row['sha256'],str(p)
        old=json.loads((A/'protected_identity_before.json').read_text())
        for i,row in enumerate(old['files']):
            p=R/row['path'];assert sha(p)==row['sha256'],str(p)
            if i%200==0:print('PRESERVED_OLD_EXPERIMENT',i,flush=True)
        s=importlib.util.spec_from_file_location('preserved_readonly_preparation',A/'prepare_source_v2.py')
        m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
        legacy=m.legacy_check();assert legacy==old['legacy']
        identities=json.loads((E/'source_identity.json').read_text())
        for kind in ('inputs','protected_historical_files'):
            for i,row in enumerate(identities[kind]):
                assert sha(row['path'])==row['sha256'],row['path']
                if i%100==0:print('PRESERVED_INPUTS',kind,i,flush=True)
        write(C/'post_correction_preservation.json',dict(gate='PASS',legacy=legacy,
            prior_v2_files=len(before['files']),old_experiment_files=len(old['files']),
            inputs=len(identities['inputs']),protected_parent_files=len(identities['protected_historical_files']),
            two_previous_A_attempts_byte_identical=True,old_v2_fail_stop_valid=True,
            original_cleanup_ledger_remaining=1,scientific_protocol_changed=False))
        print('ALL_PROTECTED_BYTES_PASS',flush=True)
    else:raise RuntimeError('INVALID_MODE')


if __name__=='__main__':main()
