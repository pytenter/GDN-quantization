"""Preserve an implementation false-positive attempt; freeze next A prospectively."""
import json
import os
from pathlib import Path
from capsule_v2 import sha,write

A=Path(__file__).resolve().parent


def main():
    old=json.loads((A/'V2_A_attempt1/verify/runtime_escape_report.json').read_text())
    assert old['gate']=='FAIL'
    candidate='/data/zypan/.nvm/versions/node/v24.18.0/bin/git'
    assert not os.path.lexists(candidate),'CANDIDATE_NOW_EXISTS_CANNOT_CLASSIFY_OLD_FAILURE'
    assert candidate in old['error']
    tests=json.loads((A/'engineering_tests/results.json').read_text())
    assert all(tests['V2_TEST_'+str(i)].startswith('PASS') for i in range(1,11))
    old_files=[dict(path=str(p.relative_to(A)),sha256=sha(p)) for p in sorted((A/'V2_A_attempt1').rglob('*')) if p.is_file()]
    write(A/'architecture_freeze_revision1.json',dict(AMENDMENT_SHA256=sha(A/'AMENDMENT_V2.md'),
        FORMAL_MEASUREMENTS_BEFORE_AMENDMENT=0,authority='V2 generic no-new-external-bytes implementation correction',
        source_files=[dict(name=p.name,sha256=sha(p)) for p in sorted(A.glob('*.py'))],
        architecture_tests=tests,architecture_frozen_before_real_dependency_discovery=True,
        prior_attempt_preserved=old_files,prior_freeze_sha256=sha(A/'architecture_freeze.json'),
        diagnosis='Nonexistent execvp PATH probe; no external artifact entered. Old FAIL preserved without PASS reinterpretation.',
        no_external_artifact_auto_authorized=True))
    print('NEW_PROSPECTIVE_ARCHITECTURE_FREEZE_PRIOR_FAILURE_PRESERVED',flush=True)


if __name__=='__main__':main()
