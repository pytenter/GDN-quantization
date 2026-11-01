"""Bounded stdlib-only tests of the preparation entry, never a GPU launcher."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run():
    entry = ROOT / 'source/a800_entry.py'
    config = ROOT / 'PAIRED_TRAINING_PREREGISTRATION.json'
    def invoke(arguments):
        p = subprocess.run([sys.executable, '-S', str(entry)] + arguments,
                           capture_output=True, text=True, cwd=tempfile.gettempdir(), timeout=30)
        if p.stderr:
            raise AssertionError(p.stderr)
        v = json.loads(p.stdout)
        if v['execution_performed'] or v['torch_imported']:
            raise AssertionError('Dry-run/denial imported a training runtime')
        return p.returncode, v
    code, result = invoke([])
    assert code == 0 and result['status'] == 'DRY_RUN_ONLY'
    assert result['FULL_MODEL_RUNNER_IMPLEMENTATION'] == 'BLOCKED_UNCLOSED_CONTROL_DEPENDENCIES'
    refused = []
    for phase in ('admission', 'teacher', 'train', 'validate', 'export'):
        for arm in ('both', 'A', 'D'):
            code, result = invoke(['--phase', phase, '--arm', arm, '--execute'])
            assert code == 4 and result['status'] == 'REFUSED'
            refused.append(phase + ':' + arm)
    original = json.loads(config.read_text())
    bad_cases = []
    with tempfile.TemporaryDirectory(prefix='qwen_readiness_entry_cpu_') as tmp:
        p = Path(tmp) / 'config.json'
        for name, change in (
            ('arm_specific_lr', lambda c: c['arms']['D'].update(lr=.003)),
            ('updates_126', lambda c: c['common'].update(optimizer_updates=126)),
            ('changed_loss', lambda c: c['common']['loss'].update(lambda_state=.2)),
            ('selected94', lambda c: c['export'].update(primary_checkpoint='SELECTED_UPDATE_94'))):
            altered = copy.deepcopy(original); change(altered); p.write_text(json.dumps(altered))
            code, result = invoke(['--config', str(p)])
            assert code == 2 and result['status'] == 'FAIL_CLOSED'
            bad_cases.append(name)
        # Merely editing the JSON authorization flags cannot create a backend.
        altered = copy.deepcopy(original)
        for key in ('A800_RENTAL_AUTHORIZED', 'A800_GPU_EXECUTION', 'A800_FULL_TRAINING_AUTHORIZED', 'FULL_ROTATION_TRAINING'):
            altered['authorization'][key] = True
        p.write_text(json.dumps(altered))
        code, result = invoke(['--config', str(p), '--phase', 'train', '--execute'])
        assert code == 4 and result['status'] == 'REFUSED'
    return dict(status='PASS_OFFLINE_ENTRY_CONTRACT_ONLY',
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        default_dry_run=True, cwd_independent=True, denied_execution_phases_and_arms=refused,
        contract_mutations_rejected=bad_cases, authorization_edit_cannot_bypass_missing_runner=True,
        Torch_or_CUDA_imported=False, model_loaded=False, GPU_used=False,
        full_model_runner='BLOCKED_UNCLOSED_CONTROL_DEPENDENCIES', full_model_A800_admission='NOT_TESTED')


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, indent=2))
