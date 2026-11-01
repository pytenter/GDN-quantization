#!/usr/bin/env python3
"""The single A800 preparation entry. Default is stdlib-only, read-only dry-run.

No full-model backend is present. This is an honest planning/denial entry,
not a launcher that silently substitutes a local operator for a 9B rollout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TASK = 'QWEN_GDN_DUALM_AWARE_ROTATION_TRAINING_READINESS_V1'
PHASES = ('dry-run', 'admission', 'teacher', 'train', 'validate', 'export')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path):
    value = json.loads(Path(path).read_text())
    if value.get('TASK_ID') != TASK:
        raise ValueError('Wrong paired task identity')
    from training_state_io import GDN_LAYERS
    common = value['common']
    expected = dict(parameters_per_layer=8128, total_trainable_parameters=195072,
        batch_size=1, optimizer_updates=125, tokens_per_document=1024,
        captures_per_document=8, target_exposures=1000, gradient_horizon=64,
        seed=0, base_model_trainable=False)
    if any(common.get(key) != wanted for key, wanted in expected.items()):
        raise ValueError('Paired common numerical/training contract drift')
    if common['rotation_layers'] != list(GDN_LAYERS):
        raise ValueError('Paired layer roster changed')
    optimizer = dict(name='Adam', lr=.001, betas=[.9, .999], eps=1e-8,
                     weight_decay=0., gradient_clip_global_norm=1.)
    if common['optimizer'] != optimizer:
        raise ValueError('Paired optimizer contract changed')
    if common['loss']['lambda_state'] != .1 or common['loss']['epsilon'] != 1e-12:
        raise ValueError('Original loss weights/epsilon changed')
    safety = common['safety']
    if (safety['orthogonality_max_abs'], safety['catastrophic_gradient_norm'], safety['memory_creep_bytes']) != (1e-4, 1e6, 256*1024**2):
        raise ValueError('Original numerical bounds changed')
    arms = value['arms']
    allowed_keys = {'training_codec', 'backward', 'loss_selector', 'output_name'}
    if set(arms) != {'A', 'D'} or any(set(a) != allowed_keys or a['loss_selector'] != 'C6R' for a in arms.values()):
        raise ValueError('Per-arm override outside codec/backward/labels')
    if arms['A']['training_codec'] != 'CANONICAL_C128' or arms['D']['training_codec'] != 'FROZEN_DUAL_M_INT8_ALL8_ONES_FP16_SCALES_NO_PIVOTS':
        raise ValueError('Paired codec identities changed')
    if value['export']['primary_checkpoint'] != 'FINAL_UPDATE_125_ONLY_BOTH_ARMS':
        raise ValueError('Symmetric final125 checkpoint selection changed')
    return value


def render_plan(config, config_path, phase, arm):
    from authoritative_training_ast import audit
    recovered = audit(ROOT)
    env = json.loads((ROOT / 'ENVIRONMENT_PLAN.json').read_text())
    cost = json.loads((ROOT / 'A800_TRAINING_COST_MODEL.json').read_text())
    return dict(status='DRY_RUN_ONLY', phase_preview=phase, requested_arm=arm,
        config_path=str(Path(config_path).resolve()), config_sha256=sha(config_path),
        configuration=config, authoritative_AST_audit=recovered,
        environment=dict(Python=env['python_version'], Torch=env['historical_numerical_packages']['torch'],
            Transformers=env['transformers_version'], full_Transformers_source_required=True,
            architecture_sha256=env['architecture_SHA256'], FLA=env['FLA']),
        scientific_pair=dict(common_engine='original C6R captured functional + 0.1 state; persistent H64 student',
            A_codec=config['arms']['A']['training_codec'], D_codec=config['arms']['D']['training_codec'],
            sole_planned_between_arm_scientific_variable='codec',
            common_new_corpus='RepoBench versus historical WikiText is shared by A and D; G_adapt cannot be attributed entirely to codec'),
        checkpoint_export=dict(primary='final125 only for both arms', historical_C6R_selected_update94='baseline only',
            state_io='training_state_io.py; rotation + optimizer + Python/Torch/optional NumPy RNG; legal document boundary',
            micro_export_forbidden=True),
        cost_gate=cost['current_cost_gate'],
        neutral_cost_scenario_hours=cost['scenario_not_prediction']['total_hours'],
        proposed_paid_cap_hours=cost['PROPOSED_A800_BILLED_HOURS_CAP'],
        actual_hourly_price='UNKNOWN',
        data_status=config['data']['status'], preregistration_frozen=config['frozen'],
        FULL_MODEL_A800_TRAINING='NOT_TESTED',
        FULL_MODEL_RUNNER_IMPLEMENTATION='BLOCKED_UNCLOSED_CONTROL_DEPENDENCIES',
        execution_performed=False, torch_imported='torch' in sys.modules,
        next_action='STOP; no A800 rental/execution or complete training is authorized')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'PAIRED_TRAINING_PREREGISTRATION.json')
    parser.add_argument('--phase', choices=PHASES, default='dry-run')
    parser.add_argument('--arm', choices=('both', 'A', 'D'), default='both')
    parser.add_argument('--execute', action='store_true', help='Always refused in this readiness package')
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.execute:
            # This package contains no paid/model execution path, even if a JSON
            # authorization flag is edited. No Torch, CUDA, model, SSH or launch
            # import occurs before refusal. A future authorized task must close
            # the missing real helpers and freeze/qualify a new implementation.
            print(json.dumps(dict(status='REFUSED', reason='A800_AND_FULL_TRAINING_NOT_AUTHORIZED; FULL_MODEL_RUNNER_BLOCKED',
                phase=args.phase, arm=args.arm, execution_performed=False,
                torch_imported='torch' in sys.modules), ensure_ascii=False))
            return 4
        print(json.dumps(render_plan(config, args.config, args.phase, args.arm), ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps(dict(status='FAIL_CLOSED', error=str(exc), execution_performed=False,
                              torch_imported='torch' in sys.modules), ensure_ascii=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
