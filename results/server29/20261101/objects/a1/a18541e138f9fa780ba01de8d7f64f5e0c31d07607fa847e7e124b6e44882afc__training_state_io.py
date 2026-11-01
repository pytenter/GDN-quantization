"""Legal document-boundary checkpoint/export helpers; no implicit Torch import.

These helpers are not a model runner or proof of full-model resumability. They
must be called only after a whole document optimizer step, after discarding its
cache. A micro-test may validate them without creating a scientific export.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import sys

GDN_LAYERS = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30)
IDENTITY_KEYS = ('preregistration_sha256', 'data_split_sha256',
                 'teacher_manifest_sha256', 'source_manifest_sha256', 'arm', 'codec')


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def validate_boundary_context(context):
    if set(IDENTITY_KEYS) - set(context):
        raise ValueError('Missing frozen checkpoint identity')
    for key in IDENTITY_KEYS[:4]:
        value = context[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Invalid frozen SHA256: ' + key)
    if context['arm'] not in ('A', 'D'):
        raise ValueError('Only paired A/D checkpoints')
    if context['codec'] != {'A': 'CANONICAL_C128', 'D': 'DUAL_M_INT8'}[context['arm']]:
        raise ValueError('Arm/codec mismatch')
    update = context.get('completed_updates')
    if type(update) is not int or not 0 <= update <= 125:
        raise ValueError('Invalid completed document updates')
    if context.get('next_document_order_index') != update:
        raise ValueError('Checkpoint order cursor mismatch')
    if context.get('boundary') != 'AFTER_DOCUMENT_OPTIMIZER_STEP_CACHE_DISCARDED':
        raise ValueError('Mid-document resume is not implemented')
    if context.get('checkpoint_kind') not in ('MICRO_QUALIFICATION_ONLY', 'FORMAL_PAIRED_TRAINING'):
        raise ValueError('Explicit checkpoint scope required')
    captures = context.get('captures_per_update')
    tokens = context.get('tokens_per_update')
    if type(captures) is not int or captures < 1 or type(tokens) is not int or tokens < 1:
        raise ValueError('Explicit actual token/capture exposure required')
    if context['checkpoint_kind'] == 'FORMAL_PAIRED_TRAINING' and (captures, tokens) != (8, 1024):
        raise ValueError('Formal exposure contract changed')
    if context.get('completed_captures') != captures * update:
        raise ValueError('Checkpoint exposure mismatch')
    return context


def validate_checkpoint_envelope(payload, expected_context):
    validate_boundary_context(payload['context'])
    validate_boundary_context(expected_context)
    if payload.get('format_version') != 1 or payload.get('model_weights_included') is not False:
        raise ValueError('Unknown or inappropriate checkpoint payload')
    if payload['context'] != expected_context:
        raise ValueError('Checkpoint identities/order do not match expected legal boundary')
    for key in ('bank', 'optimizer', 'python_random_state', 'torch_cpu_rng_state', 'torch_cuda_rng_states'):
        if key not in payload:
            raise ValueError('Incomplete rotation/optimizer/RNG checkpoint: ' + key)
    return payload


def _cpu_tree(value, torch):
    if torch.is_tensor(value):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(v, torch) for key, v in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(v, torch) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(v, torch) for v in value)
    return value


def _atomic_torch(path, payload, torch):
    path = Path(path)
    if path.exists():
        raise FileExistsError('No overwrite of checkpoint/export: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    try:
        with temporary.open('xb') as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        # Atomic no-clobber commit on the same filesystem.
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {'path': str(path), 'sha256': file_sha256(path), 'bytes': path.stat().st_size}


def save_legal_boundary_checkpoint(path, bank, optimizer, torch, context):
    """Save complete rotation/Adam/RNG state, never a recurrent mid-doc cache."""
    validate_boundary_context(context)
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
    payload = dict(format_version=1, context=dict(context), model_weights_included=False,
        bank=_cpu_tree(bank.state_dict(), torch), optimizer=_cpu_tree(optimizer.state_dict(), torch),
        python_random_state=random.getstate(), torch_cpu_rng_state=torch.get_rng_state().cpu(),
        torch_cuda_rng_states=[s.cpu() for s in cuda_states])
    if 'numpy' in sys.modules:
        payload['numpy_random_state'] = sys.modules['numpy'].random.get_state()
    validate_checkpoint_envelope(payload, context)
    return _atomic_torch(path, payload, torch)


def restore_legal_boundary_checkpoint(path, bank, optimizer, torch, expected_context, expected_sha256):
    """Load only a hash-verified self-owned checkpoint and restore RNG last."""
    if file_sha256(path) != expected_sha256:
        raise ValueError('Checkpoint SHA256 mismatch')
    payload = torch.load(Path(path), map_location='cpu', weights_only=False)
    validate_checkpoint_envelope(payload, expected_context)
    saved_cuda = payload['torch_cuda_rng_states']
    if saved_cuda and (not torch.cuda.is_initialized() or len(saved_cuda) != torch.cuda.device_count()):
        raise ValueError('CUDA RNG device inventory differs; no silent CPU/GPU resume conversion')
    if 'numpy_random_state' in payload and 'numpy' not in sys.modules:
        raise ValueError('Checkpoint used NumPy RNG but restored runtime has not imported NumPy')
    bank.load_state_dict(payload['bank'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    random.setstate(payload['python_random_state'])
    torch.set_rng_state(payload['torch_cpu_rng_state'])
    if saved_cuda:
        torch.cuda.set_rng_state_all(saved_cuda)
    if 'numpy_random_state' in payload:
        sys.modules['numpy'].random.set_state(payload['numpy_random_state'])
    return dict(status='RESTORED_REQUIRES_CALLER_EQUIVALENCE_CHECK', context=payload['context'],
                checkpoint_sha256=expected_sha256, restored=['bank', 'optimizer', 'Python RNG', 'Torch CPU RNG', 'Torch CUDA RNG'])


def export_final_rotation(path, bank, torch, context, checkpoint_sha256):
    """Materialize the unique final125 primary Delta; reject every micro bank."""
    validate_boundary_context(context)
    if context['checkpoint_kind'] != 'FORMAL_PAIRED_TRAINING' or context['completed_updates'] != 125:
        raise ValueError('Only formal final125 can be a primary scientific rotation export')
    matrices = {str(layer): bank.layer(layer).matrix().detach().float().cpu().contiguous().clone()
                for layer in GDN_LAYERS}
    if any(tuple(v.shape) != (128, 128) or not bool(torch.isfinite(v).all()) for v in matrices.values()):
        raise ValueError('Invalid final rotation shape/nonfinite')
    maximum = max(float((v.T @ v - torch.eye(128)).abs().max()) for v in matrices.values())
    if maximum > 1e-4:
        raise ValueError('Frozen orthogonality threshold exceeded')
    payload = dict(format_version=1, arm=context['arm'], matrices=matrices,
        layer_ids=GDN_LAYERS, matrix_dtype='torch.float32', matrix_shape=[128, 128],
        matrix_meaning='materialized Delta; H128 separately',
        orientation='q/k.float() @ H128 @ Delta; recover Delta then H128 on key axis',
        update=125, exposures=1000, selection_rule='FINAL_UPDATE_125_ONLY_NO_LOSS_SELECTION',
        theta_checkpoint_sha256=checkpoint_sha256, context=dict(context), orthogonality_max_abs=maximum)
    return _atomic_torch(path, payload, torch)
