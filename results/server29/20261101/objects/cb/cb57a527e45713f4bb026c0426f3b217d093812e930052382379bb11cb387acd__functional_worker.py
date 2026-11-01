"""Independent serial adapter; original V4 numerical operations are immutable.

Qualification only until all gates and Secondary definitions are frozen.
No optimizer, generation, document selection or calibration is imported.
"""
import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
import types
from pathlib import Path

sys.dont_write_bytecode = True
OUT = Path(os.environ['LING_PROSPECTIVE_RESULTS']).resolve()
V4_CODE = Path('/data/zypan/worktrees/ling-kda-p1-l7-efficient-recovery-v4')
OLD = Path('/data/zypan/experiment_runs/LING_KDA_L7_OBJECTIVE_REDUNDANCY_AND_PERSISTENT_VALIDATION_V1')
V4 = Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_EFFICIENT_RECOVERY_V4')
sys.path.insert(0, str(V4_CODE))
import persistent_engine_v1 as engine
from v2_runtime import initialize, snapshot
from run_matched_l7 import PRIOR, load_module
from gpu_admission import device, proc_identity, route


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def publish(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('x') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.link(tmp, path); tmp.unlink()


def transfer(torch, obj, target, memo=None):
    """Preserve native object/attribute schema without making a GPU deepcopy."""
    if memo is None:
        memo = {}
    if id(obj) in memo:
        return memo[id(obj)]
    if torch.is_tensor(obj):
        result = obj.detach().to(target, copy=True)
    elif isinstance(obj, dict):
        result = {}; memo[id(obj)] = result
        result.update({k: transfer(torch, v, target, memo) for k, v in obj.items()})
    elif isinstance(obj, list):
        result = []; memo[id(obj)] = result
        result.extend(transfer(torch, v, target, memo) for v in obj)
    elif isinstance(obj, tuple):
        result = tuple(transfer(torch, v, target, memo) for v in obj)
    elif isinstance(obj, (type, types.FunctionType, types.MethodType, torch.device)):
        result = obj
    elif hasattr(obj, '__dict__'):
        result = copy.copy(obj); memo[id(obj)] = result
        for name, value in vars(obj).items():
            setattr(result, name, transfer(torch, value, target, memo))
    else:
        result = obj
    memo[id(obj)] = result
    return result


def object_fingerprint(torch, obj):
    h = hashlib.sha256(); memo = {}
    def visit(x):
        if torch.is_tensor(x):
            h.update(json.dumps([list(x.shape), str(x.dtype)]).encode())
            h.update(engine.tensor_bytes(torch, x))
        elif isinstance(x, (dict, list, tuple)) or (hasattr(x, '__dict__') and not isinstance(x, type)):
            if id(x) in memo:
                h.update(f'alias:{memo[id(x)]}'.encode()); return
            memo[id(x)] = len(memo)
            h.update(f'{type(x).__module__}.{type(x).__qualname__}'.encode())
            if isinstance(x, dict):
                for k in sorted(x, key=str):
                    h.update(str(k).encode()); visit(x[k])
            elif isinstance(x, (list, tuple)):
                for v in x: visit(v)
            else:
                for k in sorted(vars(x)):
                    h.update(k.encode()); visit(getattr(x, k))
        elif isinstance(x, type):
            h.update(f'class:{x.__module__}.{x.__qualname__}'.encode())
        else:
            h.update(repr(x).encode())
    visit(obj)
    return h.hexdigest()


def audit_state(audit, condition):
    return {k: getattr(audit, k)[condition] for k in
            ('previous', 'writes', 'consumed', 'after_anchors', 'proofs')}


def restore_audit(audit, condition, state):
    for k, value in state.items():
        getattr(audit, k)[condition] = value


class HiddenObserver:
    def __init__(self, torch, capture, enabled):
        self.enabled, self.value = enabled, None
        def observe(module, inputs):
            if self.enabled:
                self.value = inputs[0].detach().cpu().clone()
        self.handle = capture.model.lm_head.register_forward_pre_hook(observe)

    def close(self):
        self.handle.remove()


def checkpoint(torch, cache, audit, condition, position, cfg, spec_sha, token_sha, rotation_sha, prefix_hash):
    cpu_cache = transfer(torch, cache, 'cpu')
    original = object_fingerprint(torch, cache)
    assert object_fingerprint(torch, cpu_cache) == original, 'CPU_CHECKPOINT_COPY_MUTATION'
    state = None if condition == 'FP' else transfer(torch, audit_state(audit, condition), 'cpu')
    return dict(schema='full_native_hybrid_cache_v1', cache=cpu_cache,
        cache_fingerprint=original, audit_state=state, next_position=position + 1,
        last_position=position, recurrent_layers=cfg['layers'],
        quantizer_state=dict(kind='original_stateless_INT8_R128', adaptive_state=None,
            rotation_sha256=rotation_sha, every_token=True, dequantized_carrier=True,
            no_high_precision_shadow=True, audit_counters_and_next_read_state=state),
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
        python_rng=random.getstate(), numpy_rng=__import__('numpy').random.get_state(),
        frozen_spec_sha256=spec_sha, token_sha256=token_sha,
        prefix_full_logit_sha256=prefix_hash)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--document', required=True)
    ap.add_argument('--condition', required=True)
    ap.add_argument('--stage', choices=('parent', 'resumed_parity'), required=True)
    a = ap.parse_args()
    tag = f'{a.document}__{a.condition}__{a.stage}'
    run = OUT / 'runs'; run.mkdir(exist_ok=True)
    started = time.time(); capture = audit = observer = None
    try:
        assert not (OUT / 'STOP.json').exists(), 'FAILED_GATE_STOP_NO_RETRY'
        assert not (run / (tag + '.json')).exists(), 'ALREADY_RUN'
        spec = read(OUT / 'FROZEN_SPEC.json')
        assert spec['primary']['positions'] == [8192 + 64*k for k in range(384)]
        assert sha(__file__) == read(OUT / 'CODE_FREEZE.json')['functional_worker.py']
        assert sha(V4_CODE / 'persistent_engine_v1.py') == spec['engine_sha256']
        from prospective_common import verify
        cfg = verify(); gpu = route(cfg)
        entry = next(x for x in spec['documents'] if x['document'] == a.document)
        assert any(u['document']==a.document and u['rotation']==a.condition for u in spec['new_units']), 'UNAUTHORIZED_NEW_SCIENTIFIC_KEY'
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(gpu)
        publish(run / (tag + '.STARTED.json'), dict(pid=os.getpid(), started=started, stage=a.stage,
            document=a.document, condition=a.condition, spec_sha256=sha(OUT / 'FROZEN_SPEC.json')))
        import torch
        runtime = [initialize(torch, cfg['seed'])]
        identity = proc_identity(os.getpid())
        hardware = device(gpu, cfg, evidence=run / (tag + '.GPU_EXCLUSIVITY.jsonl'))
        assert hardware['exclusive'], hardware['failure_conditions']
        base = load_module(PRIOR / 'run_short_horizon_proxy_v2.py', 'functional_verified_base')
        direction = load_module(base.DIRECTION_RUNNER, 'functional_verified_direction')
        host = direction.gpu_guard(torch, gpu)
        assert all(host[k] == cfg['runtime'][k] for k in ('kernel', 'driver', 'torch', 'cuda_runtime', 'selector'))
        _, _, capture_module, core = direction.load_assets(gpu)
        from prospective_common import load_inputs
        _, tokens, token_path, rotations, rotation_ids = load_inputs(torch, cfg, a.document,
            [a.condition])
        token_sha = sha(token_path)
        assert token_sha == entry['token_file']['sha256']
        loading_start = time.perf_counter()
        capture = capture_module.Capture(); capture.model.eval()
        assert not any(p.requires_grad for p in capture.model.parameters())
        assert len(capture.model.model.layers) == 24
        runtime.append(snapshot(torch, 'after_original_helpers', cfg['seed']))
        loading_seconds = time.perf_counter() - loading_start
        audit = engine.PersistentAudit(torch, direction, core, capture, cfg['layers'], [a.condition])
        rotation = rotations.get(a.condition)
        if rotation is not None: rotation = rotation.to(capture.device)
        # ON in the uninterrupted parent and OFF in fresh resumed parity.
        # Their full logits and complete native cache must be identical.
        observer = HiddenObserver(torch, capture, a.stage == 'parent')
        rotation_sha = None if a.condition == 'FP' else rotation_ids[a.condition]['sha256']
        checkpoint_path = run / f'{a.document}__{a.condition}__8192.pt'
        prefix = hashlib.sha256(); suffix = hashlib.sha256()
        prefix512 = None; memory = []; inventories = {}
        if a.stage == 'parent':
            cache, begin = None, 0
        else:
            assert sha(checkpoint_path) == read(run / f'{a.document}__{a.condition}__parent.json')['checkpoint_sha256']
            payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            assert payload['next_position'] == 8192
            assert payload['frozen_spec_sha256'] == sha(OUT / 'FROZEN_SPEC.json')
            assert payload['token_sha256'] == token_sha
            assert payload['quantizer_state']['rotation_sha256'] == rotation_sha
            cache = transfer(torch, payload['cache'], capture.device)
            assert object_fingerprint(torch, cache) == payload['cache_fingerprint'], 'CHECKPOINT_RESTORE_NOT_EXACT'
            if a.condition != 'FP':
                restore_audit(audit, a.condition, transfer(torch, payload['audit_state'], capture.device))
            torch.set_rng_state(payload['torch_rng']); torch.cuda.set_rng_state(payload['cuda_rng'])
            random.setstate(payload['python_rng']); __import__('numpy').random.set_state(payload['numpy_rng'])
            begin = payload['next_position']; del payload; gc.collect()
        torch.cuda.reset_peak_memory_stats()
        loop_start = time.perf_counter()
        for position in range(begin, 8256):
            assert not (OUT / 'STOP.json').exists(), 'SAFE_TOKEN_STOP'
            assert torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
            audit.active = None if a.condition == 'FP' else a.condition
            cache, logits, _ = direction.model_step(torch, capture, cache, tokens[position], position)
            assert bool(torch.isfinite(logits).all()), 'NONFINITE_LOGITS'
            if a.condition != 'FP':
                audit.quantize(a.condition, cache, rotation, position,
                    position in (0, 511, 8191, 8192, 8255))
            data = engine.tensor_bytes(torch, logits)
            if position < 8192:
                prefix.update(position.to_bytes(8, 'little')); prefix.update(data)
            else:
                suffix.update(position.to_bytes(8, 'little')); suffix.update(data)
            if position == 511:
                prefix512 = prefix.hexdigest()
                # This scientific key did not exist in the old 512 roster.
                # Fresh-process exact continuation is compared before science.
            if position == 8191:
                saved = checkpoint(torch, cache, audit, a.condition, position, cfg,
                    sha(OUT / 'FROZEN_SPEC.json'), token_sha, rotation_sha, prefix.hexdigest())
                tmp = checkpoint_path.with_suffix('.pt.partial')
                assert not checkpoint_path.exists()
                torch.save(saved, tmp)
                with tmp.open('rb') as f: os.fsync(f.fileno())
                os.link(tmp, checkpoint_path); tmp.unlink(); del saved; gc.collect()
            if position in (511, 8191, 8255):
                inventories[str(position)] = engine.hybrid_inventory(torch, cache, position, cfg['layers'], 24)
            if (position + 1) % 512 == 0 or position == 8255:
                guard = device(gpu, cfg, os.getpid(), binding=identity, require_context=True,
                    evidence=run / (tag + '.GPU_EXCLUSIVITY.jsonl'))
                assert guard['exclusive'], guard['failure_conditions']
                row = dict(position=position + 1, elapsed=time.perf_counter() - loop_start,
                    allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                    peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
                    inactive_split=torch.cuda.memory_stats().get('inactive_split_bytes.all.current', 0))
                memory.append(row); print(json.dumps(dict(event='qualification_progress', tag=tag, **row)), flush=True)
            del logits, data
            observer.value = None
        final_cache = object_fingerprint(torch, cache)
        integrity = None if a.condition == 'FP' else audit.integrity(a.condition, 8256)
        publish(run / (tag + '.json'), dict(status='PASS', stage=a.stage, document=a.document,
            condition=a.condition, begin=begin, endpoint=8256, prefix512=prefix512,
            parity_positions=[8192,8256], uninterrupted_suffix_full_logit_sha256=suffix.hexdigest(),
            final_full_native_cache_sha256=final_cache,
            audit_state_fingerprint=None if a.condition == 'FP' else object_fingerprint(torch, audit_state(audit,a.condition)),
            checkpoint_path=str(checkpoint_path), checkpoint_sha256=sha(checkpoint_path),
            observer_enabled=a.stage == 'parent', runtime=runtime, host=host,
            integrity=integrity, cache_inventory=inventories, memory=memory,
            loading_seconds=loading_seconds, elapsed_seconds=time.time()-started,
            completed_epoch=time.time(), science_outcomes_read=False))
    except BaseException as e:
        failure = dict(status='STOP', stage=a.stage, document=a.document, condition=a.condition,
            reason=repr(e), traceback=traceback.format_exc(), started_epoch=started,
            stopped_epoch=time.time(), physical_resource_stop='out of memory' in str(e).lower(),
            no_retry=True, no_new_training=True, science_verdict='INCOMPLETE_NOT_EVALUATED')
        if not (run / (tag + '.STOP.json')).exists(): publish(run / (tag + '.STOP.json'),failure)
        if not (OUT / 'STOP.json').exists(): publish(OUT / 'STOP.json',failure)
        raise
    finally:
        if observer is not None: observer.close()
        if audit is not None: audit.close()
        if capture is not None: capture.close()


if __name__ == '__main__':
    main()
