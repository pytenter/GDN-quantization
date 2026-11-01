#!/usr/bin/env python3
"""A800 cleanroom qualification only; guarded one-shot canaries and full loads."""
import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
HIST = Path('/root/autodl-tmp/GDN-formal-c5-c6-h64-v1')
C7 = HIST / 'experiments/QWEN_GDN_C7_PERSISTENT_FINAL_HIDDEN_OBJECTIVE_V1'
MODEL = Path('/root/autodl-tmp/models/Qwen3.5-9B')
CANONICAL = Path('/root/autodl-tmp/canonical_source')
HEAD_HASH = '6d6ff4dbf261f833246d32e7c6088f87f4ab40ddb58516d90c931aa6ae61d5e7'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def save(name, data):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(data, f, sort_keys=True, indent=2, allow_nan=False, default=str); f.write('\n')


def host():
    xml = ET.parse(ROOT / 'provenance/nvidia_smi.xml').getroot()
    gpu = xml.find('gpu')
    def field(path): return gpu.findtext(path)
    data = {'gpu_model': field('product_name'), 'gpu_uuid': field('uuid'), 'pci_bus_id': field('pci/pci_bus_id'),
            'vram': field('fb_memory_usage/total'), 'mig_current': field('mig_mode/current_mig'),
            'mig_pending': field('mig_mode/pending_mig'), 'virtualization_mode': field('gpu_virtualization_mode/virtualization_mode'),
            'driver': xml.findtext('driver_version'), 'cuda_driver_max_api': xml.findtext('cuda_version'),
            'kernel': platform.release(), 'platform': platform.platform(), 'hostname': platform.node(),
            'initial_python': sys.version, 'os_release': Path('/etc/os-release').read_text(),
            'nvidia_smi_xml_sha256': sha(ROOT / 'provenance/nvidia_smi.xml'),
            'interpretation': 'Full 80GB A800 PCIe exposed; MIG Disabled and virtualization None. No evidence of slicing exposed. This is not proof of every host-level implementation detail.',
            'no_virtualization_configuration_change': True, 'qualification_status': 'IDENTITY_RECORDED_INTEGRITY_PENDING'}
    if len(xml.findall('gpu')) != 1 or 'A800' not in data['gpu_model']:
        raise RuntimeError('unexpected GPU identity')
    save('provenance/a800_host.json', data)
    print(json.dumps(data), flush=True)


def parent():
    spec = importlib.util.spec_from_file_location('a800_historical_c7', C7 / 'scripts/c7_pipeline.py')
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


def verify_cpu_checkpoint(model, output):
    """Fresh direct safetensors comparison, not old GPU or teacher-cache reuse."""
    import trusted_materializer as T
    from safetensors import safe_open
    mapping = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
    items = T.state_items(model)
    patterns = list(getattr(model, '_keys_to_ignore_on_load_unexpected', []) or [])
    matched, rows = {}, []
    for shard in sorted(set(mapping.values())):
        with safe_open(str(MODEL / shard), framework='pt', device='cpu') as f:
            if set(f.keys()) != {k for k, v in mapping.items() if v == shard}:
                raise RuntimeError('checkpoint shard/index inventory mismatch')
            for key in sorted(f.keys()):
                value = f.get_tensor(key)
                candidates = list(dict.fromkeys(n for n in [key, key.replace('model.language_model.', 'model.', 1)] if n in items))
                if len(candidates) > 1: raise RuntimeError('ambiguous canonical checkpoint mapping')
                row = {'checkpoint_key': key, 'shard': shard, 'checkpoint_dtype': str(value.dtype), 'shape': list(value.shape)}
                if candidates:
                    name = candidates[0]
                    if name in matched: raise RuntimeError('duplicate runtime checkpoint destination')
                    expected = value.to(dtype=items[name]['tensor'].dtype)
                    result = T.fingerprint(items[name]['tensor'], expected)
                    if not result['exact_equal'] or not result['finite']:
                        save(output + '.first_mismatch.json', {'name': name, 'result': result})
                        raise RuntimeError('fresh direct CPU checkpoint comparison failed: ' + name)
                    matched[name] = result['raw_sha256']
                    row.update(runtime_name=name, classification='CHECKPOINT_BACKED', result=result)
                else:
                    reasons = [p for p in patterns if re.search(p, key)]
                    if not reasons: raise RuntimeError('unexplained canonical excluded checkpoint key: ' + key)
                    result = T.fingerprint(value)
                    if not result['finite']: raise RuntimeError('excluded source tensor nonfinite')
                    row.update(classification='CANONICAL_TEXT_CLASS_EXCLUDED_VISION_OR_MTP', reasons=reasons, result=result)
                rows.append(row)
        print(json.dumps({'event': 'fresh_checkpoint_shard_audited', 'shard': shard}), flush=True)
    if any(n not in matched and x['kind'] == 'PARAMETER' for n, x in items.items()):
        raise RuntimeError('runtime parameter has no checkpoint source')
    save(output, {'status': 'PASS', 'checkpoint_tensors': rows, 'matched_runtime_hashes': matched,
        'model_created_buffers': [n for n in items if n not in matched], 'no_parameter_exemption': True})
    return matched


def environment():
    M = parent()
    p = M.protocol()
    import torch, transformers
    versions = {name: importlib.metadata.version(name) for name in ['torch', 'triton', 'safetensors', 'numpy', 'tokenizers', 'huggingface_hub', 'accelerate']}
    expected = {'torch': '2.5.1+cu121', 'triton': '3.1.0', 'safetensors': '0.8.0', 'numpy': '2.2.6',
                'tokenizers': '0.23.1', 'huggingface_hub': '1.28.0', 'accelerate': '1.12.0'}
    if versions != expected or transformers.__version__ != '5.16.0.dev0' or torch.version.cuda != '12.1':
        raise RuntimeError('chosen cleanroom package identity differs')
    if sys.version_info[:3] != (3, 10, 18): raise RuntimeError('chosen Python identity differs')
    source_rows = []
    for path in sorted(CANONICAL.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and '.git' not in path.parts and '.egg-info' not in str(path):
            source_rows.append({'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size})
    model_files = []
    for path in sorted(MODEL.iterdir()):
        if path.is_file(): model_files.append({'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size})
    save('provenance/source_manifest.json', {'frozen_historical_inputs': p['frozen_inputs'], 'canonical_source': source_rows,
        'model_files': model_files, 'old_c7_preregistration_sha256': sha(C7 / 'protocol/preregistration.json'),
        'historical_repository_head': json.loads((ROOT / 'provenance/historical_repository_remote_identity.json').read_text())['head'],
        'historical_repository_identity_sha256': sha(ROOT / 'provenance/historical_repository_remote_identity.json'),
        'old_caches_not_formal_sources': True, 'path_alias': {'/data/zypan': str(CANONICAL)}})
    historical = json.loads((M.OLD / 'configs/environment.json').read_text())
    save('provenance/environment.json', {'versions': versions, 'transformers': transformers.__version__,
         'transformers_path': transformers.__file__, 'torch_cuda_runtime': torch.version.cuda, 'python': sys.version,
         'FLA': 'NOT_INSTALLED_NOT_USED_CANONICAL_TORCH_GDN', 'historical_environment': historical,
         'fresh_package_install': True, 'environment_not_cloned': True,
         'known_intended_differences': ['A800 GPU/driver/kernel versus historical vGPU', 'fresh pinned installation; same core numerical package and Python versions'],
         'requirements_sha256': sha(ROOT / 'scripts/a800_requirements.txt'),
         'pip_freeze_sha256': sha(ROOT / 'provenance/pip_freeze.txt'),
         'bulk_loading_path': 'canonical CPU from_pretrained, then canonical model.to(cuda:0), no tensor repairs; all GPU state compared exactly',
         'trusted': False, 'qualification_pending': True})
    print('ENVIRONMENT_RECONSTRUCTION_VERIFIED', flush=True)


def canary(run):
    import torch
    import trusted_materializer as T
    from safetensors import safe_open
    out = ROOT / f'qualification/canary_run{run}.json'
    if out.exists() or (ROOT / f'logs/qualification_canary_{run}.error.json').exists(): raise RuntimeError('canary already attempted')
    mapping = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
    shard = MODEL / mapping['lm_head.weight']
    with safe_open(str(shard), framework='pt', device='cpu') as f: w = f.get_tensor('lm_head.weight')
    cpu = T.fingerprint(w)
    if cpu['raw_sha256'] != HEAD_HASH or cpu['absmax'] != 0.35546875 or not cpu['finite']:
        raise RuntimeError('checkpoint canary differs from exact historical source')
    start = time.monotonic(); v = w.to('cuda:0'); torch.cuda.synchronize()
    gpu = T.fingerprint(v, w)
    record = {'run': run, 'pid': os.getpid(), 'CPU': cpu, 'GPU': gpu, 'shard': str(shard),
              'transfer_and_verification_seconds': time.monotonic() - start,
              'status': 'PASS' if gpu['exact_equal'] and gpu['finite'] else 'FAIL'}
    save(f'qualification/canary_run{run}.json', record)
    if record['status'] != 'PASS': raise RuntimeError('A800 transfer canary failed; do not retry')
    print(json.dumps({'event': 'canary', 'run': run, 'status': 'PASS'}), flush=True)


def full(run):
    canaries = [json.loads((ROOT / f'qualification/canary_run{i}.json').read_text()) for i in [1, 2, 3]]
    if any(x['status'] != 'PASS' for x in canaries) or len({x['pid'] for x in canaries}) != 3:
        raise RuntimeError('three independent canaries required')
    if (ROOT / f'qualification/full_model_run{run}.json').exists() or (ROOT / f'logs/qualification_full_{run}.error.json').exists(): raise RuntimeError('full model already attempted')
    M = parent(); M.seed(); torch = M.torch
    import trusted_materializer as T
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True, local_files_only=True)
    model, info = AutoModelForCausalLM.from_pretrained(str(MODEL), torch_dtype=torch.bfloat16, device_map=None,
         trust_remote_code=True, local_files_only=True, low_cpu_mem_usage=True, output_loading_info=True)
    for key in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']:
        if info.get(key): raise RuntimeError(f'canonical CPU load reports {key}: {info[key]}')
    verify_cpu_checkpoint(model, f'qualification/direct_checkpoint_run{run}.json')
    # Exact CPU checkpoint inventory was verified on the previous independent CPU audit;
    # now compare every fresh CPU runtime item to those pinned per-item expected hashes.
    reference = json.loads((HIST / 'experiments/QWEN_GDN_TRUSTED_GPU_MATERIALIZATION_AND_C7_RESUME_V1/materializer/runtime_state_inventory.json').read_text())
    expected_items = {r['name']: r for r in reference['items']}
    items = T.state_items(model)
    if set(items) != set(expected_items): raise RuntimeError('canonical full state inventory differs')
    sources = {n: x['tensor'].detach() for n, x in items.items()}
    aliases = T.alias_structure(items)
    for name, source in sources.items():
        result = T.fingerprint(source)
        if result['raw_sha256'] != expected_items[name]['expected_runtime_hash'] or not result['finite']:
            raise RuntimeError(f'fresh CPU source differs from checkpoint-backed inventory: {name}')
    del items
    model.to('cuda:0'); torch.cuda.synchronize(); M.core.freeze_model(model)
    fields = ['run', 'stage', 'name', 'classification', 'shape', 'dtype', 'numel', 'bytes', 'finite', 'nan_count', 'inf_count', 'absmax', 'exact_equal', 'raw_sha256', 'mismatch_bytes', 'verification_seconds']
    with (ROOT / f'qualification/full_model_integrity_run{run}.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        def emit(name, kind, stage, result):
            writer.writerow({'run': run, 'stage': stage, 'name': name,
                             'classification': expected_items[name]['classification'],
                             **{k: result[k] for k in fields if k in result}})
            f.flush(); os.fsync(f.fileno())
            if not result['exact_equal'] or not result['finite']:
                save(f'qualification/full_run{run}_first_mismatch.json', {'name': name, 'stage': stage, 'result': result})
            print(json.dumps({'run': run, 'name': name, 'stage': stage, 'exact_equal': result['exact_equal']}), flush=True)
        hashes = T.verify_model(model, sources, emit, 'FULL_GPU_RESIDENT')
        actual_aliases = T.alias_structure(T.state_items(model))
        # Freeze is canonical and changes requires_grad only, not aliases/layout.
        for x in aliases['layouts'].values(): x['requires_grad'] = False
        if actual_aliases != aliases: raise RuntimeError('full-load alias/layout changed')
        ids, _ = M.core.tokenize(tokenizer, M.core.corpus_rows('TRAIN')[0])
        hidden = []; outputs = []
        handle = model.lm_head.register_forward_pre_hook(lambda m, a: hidden.append(a[0].detach()))
        try:
            with torch.no_grad():
                for _ in range(2):
                    logits = model(input_ids=torch.tensor([ids[:8]], device='cuda:0'), use_cache=False).logits
                    h = hidden[-1]; explicit = torch.nn.functional.linear(h, model.lm_head.weight)
                    if not bool(torch.isfinite(h).all()) or not bool(torch.isfinite(logits).all()): raise RuntimeError('FP runtime nonfinite')
                    if not torch.equal(logits, explicit): raise RuntimeError('LM head exact reference parity failed')
                    outputs.append({'hidden': M.runtime.tensor_sha256(h), 'logits': M.runtime.tensor_sha256(logits),
                                    'absmax_logits': float(logits.abs().max()), 'hidden_norm': float(h.float().norm()), 'reference_exact': True})
        finally: handle.remove()
        if outputs[0] != outputs[1]: raise RuntimeError('FP repeated-forward exact determinism failed')
        after = T.verify_model(model, sources, emit, 'AFTER_FP_SMOKE')
        if after != hashes: raise RuntimeError('frozen state changed during short forward')
    save(f'qualification/full_model_run{run}.json', {'status': 'PASS', 'pid': os.getpid(), 'hashes': hashes,
         'aliases': actual_aliases, 'load_info': info, 'fp_smoke': outputs, 'state_items': len(hashes),
         'model_class': type(model).__name__, 'peak_allocated': torch.cuda.max_memory_allocated(), 'peak_reserved': torch.cuda.max_memory_reserved()})
    print(f'FULL_MODEL_RUN_{run}_PASS', flush=True)


def close():
    c = [json.loads((ROOT / f'qualification/canary_run{i}.json').read_text()) for i in [1, 2, 3]]
    v = [json.loads((ROOT / f'qualification/full_model_run{i}.json').read_text()) for i in [1, 2, 3]]
    if any(x['status'] != 'PASS' for x in c + v): raise RuntimeError('qualification not all PASS')
    if len({x['pid'] for x in v}) != 3: raise RuntimeError('full model processes not independent')
    if any(x['hashes'] != v[0]['hashes'] or x['aliases'] != v[0]['aliases'] or x['fp_smoke'] != v[0]['fp_smoke'] for x in v):
        raise RuntimeError('full-model repeatability failed')
    save('qualification/lm_head_canary.json', {'status': 'PASS', 'runs': c, 'LM_HEAD_CANARY_3X': 'PASS'})
    save('qualification/fp_runtime_stability.json', {'status': 'PASS', 'LM_HEAD_FORWARD_REFERENCE_PARITY': 'PASS', 'runs': [x['fp_smoke'] for x in v]})
    env = json.loads((ROOT / 'provenance/environment.json').read_text())
    source = json.loads((ROOT / 'provenance/source_manifest.json').read_text())
    save('provenance/trusted_environment_freeze.json', {'A800_TRAINING_ENVIRONMENT_TRUSTED': 'YES',
        'A800_HOST_QUALIFICATION': 'PASS', 'FULL_MODEL_GPU_PARAMETER_INTEGRITY': 'PASS',
        'FULL_MODEL_FRESH_PROCESS_REPEATABILITY': 'PASS', 'FP_RUNTIME_STABILITY': 'PASS',
        'gpu_identity_sha256': sha(ROOT / 'provenance/a800_host.json'), 'environment': env,
        'source_manifest_sha256': sha(ROOT / 'provenance/source_manifest.json'),
        'model_files': source['model_files'], 'code_hashes': {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT/'scripts').glob('*')) if p.is_file()},
        'cleanroom_code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT.parents[1], text=True).strip(),
        'package_changes_after_this_point_forbidden': True})
    print('A800_TRAINING_ENVIRONMENT_TRUSTED=YES', flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--phase', choices=['host', 'environment', 'canary', 'full', 'close'], required=True)
    p.add_argument('--run', type=int, choices=[1, 2, 3]); args = p.parse_args()
    for folder in ['provenance', 'qualification', 'logs']: (ROOT / folder).mkdir(parents=True, exist_ok=True)
    try:
        if args.phase in ['canary', 'full']: globals()[args.phase](args.run)
        else: globals()[args.phase]()
    except BaseException as exc:
        save(f'logs/qualification_{args.phase}_{args.run}.error.json', {'error': str(exc), 'traceback': traceback.format_exc(),
             'A800_TRAINING_ENVIRONMENT_TRUSTED': 'NO', 'scientific_updates': 0, 'do_not_retry_scientific_integrity_failure': True})
        raise


if __name__ == '__main__': main()
