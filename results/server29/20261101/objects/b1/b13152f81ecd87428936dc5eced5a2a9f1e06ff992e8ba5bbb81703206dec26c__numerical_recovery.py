"""Prospective, isolated V3 scan and full-model numerical qualification.

Two predeclared routes only. The historical FP32 fixed-order implementation is
route 1. Route 2 is a FP32 compensated fixed-order scan, used only after route 1
has a documented numerical ineligibility. FP64 is reference-only, never a
production recurrent state, teacher cache, or candidate accumulator.
"""
import argparse
import ast
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

TASK = 'QWEN_GDN_DETERMINISTIC_RECOVERY_TO_SCALE_DISCOVERY_V3'
REPO = Path('/data/zypan/worktrees/qwen-gdn-deterministic-scale-discovery-v3')
ROOT = REPO / 'experiments' / TASK
V2 = Path('/data/zypan/worktrees/qwen-gdn-scale-surrogate-input-recovery-v2/experiments/QWEN_GDN_SCALE_SURROGATE_INPUT_RECOVERY_AND_VALIDATION_V2')
OLD = Path('/data/zypan/worktrees/qwen-gdn-propagation-surrogate-validation-v1/experiments/QWEN_GDN_DETERMINISTIC_CUMSUM_CLOSURE_V1')
ARCH = Path('/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py')
FORMAL = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1/scripts/run_qwen_c5_c6_formal.py')
ARCH_SHA = '90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a'
FORMAL_SHA = '77b1e1f6dad8a0dd53a27cf5a47c02be86c26d2cf30732604ba6d3747e178f69'
FIXED_SHA = 'f1767d60da692119be4865e63424c0e4ac3a2193acbeec76cdd1ceaa755be31e'
SEED = 73619


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def check(ok, why):
    if not ok:
        raise RuntimeError(why)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 ** 2), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def save(rel, obj):
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def prose(rel, s):
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        f.write(s)


def pt_save(rel, obj):
    import torch
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f:
        torch.save(obj, f); f.flush(); os.fsync(f.fileno())
    return {'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'bytes': p.stat().st_size}


def tsha(x):
    import torch
    return hashlib.sha256(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def module(path, name):
    sys.path.insert(0, str(Path(path).parent))
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m; s.loader.exec_module(m)
    return m


def cmd(args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def resources():
    return {'utc': now(), 'disk_available_bytes': shutil.disk_usage('/data').free,
            'GPU_csv': cmd(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader']),
            'processes_csv': cmd(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader'])}


def fixed_scan(x, dim=-1):
    # The original candidate is imported unchanged, not reimplemented.
    return module(OLD / 'runtime_patch/deterministic_cumsum.py', 'v3_unchanged_fixed_scan').deterministic_cumsum_fixed_order(x, dim)


def compensated_scan(x, dim=-1):
    import torch
    check(x.dtype == torch.float32, 'candidate 2 requires actual original FP32 scan input')
    dim %= x.ndim
    if x.shape[dim] == 0:
        return torch.empty_like(x)
    acc = x.select(dim, 0)
    correction = torch.zeros_like(acc)
    values = [acc]
    for i in range(1, x.shape[dim]):
        term = x.select(dim, i)
        total = acc + term
        correction = correction + torch.where(acc.abs() >= term.abs(), (acc - total) + term, (term - total) + acc)
        acc = total
        values.append(acc + correction)
    return torch.stack(values, dim=dim)


def oracle_scan(x, dim=-1):
    """Reference-only offline CPU FP64 scan, rounded back before GDN consumes it."""
    import numpy as np
    import torch
    check(x.dtype == torch.float32, 'oracle original scan dtype')
    data = x.detach().cpu().double().numpy()
    values = np.cumsum(data, axis=dim, dtype=np.float64)
    return torch.from_numpy(values).to(dtype=x.dtype, device=x.device)


def backend(route):
    if route == 'fixed':
        check(sha(OLD / 'runtime_patch/deterministic_cumsum.py') == FIXED_SHA, 'historical candidate source changed')
        return module(OLD / 'runtime_patch/deterministic_cumsum.py', 'v3_fixed_scan_bound').deterministic_cumsum_fixed_order
    if route == 'compensated':
        return compensated_scan
    if route == 'oracle':
        return oracle_scan
    raise ValueError(route)


def runtime():
    import torch
    check(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'CUBLAS workspace')
    check(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'qualification mapping GPU0')
    check(torch.__version__ == '2.5.1+cu121' and torch.version.cuda == '12.1', 'runtime changed')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    return torch


def prepare():
    old = read(V2 / 'SOURCE_AND_ARTIFACT_AUDIT.json')
    check(sha(V2 / 'SHA256SUMS.txt') == 'd642889cc49d6f562fe9677051f1e843d738cf8525e1064174c6c7bab69de075', 'V2 seal')
    check(sha(ARCH) == ARCH_SHA and sha(FORMAL) == FORMAL_SHA, 'canonical source identity')
    check(sha(OLD / 'runtime_patch/deterministic_cumsum.py') == FIXED_SHA, 'fixed scan source identity')
    fixture_rows = read(OLD / 'provenance/frozen_cumsum_inputs.json')['fixtures']
    for r in fixture_rows:
        check(sha(OLD / r['path']) == r['sha256'], 'original real fixture identity ' + r['name'])
    docs = old['document_manifest']['documents']
    for doc in docs:
        check(sha(doc['token_path']) == doc['token_sha256'], 'document stream identity')
    policy = read(OLD / 'protocol/numerical_parity_policy.json')
    source = Path(__file__).read_text()
    fn = next(x for x in ast.parse(source).body if isinstance(x, ast.FunctionDef) and x.name == 'compensated_scan')
    routes = {
        'fixed': {'order': 1, 'source': str(OLD / 'runtime_patch/deterministic_cumsum.py'), 'sha256': FIXED_SHA,
                  'production_dtype': 'FP32 only', 'accumulation': 'unchanged original left-to-right inclusive scan'},
        'compensated': {'order': 2, 'source': str(Path(__file__)),
                        'function_sha256': hashlib.sha256(ast.get_source_segment(source, fn).encode()).hexdigest(),
                        'production_dtype': 'FP32 only; no double persistent/candidate accumulator',
                        'accumulation': 'fixed-order Neumaier compensated inclusive scan',
                        'entry_condition': 'only if fixed route is numerically ineligible; not on unknown engineering error'},
    }
    # This calibration subset is excluded from all held-out operator checks.
    # It validates the proposed standard error model, never fits a tolerance.
    import torch
    import numpy as np
    cal_input = torch.load(OLD / 'fixtures/historical32.pt', map_location='cpu', weights_only=True)['g'][:, :1, :1, :8]
    exact_cal = np.cumsum(cal_input.double().numpy(), axis=-1, dtype=np.float64)
    cal_rows = []
    for name in ['fixed', 'compensated']:
        value = backend(name)(cal_input, -1)
        error = np.abs(value.double().numpy() - exact_cal)
        cal_rows.append({'route': name, 'input_sha256': tsha(cal_input), 'shape': list(cal_input.shape),
                         'maximum_absolute_error': float(error.max()),
                         'per_prefix_absolute_error': error.reshape(-1).tolist(),
                         'input_maximum_absolute_value': float(cal_input.abs().max()),
                         'used_to_fit_or_enlarge_caps': False})
    save('SCAN_CALIBRATION.json', {'status': 'INDEPENDENT_CPU_CALIBRATION_COMPLETE', 'rows': cal_rows,
                                 'dtype': 'FP32 candidate, FP64 offline oracle',
                                 'qualification_tolerance_basis': 'standard gamma_k sum(abs(input)) roundoff model; no fitted safety factor'})
    save('EXISTING_ASSET_REUSE_AUDIT.json', {
        'utc': now(), 'V2_manifest_sha256': sha(V2 / 'SHA256SUMS.txt'), 'V1_identity_checks_reused': old['V1_asset_checks_reused'],
        'Direction_269_identity_checks_reused': True, 'heavy_audit_repeated': False,
        'historical_full_model_breakdown': old['closure_full_model_breakdown'],
        'historical_full_model_parity': 'FAIL', 'historical_fresh_process_determinism': 'NOT_EVALUATED',
        'original_teacher_capture_source': old['original_teacher_capture_source'],
        'sources': {'architecture': {'path': str(ARCH), 'sha256': ARCH_SHA}, 'formal': {'path': str(FORMAL), 'sha256': FORMAL_SHA}},
        'old_counterfactual_panel_path': '/data/zypan/worktrees/qwen-gdn-frozen-drive-counterfactual-replay-v1/experiments/QWEN_GDN_FROZEN_DRIVE_COUNTERFACTUAL_REPLAY_V1',
        'counterfactual_scope': 'old FAIL/PASS only; not restarted or reinterpreted',
        'old_roots': old['source_roots'], 'original_helpers': old['original_helpers'],
        'rotations': old['rotation_manifest'], 'documents': docs,
    })
    save('QUALIFICATION_POLICY.json', {
        'frozen_utc': now(), 'source_code_sha256': sha(Path(__file__)), 'routes': routes,
        'same_host_runtime': {'torch': '2.5.1+cu121', 'cuda': '12.1'},
        'scan_calibration': {'fixture': 'historical32', 'head': 0, 'chunk': 0, 'prefix_count': 8,
                             'calibration_role': 'independent check of dtype/length error model, no fitted multiplier'},
        'scan_validation': {'fixtures': fixture_rows, 'exclude_calibration': 'entire historical32 head0',
                            'binary_scale_factors': [2 ** -8, 1, 2 ** 8], 'fresh_processes': 3},
        'scan_error_rule': 'Each prefix k absolute error versus CPU FP64 cumulative sum <= (gamma_k(FP32)+gamma_k(FP64))*sum(abs(input[0:k])). No empirical inflation factor. Every prefix finite. Raw relative error also reported, not divided by near-zero as pass criterion.',
        'scan_unit_roundoff': {'FP32': 2 ** -24, 'FP64': 2 ** -53},
        'oracle': 'reference-only CPU FP64 scan at original chunk scan site, cast back to original FP32 before all remaining unmodified GDN operations. No FP64 recurrent buffer. Not a scientific teacher path.',
        'full_model': {'development_document': 1, 'held_out_qualification_document': 2,
                       'prefill_lengths': [2, 8, 32, 64, 256], 'following_cached_tokens': 2,
                       'teacher_semantics': 'actual multi-token prefill to capture position, then actual cached captured-token forward; original collector not changed to token stepping',
                       'fresh_processes': 3, 'comparison_reference': 'same deterministic model with reference-only mathematically accurate scan, all other operations unchanged',
                       'caps': {k: policy[k] for k in ['model_fp32', 'model_bf16']},
                       'caps_basis': 'reuse original prospective dtype-based full-model limits without relaxation, but explicitly test mathematical scan oracle rather than historical nondeterministic execution',
                       'required_objects': ['all 24 GDN outputs and updated states at every helper call', 'actual normalized final hidden at lm_head input', 'all full-model logits', 'all cache recurrent states at all forward boundaries'],
                       'all_caps_and_finite_required': True, 'strict_fresh_process_bitwise': True,
                       'identity_checks': 'source/checkpoint identities reused; model frozen, no optimizer/generation; same loaded weights across paired reference/candidate forward',
                       'stop_on_first_failed_numerical_object': True},
        'no_unknown_engineering_retry': True, 'no_policy_relaxation': True,
        'fixed_GPUh_cap': None, 'arbitrary_VRAM_percentage_cap': None,
        'science': {'status': 'NOT_STARTED', 'must_continue_if_all_gates_PASS': True,
                    'roster': ['H', 'C6R', 'C7'], 'negative_control': 'Native',
                    'documents': docs, 'matched_runtime_only': True},
    })
    prose('NUMERICAL_RECOVERY_PLAN.md', '''# V3 prospective numerical recovery

P0 reuse V1/V2 model/source/document/quantizer seals; verify only new dependencies and resources. Old failures remain unchanged.

P1 route 1: unchanged historical fixed-order FP32 scan. Calibrate only head0/first8 of one real captured fixture. Freeze the standard forward-roundoff bound before evaluating disjoint validation input. Test every real prefix, binary input-scale changes, three fresh strict processes. Native historical differences are descriptive, not a required bitwise pass.

P2 preserve multi-token teacher chunk-to-position semantics. Real Qwen forward: 2,8,32,64,256-token prefill followed by two actual cached token calls. Compare all recurrent helper intermediates, cache states, normalized final hidden and full logits against a separate reference-only FP64-scan oracle rounded back to FP32 at the scan boundary. Reuse the prior full-model numerical caps unchanged. A changed golden reference is explicit, not a retrospective PASS of historical parity. Repeat candidate forward in three fresh processes only after fidelity passes. Do not infer fidelity from repeatability alone.

If route 1 has a documented operator/full-model numerical ineligibility, evaluate the one preregistered FP32 compensated fixed-order route 2. Never change caps, runtime or model to make it pass. Unknown wrapper/identity/OOM/determinism failures stop rather than start route 2. Both numerical routes ineligible => STOP_NO_QUALIFIED_NUMERICAL_PATH.

P3 after numerical contract PASS: freeze and validate formal S1/S2 read-only observer OFF/ON exact main codes/cache/logit parity; Doc1 H+C6R 1024-token original eight-capture geometry qualification.

P4 execute Track B four frozen RepoBench documents and H/C6R/C7, Native independent control. True out_proj teacher targets and original B0/B1/B2 helpers. Freeze short scores before labels.

P5 qualify long-history resources, four documents continuous persistent-QDQ from token1 to8192; save actual complete anchors and prove resume identity before continuation. No clean reset or old incompatible drive stitching.

P6 only after short scores are sealed, continue the same matched panel to32768 and save original equal-token/equal-layer FP64 J boundaries and nonoverlapping E_future over0-based[8256,32768). No pairing to oldG, no outcome-based roster selection. Describe n=4 Discovery only. If any scientific/engineering gate fails preserve the earliest failure and completed stages, then stop.

No GPUh cap or 85% VRAM rule. Before enlarging a stage use actual memory/throughput/disk microbenchmark. No training, new rotation, C8, AIME, free generation, gradient optimization or independent KL pilot.
''')
    save('INITIAL_RESOURCES.json', resources())
    print('V3_PREPARED', flush=True)


def operator(route, index):
    torch = runtime()
    policy = read(ROOT / 'QUALIFICATION_POLICY.json')
    rows = policy['scan_validation']['fixtures']
    f = backend(route)
    outputs = {}; metrics = []; started = time.monotonic()
    import numpy as np
    for row in rows:
        payload = torch.load(OLD / row['path'], map_location='cpu', weights_only=True)
        x = payload['g']
        check(list(x.shape) == row['shape'] and str(x.dtype) == row['dtype'], 'actual shape/dtype identity')
        # A disjoint independent calibration subset is retained, never used to
        # enlarge the forward-error bound. Entire calibration head is excluded.
        if row['name'] == 'historical32':
            x = x[:, 1:]
        for factor in policy['scan_validation']['binary_scale_factors']:
            key = row['name'] + '/' + str(factor)
            v = (x * factor).to('cuda:0')
            start = time.monotonic(); out = f(v, -1); torch.cuda.synchronize()
            exact = np.cumsum(v.detach().cpu().double().numpy(), axis=-1, dtype=np.float64)
            sum_abs = np.cumsum(np.abs(v.detach().cpu().double().numpy()), axis=-1, dtype=np.float64)
            prefixes = np.arange(1, x.shape[-1] + 1, dtype=np.float64)
            u32 = 2 ** -24; u64 = 2 ** -53
            bounds = (prefixes * u32 / (1 - prefixes * u32) + prefixes * u64 / (1 - prefixes * u64)) * sum_abs
            err = np.abs(out.detach().cpu().double().numpy() - exact)
            finite = bool(np.isfinite(err).all())
            passed = bool(finite and np.all(err <= bounds))
            prefix_rows = []
            for k in range(x.shape[-1]):
                denom = np.maximum(np.abs(exact[..., k]), 1e-300)
                prefix_rows.append({'prefix': k + 1, 'max_abs_error': float(err[..., k].max()),
                                    'max_raw_relative_error': float((err[..., k] / denom).max()),
                                    'max_error_to_roundoff_bound_ratio': float(np.divide(err[..., k], bounds[..., k], out=np.zeros_like(err[..., k]), where=bounds[..., k] > 0).max()),
                                    'input_sum_abs_max': float(sum_abs[..., k].max()), 'PASS': bool(np.all(err[..., k] <= bounds[..., k]))})
            outputs[key] = out.detach().cpu()
            again = f(v, -1)
            check(torch.equal(out, again), 'same-process deterministic scan failure')
            metrics.append({'fixture': row['name'], 'scale_factor': factor, 'PASS': passed,
                            'finite': finite, 'input_shape': list(v.shape), 'input_stride': list(v.stride()),
                            'input_dtype': str(v.dtype), 'input_sha256': tsha(v), 'output_sha256': tsha(out),
                            'max_abs_error': float(err.max()), 'all_prefixes': prefix_rows,
                            'elapsed_seconds': time.monotonic() - start})
    artifact = pt_save(f'qualification/operator/{route}_{index}.pt', outputs)
    save(f'qualification/operator/{route}_{index}.json', {
        'route': route, 'index': index, 'pid': os.getpid(), 'utc': now(), 'rows': metrics,
        'PASS': all(x['PASS'] for x in metrics), 'artifact': artifact,
        'strict': torch.are_deterministic_algorithms_enabled(), 'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
        'TF32': torch.backends.cuda.matmul.allow_tf32, 'elapsed_seconds': time.monotonic() - started,
        'source_commit': cmd(['git', 'rev-parse', 'HEAD'], REPO),
        'policy_sha256': sha(ROOT / 'QUALIFICATION_POLICY.json'),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
    print('OPERATOR', route, index, all(x['PASS'] for x in metrics), flush=True)


def operator_gate(route):
    runs = [read(ROOT / f'qualification/operator/{route}_{i}.json') for i in [1, 2, 3]]
    hashes = [[r['output_sha256'] for r in run['rows']] for run in runs]
    same = all(x == hashes[0] for x in hashes)
    passed = same and all(x['PASS'] and x['strict'] and not x['warn_only'] for x in runs)
    save(f'qualification/{route}_OPERATOR_GATE.json', {'status': 'PASS' if passed else 'FAIL', 'fresh_process_bitwise': same,
          'source_route': route, 'fresh_processes': 3, 'all_prefix_forward_error_PASS': all(x['PASS'] for x in runs),
          'policy_sha256': sha(ROOT / 'QUALIFICATION_POLICY.json'), 'runs': [x['artifact'] for x in runs]})
    return passed


def chunk_binding(qmod, route):
    source = ARCH.read_text()
    check(sha(ARCH) == ARCH_SHA, 'architecture source identity')
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'torch_chunk_gated_delta_rule')
    original = ast.get_source_segment(source, node)
    check(original.count('g = g.cumsum(dim=-1)') == 1, 'one original scan callsite')
    patched = original.replace('g = g.cumsum(dim=-1)', 'g = v3_scan_backend(g, dim=-1)')
    ns = dict(vars(qmod)); ns['v3_scan_backend'] = backend(route)
    exec(compile(patched, str(ROOT / ('reference_only_' if route == 'oracle' else '') / 'compiled_original_chunk.py'), 'exec'), ns)
    return ns['torch_chunk_gated_delta_rule']


def compare(reference, candidate, kind):
    import torch
    import numpy as np
    p = read(ROOT / 'QUALIFICATION_POLICY.json')['full_model']['caps'][kind]
    a = reference.detach().double().cpu(); b = candidate.detach().double().cpu()
    d = (a - b).abs(); scale = max(1., float(a.abs().max()))
    floor = p['relative_floor_scale'] * scale
    l2 = float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(a).clamp_min(1e-12))
    max_abs = float(d.max()); max_rel = float((d / a.abs().clamp_min(floor)).max())
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    result = {'finite': finite, 'max_abs': max_abs, 'relative_L2': l2, 'max_rel_floor_stabilized': max_rel,
              'reference_max_abs': float(a.abs().max()), 'max_abs_cap': p['max_abs_scale'] * scale,
              'relative_L2_cap': p['relative_L2'], 'max_rel_cap': p['max_rel'], 'reference_floor': floor,
              'byte_exact': tsha(reference) == tsha(candidate), 'exact_equal_fraction': float((a == b).sum()) / a.numel(),
              'dtype': str(reference.dtype), 'shape': list(reference.shape)}
    result['PASS'] = finite and max_abs <= result['max_abs_cap'] and l2 <= result['relative_L2_cap'] and max_rel <= result['max_rel_cap']
    differs = a.numpy() != b.numpy()
    result['first_difference_index'] = list(map(int, np.unravel_index(int(np.argmax(differs)), tuple(a.shape)))) if differs.any() else None
    return result


def model_runs(route, index, fresh=False):
    import gc
    torch = runtime()
    check(read(ROOT / f'qualification/{route}_OPERATOR_GATE.json')['status'] == 'PASS', 'operator qualification prerequisite')
    p = read(ROOT / 'QUALIFICATION_POLICY.json')
    doc = p['science']['documents'][1]
    ids = read(doc['token_path'])
    check(isinstance(ids, list) and len(ids) >= 258, 'frozen qualification token stream')
    check(sha(FORMAL) == FORMAL_SHA, 'formal loader identity')
    F = module(FORMAL, 'v3_unchanged_formal_loader')
    F.setup_seed(SEED)
    started = time.monotonic()
    model, tokenizer = F.load_model_and_tokenizer()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
    check(Path(qmod.__file__).resolve() == ARCH.resolve(), 'actual architecture identity')
    layers = model.model.layers if hasattr(model.model, 'layers') else model.model.language_model.layers
    active = [None]; call_number = [0]; captured = {}; handles = []; capture = [False]
    for l in F.GDN_LAYERS:
        m = getattr(layers[l], 'linear_attn', None) or getattr(layers[l], 'self_attn', None)
        handles.append(m.register_forward_pre_hook(lambda _m, _a, l=l: active.__setitem__(0, l)))
    def head(_m, args):
        if capture[0]:
            captured[f'forward/{call_number[0]}/final_hidden'] = args[0].detach().cpu().clone()
    handles.append(model.lm_head.register_forward_pre_hook(head))
    original_chunk = qmod.torch_chunk_gated_delta_rule
    original_recurrent = qmod.torch_recurrent_gated_delta_rule
    def observed(fn):
        def invoke(*args, **kw):
            result, state = fn(*args, **kw)
            if capture[0]:
                captured[f'forward/{call_number[0]}/gdn/{active[0]}'] = result.detach().cpu().clone()
                captured[f'forward/{call_number[0]}/state/{active[0]}'] = state.detach().cpu().clone()
            return result, state
        return invoke
    def one(length, mode):
        captured.clear(); cache = None; call_number[0] = 0; capture[0] = True
        qmod.torch_chunk_gated_delta_rule = observed(chunk_binding(qmod, mode))
        qmod.torch_recurrent_gated_delta_rule = observed(original_recurrent)
        batches = [ids[:length], [ids[length]], [ids[length + 1]]]
        with torch.inference_mode():
            for i, tokens in enumerate(batches):
                call_number[0] = i
                out = model(input_ids=torch.tensor([tokens], device='cuda:0'), past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                captured[f'forward/{i}/logits'] = out.logits.detach().cpu().clone()
                for l in F.GDN_LAYERS:
                    value = F.get_state(cache, l)
                    check(value.dtype == torch.float32, 'no high-precision persistent cache')
                    captured[f'forward/{i}/cache/{l}'] = value.detach().cpu().clone()
        result = dict(captured)
        captured.clear(); capture[0] = False
        del cache, out; gc.collect(); torch.cuda.empty_cache()
        return result
    rows = []; artifacts = []; first_difference = None; first_failure = None
    try:
        for length in p['full_model']['prefill_lengths']:
            rs = resources()
            check(rs['disk_available_bytes'] >= 2 * 1024 ** 3, 'physical disk: paired numerical evidence may require2GiB temporary space')
            # All prefill probes are independently initialized, never a clean
            # reset inside a persistent student scientific trajectory.
            oracle_path = ROOT / f'qualification/full_model/oracle_n{length}.pt'
            if not fresh:
                if oracle_path.exists():
                    old_art = read(ROOT / f'qualification/full_model/oracle_n{length}.json')
                    check(sha(oracle_path) == old_art['sha256'], 'reused same-runtime numerical oracle evidence identity')
                    reference = torch.load(oracle_path, map_location='cpu', weights_only=True)
                else:
                    reference = one(length, 'oracle')
                    oracle_art = pt_save(f'qualification/full_model/oracle_n{length}.pt', reference)
                    save(f'qualification/full_model/oracle_n{length}.json', dict(oracle_art, reference_only=True,
                         persistent_cache_dtype='FP32', no_full_model_FP64_conversion=True,
                         document_sha256=doc['token_sha256'], policy_sha256=sha(ROOT / 'QUALIFICATION_POLICY.json')))
            out = one(length, route)
            artifact = pt_save(f'qualification/full_model/{route}_{index}_n{length}.pt', out)
            artifacts.append(artifact)
            hashes = {k: tsha(v) for k, v in out.items()}
            if not fresh:
                check(set(out) == set(reference), 'paired full-model object coverage')
                comparisons = []
                for name in sorted(reference, key=lambda n: (int(n.split('/')[1]), int(n.split('/')[-1]) if n.split('/')[-1].isdigit() else 100, n)):
                    kind = 'model_fp32' if reference[name].dtype == torch.float32 else 'model_bf16'
                    m = dict(object=name, **compare(reference[name], out[name], kind))
                    comparisons.append(m)
                    if not m['byte_exact'] and first_difference is None:
                        first_difference = {'prefill': length, **m}
                    if not m['PASS'] and first_failure is None:
                        first_failure = {'prefill': length, **m}
                passed = all(x['PASS'] for x in comparisons)
                row = {'length': length, 'PASS': passed, 'comparisons': comparisons, 'hashes': hashes,
                       'artifact': artifact, 'oracle_sha256': sha(oracle_path),
                       'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                       'forward_calls': 6, 'candidate_tokens': length + 2, 'resource_after': resources()}
                del reference
            else:
                base = read(ROOT / f'qualification/full_model/{route}_1.json')
                wanted = next(x for x in base['rows'] if x['length'] == length)['hashes']
                row = {'length': length, 'PASS': wanted == hashes, 'hashes': hashes, 'artifact': artifact,
                       'forward_calls': 3, 'candidate_tokens': length + 2,
                       'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
            rows.append(row)
            del out; gc.collect()
            save(f'qualification/full_model/{route}_{index}_n{length}_comparison.json', row)
            print(json.dumps({'event': 'FULL_MODEL_PREFILL', 'route': route, 'process': index, 'prefill': length,
                              'PASS': row['PASS'], 'peak_reserved_bytes': row['peak_reserved_bytes']}), flush=True)
            if not row['PASS']:
                break
    finally:
        qmod.torch_chunk_gated_delta_rule = original_chunk
        qmod.torch_recurrent_gated_delta_rule = original_recurrent
        for h in handles:
            h.remove()
    passed = len(rows) == 5 and all(x['PASS'] for x in rows)
    save(f'qualification/full_model/{route}_{index}.json', {
        'status': 'PASS' if passed else 'FAIL', 'route': route, 'index': index, 'pid': os.getpid(), 'fresh_repeat': fresh,
        'rows': rows, 'artifacts': artifacts, 'first_difference': first_difference, 'first_failure': first_failure,
        'document_id': doc['canonical_doc_id'], 'document_sha256': doc['token_sha256'],
        'strict': torch.are_deterministic_algorithms_enabled(), 'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
        'TF32': torch.backends.cuda.matmul.allow_tf32, 'elapsed_seconds': time.monotonic() - started,
        'GPU_process_wall_hours': (time.monotonic() - started) / 3600,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'source_commit': cmd(['git', 'rev-parse', 'HEAD'], REPO), 'policy_sha256': sha(ROOT / 'QUALIFICATION_POLICY.json'),
        'GPU_UUID': resources()['GPU_csv'].splitlines()[0].split(',')[1].strip(),
        'model_weight_gradients': False, 'model_generation_calls': 0, 'rotation_training': False})
    return passed


def controller():
    rs = resources()
    check(not rs['processes_csv'], 'do not compete with other GPU workloads')
    check(cmd(['git', 'status', '--porcelain'], REPO) == '', 'start qualification from clean frozen code')
    save('execution/STARTED.json', {'utc': now(), 'source_commit': cmd(['git', 'rev-parse', 'HEAD'], REPO),
                                 'pid': os.getpid(), 'resources': rs, 'policy_sha256': sha(ROOT / 'QUALIFICATION_POLICY.json')})
    status = []; selected = None
    py = sys.executable; path = str(Path(__file__))
    for route in ['fixed', 'compensated']:
        if route == 'compensated':
            check(status and status[0]['status'] == 'NUMERICALLY_INELIGIBLE', 'route2 only after numerical route1 failure')
        for index in [1, 2, 3]:
            subprocess.run([py, path, 'operator', '--route', route, '--index', str(index)], check=True)
        if not operator_gate(route):
            status.append({'route': route, 'status': 'NUMERICALLY_INELIGIBLE', 'gate': 'OPERATOR_FIDELITY_OR_REPEATABILITY'})
            # A repeatability failure, unlike numerical ineligibility, is a
            # global deterministic STOP and must not start the second route.
            check(read(ROOT / f'qualification/{route}_OPERATOR_GATE.json')['fresh_process_bitwise'], 'deterministic scan failure')
            continue
        subprocess.run([py, path, 'full_model', '--route', route, '--index', '1'], check=True)
        outcome = read(ROOT / f'qualification/full_model/{route}_1.json')
        if outcome['status'] != 'PASS':
            status.append({'route': route, 'status': 'NUMERICALLY_INELIGIBLE', 'gate': 'FULL_MODEL_NUMERICAL_ELIGIBILITY',
                           'first_failure': outcome['first_failure']})
            save(f'execution/{route}_FIRST_NUMERICAL_FAILURE.json', status[-1])
            continue
        for index in [2, 3]:
            subprocess.run([py, path, 'full_model', '--route', route, '--index', str(index), '--fresh'], check=True)
            check(read(ROOT / f'qualification/full_model/{route}_{index}.json')['status'] == 'PASS', 'full-model fresh-process deterministic failure')
        status.append({'route': route, 'status': 'PASS'}); selected = route
        break
    save('SCAN_CANDIDATE_QUALIFICATION.json', {'routes': status, 'selected': selected,
         'status': 'PASS' if selected else 'STOP_NO_QUALIFIED_NUMERICAL_PATH', 'maximum_candidates': 2,
         'old_full_model_historical_byte_parity': 'FAIL_PRESERVED', 'new_caps_not_relaxed': True})
    if not selected:
        save('execution/STOP_NO_QUALIFIED_NUMERICAL_PATH.json', {'utc': now(), 'routes': status, 'no_retry': True, 'science_units': 0})
        return
    save('NEW_NUMERICAL_CONTRACT.json', {'name': 'NEW_MATCHED_NUMERICAL_CONTRACT_V1', 'status': 'PASS',
         'selected_route': selected, 'source_commit': cmd(['git', 'rev-parse', 'HEAD'], REPO),
         'source_code_sha256': sha(Path(__file__)), 'policy_sha256': sha(ROOT / 'QUALIFICATION_POLICY.json'),
         'old_historical_parity': 'FAIL_UNCHANGED', 'new_matched_panel_only': True,
         'new_mathematical_fidelity': 'PASS', 'new_full_model_numerical_eligibility': 'PASS',
         'full_model_fresh_processes': 3, 'strict': True, 'warn_only': False, 'TF32': False})
    print('NUMERICAL_CLOSURE_PASS_NEXT_OBSERVER_AND_TRACK_B', selected, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['prepare', 'operator', 'full_model', 'controller'])
    p.add_argument('--route', choices=['fixed', 'compensated'])
    p.add_argument('--index', type=int, default=1)
    p.add_argument('--fresh', action='store_true')
    a = p.parse_args()
    try:
        if a.phase == 'prepare': prepare()
        elif a.phase == 'operator': operator(a.route, a.index)
        elif a.phase == 'full_model': model_runs(a.route, a.index, a.fresh)
        else: controller()
    except Exception as exc:
        name = f'execution/{a.phase}_{a.route or "main"}_{a.index}_ENGINEERING_STOP.json'
        if not (ROOT / name).exists():
            save(name, {'utc': now(), 'status': 'FAIL_AND_STOP', 'message': str(exc), 'type': type(exc).__name__,
                        'traceback': traceback.format_exc(), 'automatic_retry': False})
        raise


if __name__ == '__main__':
    main()
