"""CPU-only original native/reference and read-only shadow qualification."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from stepquant.adapters import RecurrentKernel
from stepquant.calibration import LayerStatistics
from stepquant.quantization import QuantizationPlan, StateCodec, fit_state
from stepquant_calibration_adapter import NativeShadowStatistics

NATIVE_SHA = '90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a'


def original_native(path):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != NATIVE_SHA:
        raise ValueError('Frozen native model source mismatch')
    names = {'l2norm', 'torch_recurrent_gated_delta_rule', 'torch_chunk_gated_delta_rule'}
    nodes = [n for n in ast.parse(path.read_text()).body if getattr(n, 'name', None) in names]
    if {n.name for n in nodes} != names:
        raise ValueError('Missing original native CPU function bodies')
    identities = {}
    for node in nodes:
        identities[node.name] = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        node.decorator_list = []  # Only optional kernel-dispatch decorators; bodies unchanged.
    ns = dict(torch=torch, F=F)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return SimpleNamespace(**ns), identities


def run(path):
    if torch.cuda.is_initialized():
        raise ValueError('CPU qualification entered with initialized CUDA')
    native, identities = original_native(path)
    records = []
    torch.set_num_threads(1)
    for dtype in (torch.float32, torch.bfloat16):
        for length in (9, 64, 128):
            torch.manual_seed(12)
            q, k = [torch.randn(1, length, 2, 8).to(dtype) for _ in range(2)]
            v = torch.randn(1, length, 2, 6).to(dtype)
            g, beta = -torch.rand(1, length, 2), torch.rand(1, length, 2).to(dtype)
            state = torch.randn(1, 2, 8, 6)
            for symbol in ('torch_recurrent_gated_delta_rule', 'torch_chunk_gated_delta_rule'):
                function = getattr(native, symbol)
                kwargs = dict(g=g, beta=beta, initial_state=state, output_final_state=True,
                              use_qk_l2norm_in_kernel=True)
                baseline = function(q, k, v, **kwargs)
                initial_before = state.clone()
                adapter = NativeShadowStatistics(None, native, sample_every=8, max_snapshots=64)
                adapter.statistics['unit'] = LayerStatistics('gdn', 8, 64)
                actual = adapter.wrap(function, 'unit')(q, k, v, **kwargs)
                if not all(torch.equal(a, b) for a, b in zip(baseline, actual)) or not torch.equal(state, initial_before):
                    raise ValueError('Shadow wrapper changed native return or initial state')
                qn, kn = native.l2norm(q), native.l2norm(k)
                observer = LayerStatistics('gdn', 8, 64)
                ref_output, ref_state = RecurrentKernel('gdn', observer)(qn, kn, v, g, beta, state, True, False)
                expected, observed = observer.export(), adapter.statistics['unit'].export()
                if any(not torch.equal(expected[key], observed[key]) for key in ('omega', 'log_decay', 'snapshots')):
                    raise ValueError('Shadow statistics differ from unchanged official reference')
                if dtype == torch.float32 and symbol == 'torch_recurrent_gated_delta_rule':
                    # Exact original tests/test_qwen.py contract. Not widened.
                    torch.testing.assert_close(baseline[0], ref_output, atol=1e-6, rtol=1e-5)
                    torch.testing.assert_close(baseline[1], ref_state, atol=1e-6, rtol=1e-5)
                records.append(dict(dtype=str(dtype), tokens=length, native_kernel=symbol,
                    native_return_exact_with_shadow=True, native_initial_state_unchanged=True,
                    official_shadow_statistics_exact=True, observer_tokens=observed['tokens'],
                    native_vs_shadow_output_max_abs=float((baseline[0].float()-ref_output.float()).abs().max()),
                    native_vs_shadow_final_state_max_abs=float((baseline[1]-ref_state).abs().max()),
                    native_vs_shadow_scope='descriptive only for chunk/BF16; no intermediate byte-equality claim'))
    torch.manual_seed(1)
    bits = torch.tensor([4, 6, 8, 16])[:, None].expand(4, 128).clone()
    impact = torch.rand(4, 128) + .2
    plan = QuantizationPlan('gdn', bits, impact, 32)
    codec = StateCodec(plan)
    state = torch.randn(1, 4, 128, 128)
    fitted = fit_state(state, plan)[-1]
    packed = codec.encode(state)
    decoded = codec.decode(packed)
    torch.testing.assert_close(decoded, fitted, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(decoded[:, bits == 16], state[:, bits == 16].half().float(), atol=0, rtol=0)
    if torch.cuda.is_initialized():
        raise ValueError('CPU qualification initialized CUDA')
    return dict(status='PASS_CPU_CONTRACT_ONLY', native_source_sha256=NATIVE_SHA,
        original_native_AST=identities, records=records,
        official_recurrence_tolerance=dict(atol=1e-6, rtol=1e-5),
        official_codec_tolerance=dict(atol=1e-6, rtol=1e-6), pivots_exact_FP16=True,
        packed_bytes=packed.nbytes, calibration_plan='NOT_GENERATED_BY_TOY_TEST',
        full_9B_calibration='NOT_RUN', full_9B_generation='NOT_RUN', CUDA_initialized=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-source', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.native_source), ensure_ascii=False, indent=2))
