"""Finite CPU-only additions: Native coordinates, actual Qwen classes, sampler."""
import os, sys, json, hashlib, ast, traceback
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R / 'src'), str(R / 'history/group32/inherited/group32/src'), str(R / 'runtime/transformers-qwen35/src')]
import torch, numpy as np
torch.cuda._lazy_init = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('GPU execution forbidden in CPU phase'))
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from four_codec_adapter import FourCodecQDQ, NativeLayerTracker, CONDITIONS, get_state
from group_codecs import encode, decode, tensor_bytes, require_consumed
from independent_reference import reference
from protocol_binding import sampler
from transformers.models.qwen3_5 import modeling_qwen3_5 as Q
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
assert hashlib.sha256(Path(Q.__file__).read_bytes()).hexdigest() == '90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a'
checks = []

def passed(name, **v):
    checks.append(dict(name=name, status='PASS', **v))

def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and (tensor_bytes(a) == tensor_bytes(b))

def fingerprint(x):
    return hashlib.sha256(tensor_bytes(x)).hexdigest()

def model(dtype):
    torch.random.default_generator.manual_seed(20261020)
    c = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=8, linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4, layer_types=['linear_attention', 'linear_attention', 'full_attention'], rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0, 'partial_rotary_factor': 1.0, 'mrope_section': [2, 1, 1]})
    c._attn_implementation = 'sdpa'
    return Q.Qwen3_5ForCausalLM(c).to(dtype=dtype, device='cpu').eval()

def trace(m, patch=None, qdq=None, capture_bytes=None):
    cache = None
    rows = []
    for (i, t) in enumerate([3, 7, 11, 13, 17, 19, 23, 29]):
        out = m(input_ids=torch.tensor([[t]]), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        if qdq is not None:
            qdq.after_token(cache)
        if capture_bytes is not None:
            capture_bytes.append(tuple([tensor_bytes(out.logits)] + [tensor_bytes(get_state(cache, j)) for j in (0, 1)]))
        assert bool(torch.isfinite(out.logits).all())
        rows.append([fingerprint(out.logits)] + [fingerprint(get_state(cache, j)) for j in (0, 1)])
    return (rows, cache)
real = []
with torch.inference_mode():
    for dtype in (torch.float32, torch.bfloat16):
        native_reference = []
        (expected, cache) = trace(model(dtype), capture_bytes=native_reference)
        real.append(get_state(cache, 0).float().clone())
        original = [Q.torch_recurrent_gated_delta_rule, Q.torch_chunk_gated_delta_rule]
        m = model(dtype)
        with NativeLayerTracker(m):
            assert original == [Q.torch_recurrent_gated_delta_rule, Q.torch_chunk_gated_delta_rule]
            native_actual = []
            (actual, _) = trace(m, capture_bytes=native_actual)
        assert actual == expected and native_actual == native_reference
        passed('NATIVE_UNQUANTIZED_COORDINATES_EXACT_UNTOUCHED_NATIVE', dtype=str(dtype), tokens=8, helper_function_objects_unchanged=True, actual_complete_logits_and_state_bytes_compared=True)
        for condition in CONDITIONS:
            m = model(dtype)
            with FourCodecQDQ(m, condition, (0, 1), observe=True) as p:
                (rows, cache) = trace(m, qdq=p)
                audit = p.audit()
                assert audit['next_consumption_checks'] == 14 and audit['qdq_layer_updates'] == 16
            m = model(dtype)
            with FourCodecQDQ(m, condition, (0, 1), observe=True) as p:
                (other, _) = trace(m, qdq=p)
            assert rows == other
            passed('TINY_REAL_QWEN_PERSISTENT_AND_CPU_REPLAY', condition=condition, dtype=str(dtype), audit=audit, trace=rows)
            m = model(dtype)
            with FourCodecQDQ(m, condition, (0, 1), observe=True) as p:
                out = m(input_ids=torch.tensor([[3]]), use_cache=True)
                p.after_token(out.past_key_values)
                get_state(out.past_key_values, 0).add_(0.03125)
                try:
                    m(input_ids=torch.tensor([[7]]), past_key_values=out.past_key_values, use_cache=True)
                    raise AssertionError('Corrupt writeback not refused')
                except RuntimeError as e:
                    assert 'did not consume' in str(e)
            passed('ACTUAL_NEXT_CONSUMPTION_CORRUPTION_NEGATIVE', condition=condition, dtype=str(dtype))
    g = torch.Generator(device='cpu').manual_seed(20261020)
    mixed = torch.randn((1, 2, 128, 128), generator=g)
    layout = torch.empty_like(mixed)
    for k in range(128):
        for v in range(128):
            layout[:, :, k, v] = (0.0001 + v / 100) * (1 + k // 32) * (1 if k % 2 else -1)
    fixtures = [('mixed', mixed), ('zeros', torch.zeros_like(mixed)), ('positive', mixed.abs()), ('negative', -mixed.abs()), ('signed_zero', torch.full_like(mixed, -0.0)), ('floor', mixed * 1e-15), ('layout', layout)] + [('real_native_' + str(i), x) for (i, x) in enumerate(real)]
    for (name, x) in fixtures:
        for (condition, (_, method)) in CONDITIONS.items():
            e = encode(x, method, group_size=64)
            ref = reference(x.numpy(), method, group_size=64)
            for key in ['codes', 'scales', 'zero_points', 'negative_zero_flags', 'decoded']:
                a = decode(e) if key == 'decoded' else getattr(e, key)
                b = ref[key]
                if a is None:
                    assert b is None
                    continue
                assert a.numpy().dtype == b.dtype and tuple(a.shape) == b.shape and (tensor_bytes(a) == b.tobytes()), (name, condition, key)
            assert e.scales.dtype == torch.float32 and tuple(e.scales.shape) == (x.shape[0], x.shape[1], 2, 128)
            require_consumed(decode(e), e)
            passed('CODEC_EXPLICIT_KEY_INDEX_AND_FIELDS', fixture=name, condition=condition)
    for method in ['B1', 'B2']:
        for val in [float('nan'), float('inf'), -float('inf')]:
            x = torch.zeros((1, 1, 128, 128))
            x[0, 0, 0, 0] = val
            try:
                encode(x, method, group_size=64)
                raise AssertionError('Nonfinite accepted')
            except FloatingPointError:
                pass
        try:
            encode(torch.zeros((1, 1, 128, 128), dtype=torch.bfloat16), method, group_size=64)
            raise AssertionError('Wrong dtype accepted')
        except ValueError:
            pass
        x = layout
        correct = decode(encode(x, method, group_size=64))
        wrong = decode(encode(x.transpose(-2, -1).contiguous(), method, group_size=64)).transpose(-2, -1).contiguous()
        assert not same(correct, wrong)
        passed('NONFINITE_DTYPE_AND_WRONG_VALUE_GROUPING_REFUSED', method=method)
    for condition in ['G64-0', 'G64-1']:
        m = model(torch.float32)
        with FourCodecQDQ(m, condition, (0, 1), observe=True) as p:
            try:
                m(input_ids=torch.tensor([[3, 7]]), use_cache=True)
                raise AssertionError('Chunk prefill accepted')
            except RuntimeError as e:
                assert 'Multi-token' in str(e)
        passed('MULTITOKEN_PREFILL_REFUSED', condition=condition)
    (f, source_rows) = sampler(R, torch)
    warpers = f.build_warpers()
    assert [type(w).__name__ for w in warpers] == ['TemperatureLogitsWarper', 'TopKLogitsWarper', 'TopPLogitsWarper']
    assert (f.TEMPERATURE, f.TOP_P, f.TOP_K, f.MIN_P, f.PRESENCE_PENALTY, f.REPETITION_PENALTY, f.MAX_NEW_TOKENS) == (1.0, 0.95, 20, 0.0, 1.5, 1.0, 81920)
    passed('ORIGINAL_SAMPLER_AST_CONSTANTS_AND_WARPER_ORDER', sources=source_rows, min_p_zero_is_noop=True)
    logits = torch.arange(64, dtype=torch.float32).unsqueeze(0) / 64
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[0, 3] = True
    mask[0, 7] = True
    seen = []

    def spy(ctx, scores):
        assert ctx.shape == (1, 0)
        seen.append(scores.clone())
        return scores
    f.sample_next(spy, logits, mask, torch.Generator().manual_seed(1))
    assert same(seen[0], logits - mask.float() * 1.5) and same(logits, torch.arange(64, dtype=torch.float32).unsqueeze(0) / 64)
    assert seen[0][0, 11] == logits[0, 11]
    passed('PRESENCE_GENERATED_BOOLEAN_ONCE_NO_PROMPT_OR_LOGIT_MUTATION')

    def sequence():
        gen = torch.Generator(device='cpu').manual_seed(1)
        generated = torch.zeros((1, 64), dtype=torch.bool)
        out = []
        for i in range(64):
            (next_token, _) = f.sample_next(warpers, logits, generated, gen)
            out.append(int(next_token))
            generated.scatter_(1, next_token, True)
        return out
    assert sequence() == sequence()
    passed('SAMPLER_FRESH_REQUEST_RNG_AND_MASK_RESET_CPU', tokens=64, CUDA_claim=False)

    class Tok:
        eos_token_id = 2
        unk_token_id = 99

        def convert_tokens_to_ids(self, t):
            return {'<|im_end|>': 2, '<|endoftext|>': 3}[t]
    assert f.stop_ids(Tok()) == {2, 3}
    passed('ORIGINAL_EOS_IDS')

# Group32 negative control uses the original sealed module, not the new reference.
import importlib.util
spec=importlib.util.spec_from_file_location('sealed_legacy_group32_codec',R/'history/group32/inherited/group32/src/group_codecs.py')
legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
legacy_hspec=importlib.util.spec_from_file_location('sealed_legacy_hadamard',R/'history/group32/inherited/group32/src/canonical_h128.py')
legacy_h=importlib.util.module_from_spec(legacy_hspec);legacy_hspec.loader.exec_module(legacy_h)
from canonical_h128 import hadamard
old_h=legacy_h.hadamard(128,device='cpu');new_h=hadamard(128,device='cpu')
assert tensor_bytes(old_h)==tensor_bytes(new_h) and hadamard(128,device='cpu') is new_h
passed('H128_CACHED_ONCE_SAME_OLD_MATRIX_BYTES_AND_NORMALIZATION_POSITION')
more=[]
mid=torch.zeros((1,1,128,128));mid[:,:,0,:]=127
for i,val in enumerate([.5,-.5,1.5,-1.5,2.5,-2.5,-0.,0.]):mid[:,:,i+1,:]=val
more.append(('RNE_midpoints',mid))
zmid=torch.zeros((1,1,128,128));zmid[:,:,0,:]=-.5;zmid[:,:,1,:]=254.5;zmid[:,:,64,:]=-1.5;zmid[:,:,65,:]=253.5
more.append(('ZP_RNE_midpoints',zmid))
more.append(('extreme_finite',mixed*1e35))
for name,x in fixtures+more:
 for method in ['B1','B2']:
  original=legacy.encode(x,method);param=encode(x,method,group_size=32)
  for field in ['codes','scales','zero_points','negative_zero_flags']:
   a=getattr(original,field);b=getattr(param,field)
   assert (a is None and b is None)or(a is not None and b is not None and same(a,b)),(name,method,field)
  assert same(legacy.decode(original),decode(param))
  passed('NEGATIVE_A_GROUP32_ORIGINAL_CODEC_EXACT_FIELDS',fixture=name,method=method)
  e=encode(x,method,group_size=64);expected=reference(x.numpy(),method,group_size=64)
  for field in ['codes','scales','zero_points','negative_zero_flags','decoded']:
   a=decode(e)if field=='decoded'else getattr(e,field);b=expected[field]
   assert(a is None and b is None)or(a is not None and b is not None and tensor_bytes(a)==b.tobytes()),(name,method,field)
  passed('GROUP64_INDEPENDENT_MIDPOINT_EXTREME_OR_REAL_FIXTURE',fixture=name,method=method)
halves=torch.zeros((2,2,128,128))
for bi in range(2):
 for hi in range(2):
  for k in range(128):
   for v in range(128):halves[bi,hi,k,v]=((k+1)*.001 if k<64 else -(k-63)*1000)*(v+1)*(hi+1)*(bi+1)/128
for method in ['B1','B2']:
 e=encode(halves,method,group_size=64)
 assert e.scales.shape==(2,2,2,128)and not torch.equal(e.scales[:,:,0,:],e.scales[:,:,1,:])
 if method=='B2':assert bool((e.zero_points[:,:,0,:]==0).all())and bool((e.zero_points[:,:,1,:]==255).all())
 expected=reference(halves.numpy(),method,group_size=64)
 assert tensor_bytes(e.scales)==expected['scales'].tobytes()
 passed('NEGATIVE_B_TWO_CONTIGUOUS_KEY_GROUPS_DISTINCT_SCALES_ZP_NO_VALUE_HEAD_BATCH_MIX',method=method)
assert not torch.cuda.is_initialized()

result = dict(status='GROUP64_CPU_CODEC_QUALIFIED', scope='ACTUAL_TINY_QWEN_CPU_NOT_9B_NOT_GPU_NOT_81K', checks=checks, count=len(checks), architecture_sha256=hashlib.sha256(Path(Q.__file__).read_bytes()).hexdigest(), GPU_dispatches=0, CUDA_initialization_forbidden=True, source_reuse='Original canonical graphs parameterized only at group extent; Group32 negative-control exact', GPU_qualification_claim=False, source_sha256={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((R/'src').glob('*.py')) if p.name in ['group_codecs.py','four_codec_adapter.py','persistent_adapter.py','canonical_h128.py','independent_reference.py','protocol_binding.py']})
print('RESULT_JSON=' + json.dumps(result, sort_keys=True))
