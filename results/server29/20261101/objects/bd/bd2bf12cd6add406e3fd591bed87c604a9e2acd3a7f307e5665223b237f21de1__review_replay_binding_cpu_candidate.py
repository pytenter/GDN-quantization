"""Fresh CPU review of exact production input reader expressions and every frozen qualification input bound."""
import ast
import copy
import hashlib
import json
import textwrap
from pathlib import Path

R = Path('/data/zypan/analyses/qwen_gdn_group64_four_codec_aime26_seed1_v1/revisions/native_admission_v2')
D = R / 'operations/REPLAY_BINDING_CPU_CANDIDATE'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
proposal = read(D / 'PROPOSAL.json')
assert sha(R / 'SOURCE_FREEZE.json') == proposal['frozen_source_sha256']
assert sha(R / 'FINAL_SEAL_RECEIPT.json') == proposal['frozen_seal_sha256']
assert sha(R / 'protocol/non_aime_fixture.json') == proposal['prior_frozen_input_sha256']
source = R / proposal['origin_relative_path']
assert sha(source) == proposal['origin_raw_sha256']
raw = read(source)
origin = raw if isinstance(raw, list) else raw.get('tokens', raw.get('input_ids'))
original = read(R / 'protocol/non_aime_fixture.json')
candidate = read(D / 'protocol/non_aime_fixture.json')
assert sha(D / 'protocol/non_aime_fixture.json') == proposal['proposed_input_sha256']

checks = []
def passed(name): checks.append(name)
def validate(value):
    assert set(value) == {'source_frozen_fixture_raw_sha256', 'tokens'}
    assert value['source_frozen_fixture_raw_sha256'] == proposal['origin_raw_sha256']
    ids = value['tokens']
    assert isinstance(ids, list) and len(ids) == 32768
    assert all(type(x) is int and 0 <= x < 248320 for x in ids)
    assert ids == origin
    assert ids[:64] == original['tokens']
    return ids
ids = validate(candidate)
passed('Full32768_exact_historical_bytes_source_and_token_sequence_not_repeated_or_padded')

class CPUReadProxy:
    """Redirect only the candidate input. No torch, model or CUDA mocks are imported."""
    def __init__(self, value): self.value = value; self.paths = []
    def read(self, path):
        assert path == R / 'protocol/non_aime_fixture.json'
        self.paths.append(str(path))
        return copy.deepcopy(self.value)
    @staticmethod
    def require(ok, message):
        if not ok: raise RuntimeError(message)

worker = (R / 'scripts/gpu_unit_worker.py').read_text()
lines = [s for s in worker.splitlines() if s.lstrip().startswith("ids=C.read(R/'protocol/non_aime_fixture.json')")]
assert len(lines) == 1
reader = compile(ast.parse(textwrap.dedent(lines[0])), '<actual-frozen-worker-input-expression>', 'exec')
scope = dict(R=R, C=CPUReadProxy(candidate))
exec(reader, scope)
assert scope['ids'] == ids and len(scope['C'].paths) == 1
passed('Actual_frozen_worker_input_reader_and_atleast512_guard_accept_candidate')
try:
    exec(reader, dict(R=R, C=CPUReadProxy(original)))
except RuntimeError as e:
    assert str(e) == 'NonAIME frozen replay identity changed'
else:
    raise AssertionError('Original64 must reproduce the actual first failure')
passed('Original64_reproduces_actual_preforward_failure_without_any_GPU')

tree = ast.parse((R / 'src/runtime_engine.py').read_text())
engine = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'Engine')
methods = {x.name: x for x in engine.body if isinstance(x, ast.FunctionDef)}
def token_expression(method):
    calls = [x for x in ast.walk(methods[method]) if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr == 'forward']
    assert len(calls) == 1
    return compile(ast.Expression(calls[0].args[0]), '<actual-' + method + '-forward-token-expression>', 'eval')
teacher_token = token_expression('teacher')
benchmark_token = token_expression('benchmark')
benchmark_ids = [x for x in ast.walk(methods['benchmark']) if isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ids' for t in x.targets)]
assert len(benchmark_ids) == 1
scope = dict(self=type('EngineRootOnly', (), dict(root=R))(), C=CPUReadProxy(candidate))
exec(compile(ast.Module(body=benchmark_ids, type_ignores=[]), '<actual-benchmark-input-reader>', 'exec'), scope)
assert scope['ids'] == ids
passed('Actual_frozen_benchmark_input_reader_uses_full_candidate_sequence')
pressure_loops = [x for x in ast.walk(methods['pressure']) if isinstance(x, ast.For) and isinstance(x.target, ast.Name) and x.target.id == 'token']
assert len(pressure_loops) == 1
pressure_tokens = eval(compile(ast.Expression(pressure_loops[0].iter), '<actual-pressure-input-reader>', 'eval'), scope)
assert pressure_tokens == origin[:16] == original['tokens'][:16]
passed('Actual_frozen_pressure_input_first16_remain_identical')

plan = read(R / 'GPU_EXECUTION_PLAN.json')['qualification_units']
assert len(plan) == 38
roster = []
for unit in plan:
    kind = unit['kind']
    if kind == 'group64_persistent_fresh':
        assert unit['tokens'] == 512
        got = [eval(teacher_token, dict(ids=scope['ids'], t=t)) for t in range(unit['tokens'])]
        assert got == origin[:512] and len(got) == 512
        assert unit['checkpoints'] == [0, 31, 127, 511]
        roster.append(dict(unit=unit['id'], consumed=512, actual_source_indexing_checked=True))
    elif kind == 'nonAIME_throughput':
        got = [eval(benchmark_token, dict(ids=scope['ids'], t=t)) for t in range(384)]
        assert got == origin[:384]
        assert any(isinstance(x, ast.For) and isinstance(x.target, ast.Name) and x.target.id == 'condition' and isinstance(x.iter, ast.Tuple) and [e.value for e in x.iter.elts] == ['G64-0','G64-1','G64-2','G64-3'] for x in ast.walk(methods['benchmark']))
        roster.append(dict(unit=unit['id'], consumed_per_condition=384, conditions=4, actual_source_indexing_checked=True))
    elif kind == 'real_model_memory_pressure':
        assert len(pressure_tokens) == 16
        roster.append(dict(unit=unit['id'], consumed_per_condition=16, conditions=4, actual_source_indexing_checked=True))
    else:
        assert kind in ('gpu_codec_micro', 'nonAIME_free_generation_fresh')
        roster.append(dict(unit=unit['id'], teacher_fixture_not_used=True))
assert sum(x['kind'] == 'group64_persistent_fresh' for x in plan) == 16
assert sum(x['kind'] == 'nonAIME_throughput' for x in plan) == 2
assert sum(x['kind'] == 'real_model_memory_pressure' for x in plan) == 2
passed('All38_unit_roster_exact_production_teacher_benchmark_pressure_input_bounds')
try:
    eval(benchmark_token, dict(ids=original['tokens'], t=64))
except IndexError:
    passed('Original64_also_exposes_benchmark_IndexError_at65th_token')
else:
    raise AssertionError('Original benchmark mismatch must remain detectable')

for name, bad in (
    ('Original_truncated64', original),
    ('Wrong_origin_SHA', dict(candidate, source_frozen_fixture_raw_sha256='0'*64)),
    ('Truncated511', dict(candidate, tokens=ids[:511])),
    ('Truncated32767', dict(candidate, tokens=ids[:-1])),
    ('Padded_short64', dict(candidate, tokens=original['tokens']*512)),
    ('Changed_token511', dict(candidate, tokens=ids[:511]+[(ids[511]+1)%248320]+ids[512:])),
    ('Wrong_negative_ID', dict(candidate, tokens=[-1]+ids[1:])),
    ('Wrong_vocab_ID', dict(candidate, tokens=[248320]+ids[1:])),
    ('Bool_not_integer', dict(candidate, tokens=[True]+ids[1:])),
    ('Float_not_integer', dict(candidate, tokens=[float(ids[0])]+ids[1:])),
    ('Unknown_schema', dict(candidate, ignored=True)),
):
    try: validate(bad)
    except (AssertionError, KeyError, TypeError): passed('Reject_' + name)
    else: raise AssertionError('Negative input accepted: ' + name)

freeze = read(R / 'SOURCE_FREEZE.json')
for entry in freeze['files']:
    if entry['path'].startswith(('src/','scripts/','protocol/')):
        p = R / entry['path']
        assert p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256']
passed('All_frozen_src_scripts_protocol_bytes_unchanged_after_CPU_candidate')
assert sha(R / 'SOURCE_FREEZE.json') == proposal['frozen_source_sha256']
assert sha(R / 'FINAL_SEAL_RECEIPT.json') == proposal['frozen_seal_sha256']
passed('Frozen_source_and_seal_bytes_unchanged')
print(json.dumps(dict(status='PASS_CPU_ONLY_REPLAY_BINDING_PROPOSAL',checks=checks,count=len(checks),roster=roster,proposal_sha256=sha(D/'PROPOSAL.json'),production_forward_expressions_tested_with_real_input=True,model_forward_calls=0,GPU_calls=0,scope='CPU validation of real frozen production input reader and indexing expressions. No model/QDQ numerical qualification claim; not an activated source revision or GPU retry.')))
