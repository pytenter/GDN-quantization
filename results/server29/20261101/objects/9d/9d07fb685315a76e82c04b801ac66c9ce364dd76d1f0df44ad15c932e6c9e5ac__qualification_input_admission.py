"""CPU-only identity and bounds admission for the actual finite qualification inputs."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ORIGIN = 'history/group32/inherited/group32/inherited/historical_port_payload/candidate/benchmark_replay/fixtures/doc1_tokens.json'
ORIGIN_SHA = '18faa0ca311362181c5c3340d19d72ac39422d1ee07846e516bf34074a51703d'
def validate(root):
    root = Path(root)
    raw = (root / ORIGIN).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ORIGIN_SHA:
        raise RuntimeError('Historical nonAIME replay raw SHA differs')
    original = json.loads(raw)
    original = original if isinstance(original, list) else original.get('tokens', original.get('input_ids'))
    bound = json.loads((root / 'protocol/non_aime_fixture.json').read_text())
    ids = bound.get('tokens')
    short = json.loads((root / 'history/group32/protocol/non_aime_fixture.json').read_text())
    if (set(bound) != {'source_frozen_fixture_raw_sha256', 'tokens'}
            or bound['source_frozen_fixture_raw_sha256'] != ORIGIN_SHA
            or not isinstance(ids, list) or len(ids) != 32768
            or any(type(t) is not int or not 0 <= t < 248320 for t in ids)
            or ids != original or len(short['tokens']) != 64 or ids[:64] != short['tokens']):
        raise RuntimeError('Full historical nonAIME input identity or length differs; no repetition/padding')
    plan = json.loads((root / 'GPU_EXECUTION_PLAN.json').read_text())['qualification_units']
    expected = Counter(gpu_codec_micro=1, group64_persistent_fresh=8,
                       nonAIME_free_generation_fresh=8, nonAIME_throughput=1,
                       real_model_memory_pressure=1)
    node=json.loads((root/'NODE.json').read_text())
    if len(plan) != 19 or len({u['id'] for u in plan}) != 19 or {u['gpu'] for u in plan} != {node['physical_gpu']}:
        raise RuntimeError('Qualification roster differs')
    for gpu in (node['physical_gpu'],):
        rows = [u for u in plan if u['gpu'] == gpu]
        if Counter(u['kind'] for u in rows) != expected:
            raise RuntimeError('Qualification kind counts differ')
        for kind in ('group64_persistent_fresh', 'nonAIME_free_generation_fresh'):
            group = [u for u in rows if u['kind'] == kind]
            if Counter((u['condition'], u['repeat']) for u in group) != Counter((c, r) for c in ('G64-0', 'G64-1', 'G64-2', 'G64-3') for r in (0, 1)):
                raise RuntimeError('Qualification condition/repeat differs')
            for u in group:
                if kind == 'group64_persistent_fresh':
                    if u['tokens'] != 512 or u['checkpoints'] != [0, 31, 127, 511] or len(ids) < u['tokens']:
                        raise RuntimeError('Persistent512 input/checkpoint bounds differ')
                elif u['tokens'] != 64 or u['max_new_tokens'] != 64:
                    raise RuntimeError('Free64 qualification length differs')
        if len(ids) < 384 or len(ids[:16]) != 16:
            raise RuntimeError('Actual benchmark/pressure input bounds differ')
    return dict(status='PASS_FULL_HISTORICAL_REPLAY_AND_NODE19_INPUT_BOUNDS_CPU',
                historical_raw_sha256=ORIGIN_SHA, frozen_teacher_token_count=len(ids),
                persistent_tokens=512, benchmark_tokens=384, pressure_tokens=16,
                qualification_units=19, node_id=node['id'], GPU_calls=0, no_padding_or_repetition=True)
