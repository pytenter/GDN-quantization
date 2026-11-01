"""CPU-only audit of exact archived sampler AST, tokenizer prompts and Frozen V4.

No model construction, forward, GPU initialization or generated-result scoring.
"""
import os
if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError('CPU-only: CUDA_VISIBLE_DEVICES must be empty')
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HISTORY = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen')
RUNNER = HISTORY / 'experiments/qwen_gdn/aime26/81920/run_qwen_aime26_formal.py'
COMMON = RUNNER.with_name('aime26_common.py')
SCORER = HISTORY / 'experiments/shared/scoring/aime26_scorer_v4.py'
DATASET = Path('/data/zypan/worktrees/aime26-sglang-rotation-v1/artifacts/aime26_v1/formal/dataset/aime26_frozen.jsonl')
OLD_UNITS = HISTORY / 'experiments/qwen_gdn/QWEN_4X3090_PRE_AIME_STORAGE_AND_RUNTIME_AUDIT_V1/analysis/aime60_identity_manifest.json'
MODEL = Path('/data/zypan/modelscope_models/Qwen3.5-9B')
FROZEN = {
    RUNNER: '9ff56d00b6b2dbaf9be3c3186bd30be8d3b9cc9d9b580a680316bcc8c01247dd',
    COMMON: '34a320d20a48cda8d0a2a194914f3fa04f8c32bf8c6c533f5f9916fc5d6f9df7',
    SCORER: 'fa5dd904c7d1887df4b8c23613f37b8d438ea494cb72607b89be8e07d55a343b',
    DATASET: '0a744825990f027c6c4455b78776512964bfdbde1e39e8826ad188c6a0d74b9d',
    OLD_UNITS: '087b0b1bf2ac351d270014d9eb7554c5063fe5cb030b59a94b870fec82d4d234',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def exact_ast(path, names, namespace):
    nodes = []
    hashes = {}
    for node in ast.parse(path.read_text()).body:
        assigned = {t.id for t in node.targets if isinstance(t, ast.Name)} if isinstance(node, ast.Assign) else set()
        found = {getattr(node, 'name', '')}.union(assigned).intersection(names)
        if found:
            nodes.append(node)
            for name in found:
                hashes[name] = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    if set(hashes) != set(names):
        raise RuntimeError('Missing original AST nodes: ' + str(set(names)-set(hashes)))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return hashes


def main():
    started = time.time()
    for p, expected in FROZEN.items():
        assert sha(p) == expected, str(p)
    sys.path.insert(0, '/data/zypan/transformers-qwen35/src')
    import torch
    import transformers
    from transformers import AutoTokenizer
    assert not torch.cuda.is_initialized()
    torch.cuda._lazy_init = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('CUDA forbidden'))
    torch.set_num_threads(1)
    namespace = dict(torch=torch)
    constants = {'TEMPERATURE', 'TOP_P', 'TOP_K', 'MIN_P', 'PRESENCE_PENALTY', 'REPETITION_PENALTY', 'MAX_NEW_TOKENS', 'TOP_K_AUDIT_VERSION', 'TOP_K_IMPLEMENTATION', 'LEGACY_TOP_K_AUDIT_COMMIT'}
    funcs = {'build_warpers', 'sample_next', 'build_top_k_audit', 'top_k_audit_valid', 'stop_ids', 'render_input'}
    common_ast = exact_ast(COMMON, {'raw_messages', 'ids_sha256', 'load_frozen_dataset'}, dict_namespace := dict(json=json, Path=Path, hashlib=hashlib))
    namespace['raw_messages'] = dict_namespace['raw_messages']
    runner_ast = exact_ast(RUNNER, constants | funcs, namespace)
    assert [namespace[n] for n in ['TEMPERATURE', 'TOP_P', 'TOP_K', 'MIN_P', 'PRESENCE_PENALTY', 'REPETITION_PENALTY', 'MAX_NEW_TOKENS']] == [1.0, .95, 20, 0., 1.5, 1., 81920]
    warpers = namespace['build_warpers']()
    logits = torch.zeros((1, 64), dtype=torch.float32)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    def stream(seed):
        gen = torch.Generator(device='cpu').manual_seed(seed)
        return [int(namespace['sample_next'](warpers, logits, mask, gen)[0].item()) for _ in range(64)]
    s1, replay, s2 = stream(1), stream(1), stream(2)
    assert s1 == replay and s1 != s2
    tied = torch.cat((torch.ones(25), torch.zeros(39))).reshape(1, -1)
    topk_scores = warpers[1](torch.empty((1, 0), dtype=torch.long), tied)
    assert int(torch.isfinite(topk_scores).sum()) == 25
    _, after_top_p = namespace['sample_next'](warpers, tied, mask, torch.Generator().manual_seed(1))
    tie_audit = namespace['build_top_k_audit']([after_top_p])
    assert after_top_p > 20 and tie_audit['tie_expansion_observed']
    probe = torch.linspace(-1, 1, 64).reshape(1, -1)
    output_mask = mask.clone(); output_mask[0, [3, 4, 5]] = True
    expected_scores = probe.float() - output_mask.float() * 1.5
    expected_probs = torch.softmax(warpers(torch.empty((1,0), dtype=torch.long), expected_scores), dim=-1)
    expected_token = torch.multinomial(expected_probs, 1, generator=torch.Generator().manual_seed(7))
    actual_token, _ = namespace['sample_next'](warpers, probe, output_mask, torch.Generator().manual_seed(7))
    assert torch.equal(actual_token, expected_token)
    # The unmodified generate_one body starts an empty output-only boolean mask.
    generate = next(n for n in ast.parse(RUNNER.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'generate_one')
    generate_source = ast.get_source_segment(RUNNER.read_text(), generate)
    assert 'generated_token_mask = torch.zeros_like' in generate_source
    assert 'generated_token_mask.scatter_(1, current, True)' in generate_source
    assert 'torch.Generator(device=device).manual_seed(seed)' in generate_source

    spec = importlib.util.spec_from_file_location('exact_frozen_aime26_v4', SCORER)
    scorer = importlib.util.module_from_spec(spec); sys.modules[spec.name] = scorer; spec.loader.exec_module(scorer)
    fixtures = [
        ('explicit_final', 'Final answer is 123.', 'eos', '', '123', 123, True),
        ('rightmost_restatement', 'Final answer is 123. This is incorrect. Final answer is 456.', 'eos', '', '456', 456, True),
        ('tentative', 'Maybe the answer is 123.', 'eos', '', '123', None, False),
        ('prompt_restatement', 'The final result is 123.', 'eos', 'The final result is 123.', '123', None, False),
        ('domain_rejection', 'Final answer is 1000.', 'eos', '', '1000', 1000, False),
        ('exact_arithmetic', r'\boxed{21+21}', 'eos', '', '42', 42, True),
        ('truncated_weak', 'Thus the value is 123.\nWe continue to check.', 'length', '', '123', None, False),
        ('truncated_explicit_final', 'Final Answer: 123', 'length', '', '123', 123, True),
        ('no_global_last_number', 'Reasoning includes 17 and 234.', 'eos', '', '234', None, False),
        ('retracted_box', r'\boxed{123}. That is wrong. Final answer is 456.', 'eos', '', '456', 456, True),
        ('tentative_box', r'Perhaps \boxed{123}.', 'eos', '', '123', None, False),
    ]
    tested = []
    for name, text, termination, problem, gold, expected_value, expected_correct in fixtures:
        result = scorer.extract_v4(text, termination, problem)
        correct = scorer.compare_with_gold(result, gold)
        assert result.normalized_value == expected_value and correct == expected_correct, (name, result.to_dict(), correct)
        tested.append(dict(name=name, status='PASS', extracted=result.to_dict(), correct=correct))

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True, local_files_only=True)
    dataset = dict_namespace['load_frozen_dataset'](DATASET)
    old = {(x['problem_id'], x['seed']):x for x in json.loads(OLD_UNITS.read_text())}
    prompt_rows = []
    for item in dataset:
        messages, ids = namespace['render_input'](tokenizer, item['problem'])
        assert messages == [{'role':'user', 'content':item['problem']}]
        token_sha = dict_namespace['ids_sha256'](ids[0].tolist())
        assert all(old[(item['problem_id'], seed)]['input_ids_hash'] == token_sha for seed in (1,2)), item['problem_id']
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        problem_sha = hashlib.sha256(item['problem'].encode()).hexdigest()
        assert all(old[(item['problem_id'], seed)]['problem_sha256'] == problem_sha for seed in (1,2))
        prompt_rows.append(dict(problem_id=item['problem_id'], problem_idx=item['problem_idx'], problem_SHA256=problem_sha,
            input_ids_hash=token_sha, input_ids=ids[0].tolist(), input_tokens=int(ids.shape[-1]), prompt_text_SHA256=hashlib.sha256(prompt.encode()).hexdigest()))
    stop_ids = sorted(namespace['stop_ids'](tokenizer))
    assert not torch.cuda.is_initialized()
    blobs = {}
    for rel in ['experiments/qwen_gdn/aime26/81920/run_qwen_aime26_formal.py','experiments/shared/scoring/aime26_scorer_v4.py']:
        value = subprocess.check_output(['git','show','a9d69b4609f19acb5a2d42aa2ec31bf704bd1fb3:'+rel], cwd=HISTORY)
        blobs[rel] = hashlib.sha256(value).hexdigest()
        assert blobs[rel] == sha(HISTORY/rel)
    result = dict(status='PASS_CPU_SAMPLER_SCORER_AND_30_PROMPT_IDENTITY_ONLY', source_sha256=sha(Path(__file__)),
        exact_source_files={str(p):h for p,h in FROZEN.items()}, archived_commit='a9d69b4609f19acb5a2d42aa2ec31bf704bd1fb3', git_blob_SHA_matches=blobs,
        AST_sha256=dict(runner=runner_ast, common=common_ast),
        sampling_tests=dict(seed1_exact_replay=True, seed1_vs_seed2_differ=True, sampled_tokens_not_greedy=True,
            CPU_generator_only=True, CUDA_generator_execution='NOT_TESTED', top_k_equal_cutoff_survivors=25,
            top_k_tie_after_top_p_survivors=after_top_p, top_k_audit=tie_audit,
            presence_penalty_applied_once_to_boolean_generated_output_mask=True, prompt_tokens_not_added_to_presence_mask=True,
            min_p0_and_repetition1_no_extra_warper=True, max_new_tokens=81920),
        FrozenV4=dict(name=scorer.SCORER_NAME, source_literal_status=scorer.STATUS, source_SHA256=sha(SCORER),
            formal_freeze_uses_this_exact_candidate_named_source=True, fixtures=tested, no_generated_result_scored=True),
        tokenizer_class=type(tokenizer).__name__, tokenizer_stop_ids=stop_ids, tokenizer_eos_token_id=tokenizer.eos_token_id,
        token_strings={x:tokenizer.convert_tokens_to_ids(x) for x in ['<|im_end|>','<|endoftext|>']},
        actual_loaded_chat_template_SHA256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        chat_template_file_SHA256=sha(MODEL/'chat_template.jinja'),
        all30_prompt_token_SHA_match_archived60_identity=True, prompt_rows=prompt_rows,
        torch=str(torch.__version__), transformers=str(transformers.__version__), CUDA_initialized=False,
        model_loaded=False, model_forward_count=0, elapsed_seconds=time.time()-started,
        scope_limits=['CPU artificial logits only; CUDA sampling parity remains a separate GPU gate',
            '81920 constant verified; no long-context resource qualification is inferred',
            'Original runner internal score_aime is legacy, not FrozenV4, and is not used here',
            'Historical runner prefill_quantized=false; new common prefill timing requires explicit new qualification'])
    out = ROOT/'protocol_recovery/CPU_PROTOCOL_QUALIFICATION.json'
    put(out, result)
    print(json.dumps(dict(status=result['status'],output=str(out),sha256=sha(out),fixtures=len(tested),prompts=len(prompt_rows),CUDA_initialized=False)))


if __name__ == '__main__':
    main()
