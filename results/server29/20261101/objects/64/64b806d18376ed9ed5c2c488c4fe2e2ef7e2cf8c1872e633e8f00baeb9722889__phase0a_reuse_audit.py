"""One bounded CPU-only historical-score reuse audit; no model/GPU forward.

All historical files are read-only. New receipts use exclusive creation.
The original reducer is imported from its sealed historical path, unchanged.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

TASK = 'QWEN_GDN_C6R_DUAL_M_INTERACTION_REUSE_5DOC_V1'
OLD_TASK = 'QWEN_GDN_DUAL_AXIS_PROSPECTIVE_REPLICATION_5DOC_V1'
EXPECTED = {
    'source_manifest.json': 'ecc5f10d5c93497769d345dbaed2056974edd92573eb8e679646acfe993fa0df',
    'preregistration.json': 'da2cdb37fc787b84e73d32c51c16e296728b3b9c45875b5658f8c40b93832126',
    'frozen_documents.json': '6b73f4df5171e028bbf0138b65d8cd853038cdebfa8f44127d505c38ade1e167',
    'hashes/SHA256SUMS.txt': '9363b152c991aba56d5696b6db816b48d7940aef63300b24845d7f87b7de2106',
    'hashes/FINAL_ARCHIVE_VERIFICATION.json': 'eaae18358fc353b22aba2c59ea1f413878a8f793affec9c497324713abe8eb66',
    'src/reducer.py': '7e94c95803acb7ae47f0e3ec44fa97faf7cc8d93c4ecc5791a56ea2cf305b84f',
}
POSITIONS = tuple(range(8192, 32768, 64))
CONDITIONS = ('FP_NATIVE', 'H_C128', 'H_DUAL_M')
DOCMAP = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 0}
STEERING = '我希望是使用这个服务器上面的4张GPU一起跑'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(p):
    return json.loads(Path(p).read_text())


def canonical(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def sha(p):
    p = Path(p)
    a = p.stat()
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(16 * 1024**2), b''):
            h.update(b)
    z = p.stat()
    need((a.st_dev, a.st_ino, a.st_size, a.st_mtime_ns) ==
         (z.st_dev, z.st_ino, z.st_size, z.st_mtime_ns), 'File changed during hash: ' + str(p))
    return h.hexdigest()


def write(p, value):
    with Path(p).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def event(stage, **kw):
    print(json.dumps(dict(stage=stage, unix=time.time(), **kw), ensure_ascii=False), flush=True)


def same(actual, expected, message):
    if isinstance(actual, float) and isinstance(expected, float):
        need(actual.hex() == expected.hex(), message)
    else:
        need(actual == expected, message)


def timeout_handler(signum, frame):
    raise TimeoutError('Bounded CPU historical audit exceeded 2700 seconds')


def run(root, historical):
    start = time.time()
    root, old = Path(root).resolve(), Path(historical).resolve()
    need(root != old and not root.is_relative_to(old) and not old.is_relative_to(root), 'Independent new root required')
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only CUDA_VISIBLE_DEVICES must be empty')
    need(not (root / 'HISTORICAL_ASSET_AUDIT.json').exists(), 'Audit already completed; no overwrite/retry')
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(2700)
    event('CPU_AUDIT_STARTED', TASK=TASK, historical_root=str(old), no_GPU=True)
    hashes = {}
    for rel, expected in EXPECTED.items():
        actual = sha(old / rel)
        need(actual == expected, 'Historical root binding mismatch: ' + rel)
        hashes[rel] = actual
    source, pre, docs = (read(old / p) for p in ('source_manifest.json', 'preregistration.json', 'frozen_documents.json'))
    need(source['task'] == pre['TASK_ID'] == OLD_TASK, 'Wrong historical task')
    need(pre['TARGET_N'] == docs['DOCS_FROZEN'] == len(docs['selected_documents']) == 5, 'Full five document contract required')
    need(pre['documents_manifest_sha256'] == hashes['frozen_documents.json'], 'Historical prereg document binding')
    need(pre['score_positions'] == list(POSITIONS), 'Historical 384-position definition changed')
    need(source['KL_reducer_source_sha256'] == hashes['src/reducer.py'], 'Historical reducer binding')
    need(pre['doc_GPU_assignment'] == DOCMAP and set(pre['GPU_UUIDS']) == {'0', '1', '2', '3'}, 'Historical device map changed')
    need(pre['GPU_COUNT_OVERRIDE_HUMAN_MESSAGE'] == STEERING and
         (old / 'HUMAN_4GPU_STEERING.txt').read_text().strip() == STEERING, 'Explicit historical four-card authorization unavailable')
    for a in source['execution_sources']:
        p = old / a['path']
        need(p.resolve().is_relative_to(old) and p.stat().st_size == a['bytes'] and sha(p) == a['sha256'], 'Historical frozen source drift: ' + a['path'])
    need(len(source['execution_sources']) == 104, 'Historical source count changed')
    receipt = read(old / 'hashes/FINAL_ARCHIVE_VERIFICATION.json')
    manifest = {rel: h for h, rel in (x.split('  ', 1) for x in (old / 'hashes/SHA256SUMS.txt').read_text().splitlines())}
    verified = {a['path']: a for a in receipt['verified_assets']}
    need(receipt['status'] == 'PASS' and receipt['all_current_tensors_verified'] is True and
         receipt['manifest_sha256'] == hashes['hashes/SHA256SUMS.txt'] and set(manifest) == set(verified), 'Historical permanent seal mismatch')
    for rel, a in verified.items():
        need(a['sha256'] == manifest[rel] and a['python_stream_SHA'] and a['independent_sha256sum'], 'Historical archive verification missing: ' + rel)
    catalog = read(old / 'hashes/PAYLOAD_CATALOG.json')
    for a in catalog['assets']:
        need(verified[a['path']]['sha256'] == a['sha256'] and verified[a['path']]['bytes'] == a['bytes'], 'Seal bytes mismatch')
    audit_small = ['science_results.json', 'window_results.csv', 'gpu_resource_ledger.json', 'GATE_STATUS_FINAL.json',
                   'analysis/SCORE_ARTIFACT_VERIFICATION.json', 'analysis/POSTPROCESS_STATUS.json', 'FINAL_SCIENTIFIC_REPORT.md',
                   'qualification_cpu/MODEL_CONTENT_IDENTITY.json', 'qualification_cpu/PARENT_IDENTITY_POSTRUN.json',
                   'HUMAN_4GPU_STEERING.txt']
    for rel in audit_small:
        actual = sha(old / rel)
        need(actual == manifest[rel], 'Sealed historical small asset changed: ' + rel)
        hashes[rel] = actual
    model = read(old / 'qualification_cpu/MODEL_CONTENT_IDENTITY.json')
    need(model['status'] == 'PASS' and hashes['qualification_cpu/MODEL_CONTENT_IDENTITY.json'] == source['model_identity_receipt_sha256'], 'Historical model receipt mismatch')
    current_dependencies = []
    for kind in ('model_weight_and_config_current_content_hashes', 'runtime_source_files'):
        for a in model[kind]:
            p = Path(a['path'])
            need(p.stat().st_size == a['bytes'] and sha(p) == a['sha256'], 'Current historical dependency changed: ' + str(p))
            current_dependencies.append(dict(kind=kind, path=str(p), bytes=a['bytes'], sha256=a['sha256'], fresh_stream_hash=True))
        event('CURRENT_DEPENDENCY_GROUP_VERIFIED', kind=kind, count=len(model[kind]), elapsed_seconds=time.time()-start)
    for a in docs['tokenizer']['files']:
        need(sha(Path(docs['tokenizer']['path']) / a['name']) == a['sha256'], 'Tokenizer changed: ' + a['name'])
    gate, ledger = read(old / 'GATE_STATUS_FINAL.json'), read(old / 'gpu_resource_ledger.json')
    need(gate['status'] == 'COMPLETE_AND_STOP' and gate['GPU_release_verified'] and ledger['final_GPU_release_verified'] and all(a['released'] for a in ledger['phases']), 'Historical full completion/release missing')
    need(ledger['GPU_UUIDS'] == pre['GPU_UUIDS'] and len(ledger['phases']) == 23, 'Historical GPU ledger mismatch')
    need(math.isclose(math.fsum(a['GPU_hours'] for a in ledger['phases']), ledger['GPU_HOURS_USED'], rel_tol=0, abs_tol=1e-9), 'Historical summed GPUh mismatch')
    need(source['frozen_unix'] < min(a['dispatch_unix'] for a in ledger['phases']) and pre['frozen_unix'] < min(a['dispatch_unix'] for a in ledger['phases']), 'Historical preregistration not before GPU dispatch')
    need(gate['gates']['cross_gpu_canary_exact'] == 'PASS', 'Historical cross-card canary gate absent')
    original = read(old / 'science_results.json')
    need(original['status'] == 'COMPLETE' and len(original['documents']) == 5, 'Historical science is incomplete')
    score_verification = read(old / 'analysis/SCORE_ARTIFACT_VERIFICATION.json')
    old_assets = {a['phase']: a for a in score_verification['assets']}
    need(score_verification['status'] == 'PASS' and len(old_assets) == 15 and score_verification['source_manifest_sha256'] == hashes['source_manifest.json'], 'Historical score verification incomplete')
    # Read-only import of the original exact source; importing never runs a model.
    import torch
    need(not torch.cuda.is_initialized(), 'CUDA unexpectedly initialized')
    spec = importlib.util.spec_from_file_location('historical_exact_reducer', old / 'src/reducer.py')
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    R.configure_cpu()
    need(tuple(R.SCORE_POSITIONS) == POSITIONS and R.REDUCTION_POLICY_SHA256 == original['reducer_policy_sha256'], 'Reducer policy identity changed')
    dependencies = dict(reducer_source_sha256=hashes['src/reducer.py'], reducer_policy_sha256=R.REDUCTION_POLICY_SHA256,
                        torch_version=torch.__version__, CPU_threads=torch.get_num_threads(), CPU_interop_threads=torch.get_num_interop_threads())
    results, reused, fresh_assets = [], [], []
    for d in docs['selected_documents']:
        i = d['document_index']
        expected_summary = next(x for x in original['documents'] if x['document_index'] == i)
        base = old / f'documents/doc_{i:02d}'
        need(sha(base / 'raw_context.txt') == d['raw_text_sha256'], 'Historical document raw changed')
        vectors = {}
        for name, a in d['artifacts'].items():
            value = read(old / a['path'])
            need(sha(old / a['path']) == a['file_sha256'] and canonical(value) == a['canonical_json_sha256'] and len(value) == a['count'], 'Historical document token artifact changed')
            vectors[name] = value
        ids = vectors['input_ids_32769']
        need(vectors['input_ids_32768'] == ids[:32768] and len(ids) == 32769 and vectors['full_token_ids'][:32769] == ids, 'Historical prefix/label mismatch')
        position_rows = read(base / 'score_positions.json')['positions']
        need(len(position_rows) == 384 and sha(base / 'score_positions.json') == d['score_positions_file_sha256'], 'Historical score positions changed')
        for k, a in enumerate(position_rows):
            t = POSITIONS[k]
            need(a['token_index_zero_based'] == t and a['consumed_token_id'] == ids[t] and a['predicted_next_token_index'] == t+1 and a['target_token_id'] == ids[t+1], 'Historical position semantics mismatch')
        payloads, doc_assets = {}, {}
        for condition in CONDITIONS:
            phase = f'science_doc{i}_{condition}'
            entry = next(x for x in ledger['phases'] if x['phase'] == phase)
            for rel in ['analysis/' + phase + '.json', 'analysis/RECEIPT_' + phase + '.json',
                        'analysis/ENV_' + phase + '.json', 'analysis/PROCESS_' + phase + '.json']:
                need(sha(old / rel) == manifest[rel], 'Historical phase metadata differs from seal: ' + rel)
            result = read(old / ('analysis/' + phase + '.json'))
            rec = read(old / ('analysis/RECEIPT_' + phase + '.json'))
            env = read(old / ('analysis/ENV_' + phase + '.json'))
            card = DOCMAP[str(i)]
            need(entry['GPU_UUID'] == env['UUID'] == pre['GPU_UUIDS'][str(card)] and entry['physical_GPU_ID'] == card and env['CUDA_VISIBLE_DEVICES'] == str(card), 'Historical same-document GPU mismatch')
            need(env['strict'] is True and env['TF32'] is False and env['load']['parameter_dtypes'] == ['torch.bfloat16'], 'Historical deterministic/model dtype contract changed')
            need(result['condition'] == condition and result['doc_id'] == d['document_id'] and result['tag'] == phase and result['consumed_tokens'] == 32768, 'Historical phase analysis metadata mismatch')
            need(rec['status'] == 'PASS' and rec['result']['doc'] == i and rec['result']['condition'] == condition and rec['result']['consumed'] == 32768 and rec['result']['scores'] == 384 and sha(old / ('analysis/RECEIPT_' + phase + '.json')) == entry['receipt_sha256'], 'Historical phase receipt mismatch')
            a = old_assets[phase]
            p = old / a['path']
            actual_sha = sha(p)
            need(actual_sha == a['sha256'] == result['score_file_sha256'] == manifest[a['path']] and p.stat().st_size == a['bytes'] == verified[a['path']]['bytes'], 'Historical actual score SHA/bytes mismatch')
            payload = torch.load(p, map_location='cpu', weights_only=True, mmap=True)
            need(payload['condition'] == condition and payload['doc_id'] == d['document_id'] and payload['consumed_tokens'] == 32768 and payload['input_ids_sha256'] == d['input_ids_32768_canonical_sha256'] and payload['source_manifest_sha256'] == hashes['source_manifest.json'], 'Historical score payload identity mismatch')
            need(payload['score_positions'] == list(POSITIONS) and tuple(payload['logits']) == POSITIONS, 'Historical full score ordering incomplete')
            for t, x in payload['logits'].items():
                need(x.dtype == torch.float32 and x.device.type == 'cpu' and tuple(x.shape) == (1, 1, 248320) and bool(torch.isfinite(x).all()), 'Full reference/student logit vector invalid')
                proof = a['full384_logit_hashes'][(t-8192)//64]
                need(proof['token_index'] == t and proof['FP32_logits_sha256'] == R.tensor_sha256(x), 'Historical per-position saved logit hash mismatch')
            payloads[condition] = payload
            asset = dict(document_index=i, condition=condition, phase=phase, relative_path=a['path'], actual_path=str(p),
                         resolved_path=str(p.resolve()), permanent_archive_path=receipt['permanent_archive_root'] + '/' + a['path'],
                         sha256=actual_sha, bytes=a['bytes'], dtype='torch.float32', shape=[1, 1, 248320], score_count=384,
                         GPU_UUID=entry['GPU_UUID'], physical_GPU_ID=card, source_manifest_sha256=hashes['source_manifest.json'],
                         phase_receipt_sha256=entry['receipt_sha256'], verified_all_logit_hashes=True)
            fresh_assets.append(asset)
            doc_assets[condition] = asset
        identity = expected_summary['identity']
        need(identity['doc_id'] == d['document_id'] and identity['reference_score_file_sha256'] == doc_assets['FP_NATIVE']['sha256'] and identity['input_ids_sha256'] == d['input_ids_32768_canonical_sha256'], 'Historical reducer shared identity mismatch')
        arms = {}
        for condition, arm in [('H_C128', 'A'), ('H_DUAL_M', 'D')]:
            need(sha(old / f'analysis/score_rows_doc{i}_{arm}.json') == manifest[f'analysis/score_rows_doc{i}_{arm}.json'], 'Historical raw score rows differ from seal')
            expected_rows = read(old / f'analysis/score_rows_doc{i}_{arm}.json')
            need(len(expected_rows) == 384, 'Historical raw score rows incomplete')
            rows = []
            for n, t in enumerate(POSITIONS):
                row = R.score_pair(payloads['FP_NATIVE']['logits'][t], payloads[condition]['logits'][t], t, ids[t+1], identity)
                for key, value in row.items():
                    if key != 'scoring_wall_seconds':
                        same(value, expected_rows[n][key], f'Recomputed historical raw score mismatch doc{i}/{condition}/t{t}/{key}')
                row.update(student_condition=condition, student_score_file_sha256=doc_assets[condition]['sha256'],
                           student_phase_receipt_sha256=doc_assets[condition]['phase_receipt_sha256'])
                rows.append(row)
                if (n+1) % 64 == 0:
                    event('ORIGINAL_KL_PROGRESS', document=i, condition=condition, audited_positions=n+1, total=384, elapsed_seconds=time.time()-start)
            arms[arm] = rows
        summary = R.summarize_document(arms['A'], arms['D'])
        for key, value in summary.items():
            if key == 'windows':
                for w, block in value.items():
                    for field, scalar in block.items():
                        same(scalar, expected_summary['windows'][w][field], f'Historical window result mismatch doc{i}/{w}/{field}')
            else:
                same(value, expected_summary[key], f'Historical document result mismatch doc{i}/{key}')
        need(summary['K_A'] == math.fsum(x['KL_FP32'] for x in arms['A'])/384 and summary['K_D'] == math.fsum(x['KL_FP32'] for x in arms['D'])/384, 'Original fsum/384 changed')
        write(root / f'audits/FP_REUSE_RECOMPUTED_doc{i}.json', dict(status='PASS_EXACT_REPRODUCTION', historical_document_index=i,
              original_reducer=dependencies, summary=summary, score_rows=arms, historical_result_SHA256=hashes['science_results.json'], no_GPU=True))
        results.append(dict(document_index=i, status='PASS_EXACT_FLOAT_HEX_AND_HASH_MATCH', K_H_C128=summary['K_A'], K_H_DUAL_M=summary['K_D'],
                            Delta_H=summary['Delta'], Delta_resolution=summary['Delta_resolution'], positions_per_arm=384, independent_scalar_FP64=True))
        reused.append(dict(**d, historical_GPU_UUID=pre['GPU_UUIDS'][str(DOCMAP[str(i)])], historical_physical_GPU_ID=DOCMAP[str(i)],
                           reference_asset=doc_assets['FP_NATIVE'], H_C128_asset=doc_assets['H_C128'], H_DUAL_M_asset=doc_assets['H_DUAL_M']))
        del payloads, vectors, arms
        gc.collect()
        event('HISTORICAL_DOCUMENT_REPRODUCED', document=i, status='PASS_EXACT', elapsed_seconds=time.time()-start)
    need(not torch.cuda.is_initialized(), 'CPU audit initialized CUDA')
    # Recheck source and seal bindings after all read-only reductions.
    for rel, expected in EXPECTED.items():
        need(sha(old / rel) == expected, 'Historical binding drifted during CPU audit: ' + rel)
    audit = dict(TASK_ID=TASK, status='PASS', FP_REFERENCE_REUSE_GATE='PASS', HISTORICAL_REUSE_GATE='PASS',
                 scope='CPU_ONLY_FULL_ORIGINAL_HADAMARD_SCORE_REPRODUCTION', historical_root=str(old),
                 historical_seal=hashes, frozen_source_entries_rehashed=104, current_model_runtime_hashes=current_dependencies,
                 tokenizer_identity=docs['tokenizer'], original_reducer=dependencies, all15_actual_score_assets=fresh_assets,
                 all15_score_SHA_and384_vector_hashes_freshly_verified=True, original_five_document_KL_reproduction=results,
                 arithmetic_comparison='Every original score_pair field except elapsed timing is exact; floats compared via float.hex; all document/window aggregates exact',
                 reference_reuse_format='torch.save dict, logits insertion-ordered dict384 t -> CPU contiguous FP32[1,1,248320], original source/doc/input metadata preserved',
                 reference_reuse_contract='New I/O adapter must bind historical FP source/receipt/GPU/scoreSHA separately from new C6R student source/GPU; do not relabel historical assets or require old/new sources to be equal',
                 source_reference_paths_readonly=True, historical_mutations=False, GPU_work=False, CUDA_initialized=False,
                 PROJECT_INDEPENDENCE='UNKNOWN', sample_analysis_scope='POSTHOC_ON_HISTORICALLY_OBSERVED_DOCUMENTS',
                 current_new_GPU_authorization='ONE_GPU_ONLY_MAX20_GPUH_PROTECTIVE19_5_NO_REUSE_OF_OLD_FOUR_GPU_PERMISSION',
                 old_FP_NATIVE_H_failure='PRESERVED_NOT_RECLASSIFIED', old_physical_packed_INT8='NOT_QUALIFIED',
                 started_unix=start, verified_unix=time.time(), CPU_wall_seconds=time.time()-start,
                 audit_source_sha256=sha(__file__))
    reuse = dict(TASK_ID=TASK, status='PASS_FROZEN_HISTORICAL_FIVE_IN_ORIGINAL_ORDER', analysis_scope='POSTHOC_ON_HISTORICALLY_OBSERVED_DOCUMENTS',
                 DOCS_FROZEN=5, selected_documents=reused, historical_source_manifest_SHA256=hashes['source_manifest.json'],
                 historical_documents_manifest_SHA256=hashes['frozen_documents.json'], historical_preregistration_SHA256=hashes['preregistration.json'],
                 historical_archive_manifest_SHA256=hashes['hashes/SHA256SUMS.txt'], original_order_preserved=True,
                 selection_by_new_or_old_Delta=False, PROJECT_INDEPENDENCE='UNKNOWN', score_positions=list(POSITIONS),
                 position_convention='consume ids[t], logits predict ids[t+1]; retain32769 ids; full32768 persistent student history',
                 FP_NATIVE_reused=True, H_C128_reused=True, H_DUAL_M_reused=True, no_old_FP_or_H_forward_authorized_or_performed=True)
    write(root / 'HISTORICAL_ASSET_AUDIT.json', audit)
    write(root / 'FROZEN_DOCUMENT_REUSE_MANIFEST.json', reuse)
    lines = [
        '# 历史五文档多 GPU 协议审计',
        '原任务附件最初要求一张 GPU；随后人类明确要求“' + STEERING + '”。',
        '该覆盖原文保存在旧根 HUMAN_4GPU_STEERING.txt，SHA256=' + sha(old / 'HUMAN_4GPU_STEERING.txt') + '，并进入原冻结104源清单及最终seal。',
        '原 preregistration 显式记录 GPU_COUNT_OVERRIDE_HUMAN_MESSAGE、四个设备 UUID 和 Doc1/5→GPU0、Doc2→GPU1、Doc3→GPU2、Doc4→GPU3。',
        '本次逐条核查15个正式轨迹 receipt/ENV/ledger，三臂在各自预注册同一物理GPU执行。不存在无证据推定授权或静默追溯改写历史的情况。',
        '原跨GPU128-token canary exact PASS，仅验证该canary范围；未建立32K full-model跨GPU轨迹等价或其数值误差上界。',
        '本新任务明确仅授权一张GPU、总20GPUh、保护停止19.5GPUh；旧四卡许可不延伸到本轮。',
        '本轮新GPU尚未在本审计中选择或执行。若新GPU为历史GPU0且当前数学/软件/runtime身份资格完整，Doc1和Doc5可归A同设备探索层；Doc2/3/4归B跨设备描述层。其他新GPU需按真实UUID逐doc重新分层；身份不全归C，不进入完整交互计算。',
        'A也不表示32K exact repeat或确认性因果交互；B不能与A无区别合并为严格同环境2×2确认。',
        '原五篇Hadamard结果已经被观察，本轮固定原顺序和完整五篇，范围为POSTHOC_ON_HISTORICALLY_OBSERVED_DOCUMENTS；项目独立性UNKNOWN。',
        '原FP_NATIVE/FP_H token1 relative-L2失败、旧BrokenPipe及后续授权完成历史、PHYSICAL_PACKED_INT8未资格均永久保留。',
        '本审计只读历史文件，使用原reducer对全部384×2×5位置执行全248320词表FP32 KL和独立标量FP64复核，逐标量精确复现原结果；未做GPU forward、训练或新研究。',
        '历史根：' + str(old),
        '原最终manifest SHA256：' + hashes['hashes/SHA256SUMS.txt'],
    ]
    with (root / 'HISTORICAL_PROTOCOL_DEVIATION_AUDIT.md').open('x') as f:
        f.write('\n\n'.join(lines) + '\n')
    signal.alarm(0)
    event('CPU_AUDIT_COMPLETE', status='PASS', FP_REFERENCE_REUSE_GATE='PASS', documents=5, all15scores=True, positions_per_codec=1920, CPU_wall_seconds=time.time()-start)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--historical', required=True)
    args = parser.parse_args()
    try:
        run(args.root, args.historical)
    except BaseException as exc:
        failure = dict(TASK_ID=TASK, status='FAIL_CLOSED_STOP', FP_REFERENCE_REUSE_GATE='FAIL',
                       error=repr(exc), traceback=traceback.format_exc(), no_GPU=True, no_automatic_retry=True,
                       historical_mutations=False, failed_unix=time.time())
        target = Path(args.root) / 'audits/PHASE0A_FIRST_FAILURE.json'
        if not target.exists():
            write(target, failure)
        target = Path(args.root) / 'HISTORICAL_ASSET_AUDIT.json'
        if not target.exists():
            write(target, failure)
        event('CPU_AUDIT_FAILED', error=repr(exc), no_GPU=True)
        raise
