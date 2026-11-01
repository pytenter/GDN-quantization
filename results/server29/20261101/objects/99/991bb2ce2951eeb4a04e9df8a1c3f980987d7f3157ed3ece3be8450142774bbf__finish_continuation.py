"""One CPU closeout of the explicitly requested continuation, no GPU entry."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
TASK = 'QWEN_GDN_C6R_DUAL_M_INTERACTION_REUSE_5DOC_V1'
ARCHIVE = '/data/zypan/qwen_c6r_dual_m_interaction_archive/CONTINUATION_20261017/' + TASK
SSH = ['ssh', '-S', '/tmp/qwen_replication_archive29.sock', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'zypan@202.38.247.29']
CP_SHA = '0d9c2e22a634b4327c03e34343087fd3d796276035f653b4b22ead3d8cf442ce'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def need(v, msg):
    if not v:
        raise RuntimeError(msg)


def save(rel, obj):
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def validate_history(actual, condition, identity, asset, doc):
    need(actual['condition'] == condition and actual['doc_id'] == identity['doc_id'] and
         actual['consumed_tokens'] == 32768 and actual['score_file_sha256'] == asset['sha256'],
         'Actual completed C6R history/score binding mismatch')
    if doc == 5 and condition == 'C6R_C128':
        phase = 'science_doc5_C6R_C128'
        restore = read(ROOT / ('analysis/RESTORE_' + phase + '.json'))
        c = actual['continuation']
        need(restore['status'] == 'PASS' and restore['checkpoint_sha256'] == CP_SHA and
             restore['cursor'] == 16384 and restore['first_suffix_token_QDQ_previous_seeded'] is True,
             'Exact restored boundary qualification missing')
        need(actual['resume_cursor'] == 16384 and actual['new_model_tokens'] == 16384 and
             actual['prefix_score_count'] == 128 and actual['new_score_count'] == 256 and
             c['checkpoint_sha256'] == CP_SHA and c['prefix_diagnostics'] == 'NOT_RETAINED',
             'Prefix/suffix lineage or missing-diagnostic declaration mismatch')
        need(actual['writebacks'] == actual['rotation_calls'] == actual['next_token_consumption_checks'] == 16384 * 24,
             'Suffix real QDQ/rotation/boundary call count mismatch')
        need([v['t'] for v in actual['performance']] == list(range(16384, 32768)),
             'Continuation reprocessed or omitted formal tokens')
        need([v['token_index'] for v in actual['mechanism']] == list(range(16384, 32768, 64)),
             'Continuation mechanism positions mismatch')
        prefix = read(ROOT / 'provenance/READONLY_DOC5_CHECKPOINT_RECOVERY_AUDIT_20261017.json')
        need(prefix['status'] == 'PASS' and prefix['sha256'] == CP_SHA and prefix['cache_cursor'] == 16384 and
             prefix['completed_score_count'] == 128 and prefix['physical_GPU_UUID'] == identity['GPU_UUID'] and
             prefix['source_manifest_SHA'] == identity['source_manifest_sha256'],
             'Inherited formal-prefix identity contract failed')
    else:
        need(actual['writebacks'] == actual['rotation_calls'] == 32768 * 24 and
             actual['next_token_consumption_checks'] == 32767 * 24,
             'Uninterrupted full history call count failed')


def annotate_runtime(row, actual, doc, condition):
    row['measurement_scope'] = 'FULL_UNINTERRUPTED_TRAJECTORY'
    if doc == 5 and condition == 'C6R_C128':
        row.update(measurement_scope='RESTORED_SUFFIX_ONLY_16384_TO_32768',
                   prefix_per_token_performance='NOT_RETAINED', prefix_mechanism='NOT_RETAINED',
                   prefix_allocator_peak='NOT_RETAINED',
                   logical_full_history_tokens=32768, newly_executed_tokens=16384,
                   runtime_wall_field_scope='suffix loop/checkpoint/score I/O only; excludes model load and restore, which are included by dispatch-to-release GPU ledger')
    return row


def validate_receipt_result(receipt, condition, doc):
    result = receipt['result']
    base = dict(status='PASS', condition=condition, doc=doc, consumed=32768, scores=384)
    need(all(result.get(k) == v for k, v in base.items()), 'Completed trajectory receipt base mismatch')
    if doc != 5:
        need(result == base, 'Inherited receipt unexpectedly changed')
    else:
        resumed = condition == 'C6R_C128'
        need(result['resume_cursor'] == (16384 if resumed else 0) and
             result['new_model_tokens'] == (16384 if resumed else 32768) and
             result['prefix_score_count'] == (128 if resumed else 0) and
             result['new_score_count'] == (256 if resumed else 384) and
             result['new_writebacks'] == result['rotation_calls'] == (16384 if resumed else 32768) * 24 and
             result['new_next_token_consumption_checks'] == (16384 if resumed else 32767) * 24,
             'Continuation receipt actual counter mismatch')


def analyze():
    path = ROOT / 'analysis_scripts/analyze_interaction.py'
    need(sha(path) == '762892993aad13158e531aaef94875ec3c5dc24165840a5ec63bccbc9eeb8d53', 'Original CPU analyzer changed')
    spec = importlib.util.spec_from_file_location('continuation_frozen_analyzer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tree = ast.parse(path.read_text())
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    candidate = copy.deepcopy(original)
    edits = []
    for node in ast.walk(candidate):
        if isinstance(node, ast.Constant) and node.value == '/tmp/qwen_c6r_interaction_cpu_score_cache_20261016':
            old = copy.deepcopy(node)
            node.value = str(ROOT / 'analysis_scratch')
            edits.append((node, old, 'task-local scratch path only'))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'need' and len(node.args) == 2 and isinstance(node.args[1], ast.Constant) and node.args[1].value == 'Actual persistent C6R history/completion/score binding failed':
            old = copy.deepcopy(node)
            replacement = ast.parse('validate_history(actual, condition, identity, asset, doc)', mode='eval').body
            node.__dict__.update(replacement.__dict__)
            edits.append((node, old, 'exact inherited-prefix plus actual suffix history contract'))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'need' and len(node.args) == 2 and isinstance(node.args[1], ast.Constant) and node.args[1].value == 'New completed trajectory receipt mismatch':
            old = copy.deepcopy(node)
            replacement = ast.parse('validate_receipt_result(receipt, condition, doc)', mode='eval').body
            node.__dict__.update(replacement.__dict__)
            edits.append((node, old, 'preserved base receipt plus explicit continuation actual counters'))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'runtime' and node.func.attr == 'append':
            old = copy.deepcopy(node)
            need(len(node.args) == 1, 'Unexpected original runtime append')
            node.args = [ast.Call(func=ast.Name(id='annotate_runtime', ctx=ast.Load()), args=[node.args[0], ast.Name(id='actual', ctx=ast.Load()), ast.Name(id='doc', ctx=ast.Load()), ast.Name(id='condition', ctx=ast.Load())], keywords=[])]
            edits.append((node, old, 'label missing prefix diagnostics; no invented metrics'))
    need(len(edits) == 4, 'Unexpected CPU adapter edit count')
    candidate_dump = ast.dump(candidate, include_attributes=False)
    restored = copy.deepcopy(candidate)
    # A path-index independent inverse uses the exact four mutated object nodes.
    for node, old, _ in reversed(edits):
        node.__dict__.clear()
        node.__dict__.update(copy.deepcopy(old.__dict__))
    need(ast.dump(candidate, include_attributes=False) == ast.dump(original, include_attributes=False), 'CPU adapter inverse AST differs')
    save('analysis/CONTINUATION_CPU_ADAPTER_AUDIT.json', dict(status='PASS', original_SHA256=sha(path),
         original_AST_SHA256=hashlib.sha256(ast.dump(original, include_attributes=False).encode()).hexdigest(),
         candidate_AST_SHA256=hashlib.sha256(candidate_dump.encode()).hexdigest(), inverse_AST_exact=True,
         changes=[v[2] for v in edits], original_KL_reducer_and_all_statistical_arithmetic_unchanged=True))
    module.__dict__.update(validate_history=validate_history, validate_receipt_result=validate_receipt_result, annotate_runtime=annotate_runtime)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[restored], type_ignores=[])), str(path), 'exec'), module.__dict__)
    module.main()


VERIFY_SOURCE = r'''import hashlib,json,pathlib,subprocess,sys,time
r=pathlib.Path(sys.argv[1]);expected=json.loads((r/'hashes/DELIVERY_PAYLOAD_EXPECTED.json').read_text());refs=json.loads((r/'hashes/EXTERNAL_ARCHIVED_ASSETS.json').read_text())['assets']
def need(v,s):
 if not v:raise RuntimeError(s)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(16777216),b''):h.update(b)
 return h.hexdigest()
def check(a,p):
 need(p.is_file() and p.stat().st_size==a['bytes'],'Archive bytes '+str(p));before=p.stat();h=sha(p);s=subprocess.check_output(['sha256sum','--',str(p)],text=True).split()[0];after=p.stat();need(h==s==a['sha256'] and (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Archive doubleSHA '+str(p));return dict(path=str(p),sha256=h,bytes=after.st_size,python_stream_SHA=True,independent_sha256sum=True)
need(not (r/'hashes/FINAL_ARCHIVE_VERIFICATION.json').exists(),'Existing final seal; no overwrite')
need(sha(r/'hashes/DELIVERY_PAYLOAD_EXPECTED.json')==sys.argv[2],'Expected payload document content mismatch')
for a in expected['assets']:check(a,r/a['path'])
checked=[];current=[]
for p in sorted(r.rglob('*')):
 if not p.is_file() or p.is_symlink():continue
 rel=str(p.relative_to(r));need(not p.name.endswith(('.partial','.tmp')),'Unfinished archive file '+rel)
 if rel in ['SHA256_MANIFEST.json','hashes/SHA256SUMS.txt','hashes/FINAL_ARCHIVE_VERIFICATION.json']:continue
 a=dict(path=rel,sha256=sha(p),bytes=p.stat().st_size);checked.append(check(a,p));current.append(a)
external=[check(a,pathlib.Path(a['archive_path'])) for a in refs]
manifest=dict(TASK_ID=r.name,status='PASS',assets=current,external_archived_assets=refs,all_current_tensors_verified=True,verified_unix=time.time())
with (r/'SHA256_MANIFEST.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
with (r/'hashes/SHA256SUMS.txt').open('x') as f:
 for a in current:f.write(a['sha256']+'  '+a['path']+'\n')
 f.write(sha(r/'SHA256_MANIFEST.json')+'  SHA256_MANIFEST.json\n')
out=dict(status='PASS',archive_root=str(r),verified_asset_count=len(checked),verified_external_assets=len(external),all_current_tensors_verified=True,current_payload=checked,referenced_external_assets=external,SHA256_MANIFEST_SHA256=sha(r/'SHA256_MANIFEST.json'),SHA256SUMS_SHA256=sha(r/'hashes/SHA256SUMS.txt'),actual_report_sha256=sha(r/'FINAL_SCIENTIFIC_REPORT.md'),no_GPU=True)
with (r/'hashes/FINAL_ARCHIVE_VERIFICATION.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(dict(status='PASS',verified_asset_count=len(checked),verified_external_assets=len(external),report_SHA256=out['actual_report_sha256'])))
'''


def seal():
    refs = {}
    for path in (ROOT / 'archive').glob('DOC_*_CURRENT_ASSET_LOCATIONS.json'):
        for a in read(path)['assets']:
            refs[a['archive_path']] = dict(archive_path=a['archive_path'], sha256=a['sha256'], bytes=a['bytes'])
    for a in read(ROOT / 'provenance/HISTORICAL_SCORE_PATH_RESTORATION.json')['assets']:
        refs[a['archive_path']] = dict(archive_path=a['archive_path'], sha256=a['sha256'], bytes=a['bytes'])
    cp = read(ROOT / 'provenance/INDEPENDENT_ROOT_PREPARATION.json')['checkpoint']
    refs[cp['archive_path']] = dict(archive_path=cp['archive_path'], sha256=cp['sha256'], bytes=cp['bytes'])
    save('hashes/EXTERNAL_ARCHIVED_ASSETS.json', dict(status='PASS', assets=list(refs.values()), original_archives_unchanged=True))
    with (ROOT / 'hashes/CONTINUATION_FINAL_ARCHIVE_VERIFY_SOURCE.py').open('x') as f:
        f.write(VERIFY_SOURCE)
    assets = []
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts:
            continue
        rel = p.relative_to(ROOT)
        if rel.parts[0] in ['inputs', 'logs'] or p.suffix in ['.tmp', '.lock', '.pyc', '.partial']:
            continue
        need(p.stat().st_uid == os.getuid(), 'Unexpected delivery owner')
        assets.append(dict(path=str(rel), bytes=p.stat().st_size, sha256=sha(p)))
    save('hashes/DELIVERY_PAYLOAD_EXPECTED.json', dict(status='PASS', assets=assets, live_exclusions=['logs/**', 'inputs/**'], large_inputs_already_permanently_archived=True))
    extra = ROOT / 'hashes/DELIVERY_PAYLOAD_EXPECTED.json'
    expected_sha = sha(extra)
    paths = [a['path'] for a in assets] + ['hashes/DELIVERY_PAYLOAD_EXPECTED.json']
    listing = ROOT / 'hashes/DELIVERY_COPY_PATHS.txt'
    with listing.open('x') as f:
        f.write('\n'.join(paths) + '\n')
    subprocess.run(SSH + [shlex.join(['mkdir', '-p', ARCHIVE])], check=True, timeout=30)
    subprocess.run(['rsync', '-aL', '--ignore-existing', '--files-from=' + str(listing), '-e', shlex.join(SSH[:-1]), str(ROOT) + '/', SSH[-1] + ':' + ARCHIVE + '/'], check=True, timeout=1200)
    out = subprocess.run(SSH + [shlex.join(['/usr/bin/python3', '-c', VERIFY_SOURCE, ARCHIVE, expected_sha])], check=True, capture_output=True, text=True, timeout=1800)
    print(out.stdout, flush=True)
    for rel in ['SHA256_MANIFEST.json', 'hashes/SHA256SUMS.txt', 'hashes/FINAL_ARCHIVE_VERIFICATION.json']:
        subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ControlPath=/tmp/qwen_replication_archive29.sock', SSH[-1] + ':' + ARCHIVE + '/' + rel, str(ROOT / rel)], check=True, timeout=120)
    v = read(ROOT / 'hashes/FINAL_ARCHIVE_VERIFICATION.json')
    need(v['status'] == 'PASS' and v['actual_report_sha256'] == sha(ROOT / 'FINAL_SCIENTIFIC_REPORT.md'), 'Final report verification mismatch')
    # Verify exact local small-file coverage against the sealed current payload.
    need(sha(extra) == expected_sha, 'Expected metadata drift')
    save('DELIVERY_READY.json', dict(status='VERIFIED_COMPLETE_OR_REAL_STOP', report_sha256=v['actual_report_sha256'], archive_root=ARCHIVE,
         verification_sha256=sha(ROOT / 'hashes/FINAL_ARCHIVE_VERIFICATION.json'), no_GPU=True, no_next_experiment=True))


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    final = read(ROOT / 'FINAL_GPU_STATUS.json')
    need(final['GPU_release_verified'] and final['no_further_GPU_work'], 'GPU release boundary not verified')
    started = read(ROOT / 'COORDINATOR_STARTED.json')
    need(sha(ROOT / 'CONTINUATION_SOURCE_MANIFEST.json') == started['hashes']['CONTINUATION_SOURCE_MANIFEST.json'], 'Continuation source roster drifted')
    cm = read(ROOT / 'CONTINUATION_SOURCE_MANIFEST.json')
    for a in cm['execution_sources']:
        p = ROOT / a['path']
        need(p.is_file() and p.stat().st_size == a['bytes'] and sha(p) == a['sha256'], 'Frozen continuation source changed ' + str(p))
    m = read(ROOT / 'SOURCE_MANIFEST.json')
    checked = []
    for a in m['execution_sources'] + m['external_dependencies']:
        p = Path(a['path']) if Path(a['path']).is_absolute() else ROOT / a['path']
        need(p.is_file() and p.stat().st_size == a['bytes'] and sha(p) == a['sha256'], 'Frozen scientific dependency changed ' + str(p))
        checked.append(dict(path=str(p), sha256=a['sha256'], bytes=a['bytes']))
    save('analysis/POSTRUN_SOURCE_VERIFICATION.json', dict(status='PASS', actual_stream_hashed_dependencies=checked, no_GPU=True))
    analyze()
    subprocess.run([sys.executable, str(ROOT / 'analysis_scripts/final_report.py')], check=True, timeout=120)
    note = '\n\n## 人工请求后的断点续跑与诊断限制\n\n'
    note += '本报告属于清理存储后、用户请求继续的独立续跑交付。原FAIL_CLOSED_STOP、失败回执和先前报告均保存在provenance/original，未覆盖或抹除。累计GPU预算包含原14.467484712157813 GPUh，未重置。\n\n'
    note += 'Doc5 C6R_C128从已核验的同条件、同文档、同GPU0 checkpoint cursor16384恢复；前128个全词表评分沿用该checkpoint，后256点评分由续跑生成，不重跑该已提交16K前缀。原失败进程在断点之后可能执行但未提交的计算量未知，其全部实际占用仍已计费。Doc5 Dual-M独立从token0执行。\n\n'
    note += 'checkpoint未保存C128前16384个token的逐token性能、机制统计或峰值；这些诊断为NOT_RETAINED。Doc5 C128表中吞吐、wall和峰值仅覆盖恢复后的suffix，不能解释为整条32K测量；全任务资源成本以含原失败reservation的累计GPU账本为准。Primary与三窗口KL使用完整384点评分，机制诊断缺失不能补造。\n'
    with (ROOT / 'FINAL_SCIENTIFIC_REPORT.md').open('a') as f:
        f.write(note)
    seal()


if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        traceback.print_exc()
        p = ROOT / 'analysis/CONTINUATION_DELIVERY_STOP.json'
        if not p.exists():
            save(str(p.relative_to(ROOT)), dict(status='FAIL_CLOSED_STOP', error=repr(e), traceback=traceback.format_exc(), no_GPU_retry=True, original_seals_preserved=True))
        if not (ROOT / 'FIRST_FAILURE.json').exists():
            save('FIRST_FAILURE.json', read(p))
        if not (ROOT / 'FINAL_SCIENTIFIC_REPORT.md').exists():
            try:
                subprocess.run([sys.executable, str(ROOT / 'analysis_scripts/final_report.py')], check=True, timeout=120)
            except BaseException:
                q = '\n'.join(str(i) + '. 本项完整合法证据未完成独立交付核验，不作科学结论。' for i in range(1, 14))
                with (ROOT / 'FINAL_SCIENTIFIC_REPORT.md').open('x') as f:
                    f.write('# 断点续跑真实停止交付\n\nFINAL_EXECUTION_STATUS=FAIL_CLOSED_STOP\n\n' + repr(e) + '\n\n原STOP和封存保持不变；已完成资产与真实部分进展保留，未追加GPU重试。\n\n' + q + '\n')
        with (ROOT / 'RECOVERY_DELIVERY_STOP_REPORT.md').open('x') as f:
            f.write('# 续跑交付停止\n\n独立CPU分析、报告或封存未全部完成。错误：' + repr(e) + '\n\n不能把现有报告解释为已完成交付；请核验analysis/CONTINUATION_DELIVERY_STOP.json。没有自动追加GPU工作。\n')
        raise
