"""Assemble source/identity findings without importing execution or GPU modules."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import runpy

here=Path(__file__).resolve().parent
root=here.parents[1]
cfg=json.loads((root/'ROOT.json').read_text())
download=root/'delivery/remote_evidence'
sha=lambda b:hashlib.sha256(b).hexdigest()
helpers=runpy.run_path(str(here/'audit_frozen_sources_remote.py'))
verify=helpers['verify_rows'];cache={}
payload=json.loads((download/'PAYLOAD_BUILD_MANIFEST.json').read_text())
control=json.loads((download/'CONTROL_SOURCE_MANIFEST.json').read_text())
local=dict(status='PASS_LOCAL_DOWNLOADED_FROZEN_SOURCE_IDENTITIES',
           read_only=True,base=str(download),
           payload_manifest_SHA256=sha((download/'PAYLOAD_BUILD_MANIFEST.json').read_bytes()),
           control_manifest_SHA256=sha((download/'CONTROL_SOURCE_MANIFEST.json').read_bytes()),
           payload_files=verify(download,payload['files'],cache),
           control_files=verify(download,control['files'],cache),cards={})
for card in cfg['cards']:
    child=download/card['card_key'];m=json.loads((child/'SOURCE_MANIFEST.json').read_text())
    local['cards'][card['card_key']]=verify(child,m['execution_sources'],cache)
assert local['payload_manifest_SHA256']==cfg['payload_build_manifest_sha256']
assert local['control_manifest_SHA256']==cfg['control_manifest_sha256']
assert local['payload_files']['status']==local['control_files']['status']=='PASS'
assert all(x['status']=='PASS'for x in local['cards'].values())
p=here/'LOCAL_FROZEN_PAYLOAD_AUDIT.json'
with p.open('x')as f:json.dump(local,f,sort_keys=True,ensure_ascii=False,indent=2);f.write('\n')

remote_path=next(here.glob('REMOTE_SOURCE_AUDIT_*.json'))
remote=json.loads(remote_path.read_text())
partial=json.loads((here/'PARTIAL_INPUT_GDN_DESCRIPTOR_AUDIT.json').read_text())
worker=download/'gpu3/scripts/cublas_worker.py';workertext=worker.read_text()
tree=ast.parse(workertext)
func=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='loaded_library_identity')
namespace={'Path':Path}
def require(ok,message):
    if not ok:raise RuntimeError(message)
called=[]
namespace['require']=require
namespace['sha']=lambda path:called.append(str(path))
exec(compile(ast.Module(body=[func],type_ignores=[]),str(worker),'exec'),namespace)
contract=json.loads((download/'gpu3/analysis/STRICT_BASELINE_CONTRACT.json').read_text())
target=remote['unbound_lib_current_identity']['path']
maps_fixture='7f000000-7f001000 r-xp 00000000 00:01 1 '+target+'\n'
try:
    namespace['loaded_library_identity'](contract,maps_fixture)
    reproduction={'status':'FAIL_EXPECTED_REJECTION_NOT_OBSERVED'}
except RuntimeError as exc:
    reproduction=dict(status='PASS_REPRODUCED_FROZEN_GUARD_REJECTION',
                      exact_exception=str(exc),
                      basename=Path(target).name,
                      startswith_libtorch=Path(target).name.startswith('libtorch'),
                      disk_sha_function_called=bool(called),
                      evidence_type='SYNTHETIC_MAP_LINE_FROZEN_FUNCTION_CPU_REPRODUCTION_NOT_LIVE_MAP',
                      function_lines=[func.lineno,func.end_lineno])
assert reproduction['status']=='PASS_REPRODUCED_FROZEN_GUARD_REJECTION'
assert reproduction['exact_exception']=='Unbound actual loaded library:'+target
assert reproduction['disk_sha_function_called']is False

source_excerpts=[]
for start,end in [(185,202),(245,275),(286,307)]:
    lines=workertext.splitlines()
    source_excerpts.append(dict(path='gpu3/scripts/cublas_worker.py',start_line=start,end_line=end,
                                text='\n'.join(f'{i}: {lines[i-1]}'for i in range(start,end+1))))
meta_excerpts=[]
for entry in remote['torchaudio_current_metadata_sources']:
    found=[]
    for number,line in enumerate(entry['text'].splitlines(),1):
        if any(x in line for x in ('_load_lib("libtorchaudio")','torch.ops.load_library(path)')) or line.startswith(('__version__','Version:','Requires-Dist: torch')):
            found.append(dict(line=number,text=line))
    meta_excerpts.append(dict(path=entry['path'],bytes=entry['bytes'],sha256=entry['sha256'],selected_lines=found))

summary=dict(
 status='INDEPENDENT_SOURCE_IDENTITY_AUDIT_COMPLETE_REAL_FAIL_CLOSED_STOP_UNCHANGED',
 task_id=cfg['task_id'],execution_run_id=cfg['execution_run_id'],
 audit_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 GPU_imports=False,GPU_queries=False,GPU_dispatches=False,
 frozen_sources_or_records_modified=False,
 source_worker_SHA256=sha(worker.read_bytes()),
 direct_failure=dict(origin_card='gpu3',unit_id='B0_FP_Q0',
                    type='POST_EXECUTION_UNBOUND_LOADED_LIBRARY_IDENTITY',
                    cause='libtorchaudio.so basename matches the frozen libtorch prefix classifier, but its resolved path is absent from the frozen prospective_library_bindings.',
                    actual_exception=remote['global_first_failure']['content']['error'],
                    execution_context=remote['global_first_failure']['content']['execution_context'],
                    global_first_failure_SHA256=remote['global_first_failure']['identity']['sha256'],
                    callsite='worker line298, after qualify() returned and before RESULT status PASS is committed at lines301-302',
                    classifier='worker line194: base.startswith((libtorch,libc10)) => torch',
                    enforcing_gate='worker line197 requires bound non-driver library path; SHA check at line199 was not reached for this unbound path',
                    distinction='Unbound identity is not evidence of a changed binary, numerical mismatch, NaN, OOM, watchdog timeout, or failed workspace candidate.'),
 guard_reproduction=reproduction,
 current_library_supplement=remote['unbound_lib_current_identity'],
 torchaudio_current_metadata=meta_excerpts,
 current_hash_verification=dict(
     remote_payload_files=[remote['payload_files']['passed'],remote['payload_files']['count']],
     remote_shared_control_files=[remote['control_files']['passed'],remote['control_files']['count']],
     per_card_execution_files={k:[v['execution_files']['passed'],v['execution_files']['count']]for k,v in remote['cards'].items()},
     all_auth_origin_source_device_plan_links_PASS=all(all(x['binding_links'].values())for x in remote['cards'].values()),
     fresh_external_dependencies=[remote['fresh_external_dependencies']['passed'],remote['fresh_external_dependencies']['count']],
     fresh_bound_library_path_rows=[remote['fresh_bound_library_rows']['passed'],remote['fresh_bound_library_rows']['count']],
     model_safetensor_shards=[x for x in remote['fresh_external_dependencies']['rows']if x['path'].endswith('.safetensors')],
     remote_full_receipt=remote_path.name,remote_full_receipt_SHA256=sha(remote_path.read_bytes()),
     local_downloaded_payload_receipt='LOCAL_FROZEN_PAYLOAD_AUDIT.json',local_downloaded_payload_receipt_SHA256=sha(p.read_bytes())),
 partial_input_descriptor_audit=dict(status=partial['status'],total_committed_tokens=partial['total_token_receipts'],
     per_card_tokens={x['card']:x['tokens']for x in partial['cards']},
     descriptor_shape_dtype_checks=partial['descriptor_shape_and_dtype_checks'],
     updated_state_current_cache_recorded_SHA_checks=partial['current_state_recorded_SHA_checks'],
     next_initial_previous_cache_recorded_SHA_checks=partial['next_state_recorded_SHA_lineage_checks'],
     full_receipt='PARTIAL_INPUT_GDN_DESCRIPTOR_AUDIT.json',full_receipt_SHA256=sha((here/'PARTIAL_INPUT_GDN_DESCRIPTOR_AUDIT.json').read_bytes()),
     fixture_receipt='FIXTURE_PREFIX256_IDENTITY.json',fixture_receipt_SHA256=sha((here/'FIXTURE_PREFIX256_IDENTITY.json').read_bytes()),
     qualification_status='NOT_NUMERICAL_EQUIVALENCE_ALL_B0_FP_Q0_SELF_REFERENCE_ONLY'),
 source_excerpts=source_excerpts,
 limitations=[
   'The fatal exception records a path observed through /proc/self/maps, but the complete failing-process map and successful post-execution library manifest were not retained before exit.',
   'libtorchaudio.so is the first rejected unbound path reached in the sorted map traversal. Completeness of the remaining actual mapped-library closure is unknown; binding this one current path is not proof that any future guard would pass.',
   'Current disk SHA/bytes and current torchaudio source/metadata supplement provenance only. They cannot retrospectively bind the exact binary mapped during the failed process, cannot change STOP to PASS, and do not amend the frozen manifest.',
   'The exact Python import caller that brought torchaudio into the failing process is NOT_RETAINED. Current package _extension source shows a possible loading mechanism, not the historical caller trace.',
   'GPU3 committed all256 baseline reference snapshots; other cards committed252/253/253 before shared STOP. No second B0 fresh process, B1, performance, profile, H_C128 or Dual-M unit completed in this run.',
   'Baseline SnapshotStore FULL_REFERENCE byte_equal=true and index status PASS mean self-reference archival completion, not comparison across independent executions or candidate numerical equivalence.',
   'Partial descriptor/state-hash lineage is CPU metadata auditing. Independent archive verification owns fresh current raw tensor SHA/bytes; this source audit does not reread archived raw tensors or run numerical forward.',
   'No repair, retry, live code change, new numerical experiment, global fill disable, or fast executable is authorized or produced by this supplemental audit.'],
 supplemental_tool_history='AUDIT_CPU_TOOL_HISTORY.json documents one corrected local auditor role-label expectation; frozen scientific records were never changed.',
 local_delivery_closure_history='LOCAL_DOWNLOAD_CLOSURE_GAP.json preserves the initial116 missing .cpp/.cu/.cuh/.h source-evidence paths. This final local485 closure assertion only succeeds after the independent read-only delta download fills them; old catalog and frozen manifests remain unchanged.')
target=here/'SOURCE_FAILURE_AUDIT.json'
with target.open('x')as f:json.dump(summary,f,sort_keys=True,ensure_ascii=False,indent=2);f.write('\n')
md='''# 独立源码及依赖停止审计

本次真实状态仍为 `FAIL_CLOSED_STOP`。GPU3 的 `B0_FP_Q0` 完成256个输入及基线参考归档后，在发布 worker PASS 之前，被未绑定动态库身份门禁拒绝。它没有进入 B0 第二次 fresh 或 B1，因此候选数值等价与吞吐均未取得结论。

冻结 worker 第194行按 basename 的 `libtorch`/`libc10` 前缀归类。`libtorchaudio.so` 也以 `libtorch` 开头，因此被归类为 `torch`；该路径不在冻结的 `prospective_library_bindings` 或237项外部依赖中。第197行在读取该路径 SHA 之前拒绝它。独立 CPU 使用原函数和合成 map 单行复现了同一异常，未导入 torch 或访问 GPU。

远端当前逐文件核验：485/485 payload、34/34共享控制文件、四卡各113/113执行文件及 auth/origin/source/device/plan链接均匹配；237/237外部依赖（含四个模型分片）及28/28绑定库路径匹配。已下载的小文件也独立通过同一闭包核验。完整逐文件记录及SHA见 `REMOTE_SOURCE_AUDIT_20261019T052339498686Z.json` 和 `LOCAL_FROZEN_PAYLOAD_AUDIT.json`。

未绑定库的当前磁盘文件为2393848字节，SHA256为 `b7802c164ccf0911047a265ce489961615df2848fcca2403795cf854641ce073`。当前 package metadata 标示 torchaudio `2.5.1+cu121`。这只补充现在磁盘上的身份，不能追溯证明失败时已映射二进制，也不能把真实 STOP 改为 PASS。完整失败时 `/proc/self/maps` 和动态库身份列表未在退出前保留；引入 torchaudio 的确切 Python caller 也未保留，当前加载器源码不能替代历史调用轨迹。它是排序遍历中首个拒绝的未绑定路径，不能推断只补这一个库就足够。

全部1014条已提交基线元数据（四卡252/253/253/256）均与冻结 fixture 的input/next_input/cache_cursor规则一致。完整32768输入的raw/canonical SHA在远端独立核验，仅下载prefix256。261516项tensor dtype/shape、24336项当前updated_state/cache记录SHA以及24240项跨token initial_state/前一cache记录SHA一致。它们是元数据与记录哈希谱系核验，均未被当作跨fresh数值等价；当前大tensor的远端独立SHA/bytes由归档审计单独负责。

没有修复、重试、改动冻结来源或重新执行GPU；所有补充位于 `independent_delivery/source`。主要机器可读结论为 `SOURCE_FAILURE_AUDIT.json`。
'''
with (here/'SOURCE_FAILURE_AUDIT.md').open('x')as f:f.write(md)
print(json.dumps(dict(status=summary['status'],report=str(target),sha256=sha(target.read_bytes()),worker_SHA256=summary['source_worker_SHA256'])))
