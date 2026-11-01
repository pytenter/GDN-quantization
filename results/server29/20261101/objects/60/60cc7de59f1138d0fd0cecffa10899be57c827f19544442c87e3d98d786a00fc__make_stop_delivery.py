#!/usr/bin/env python3
"""Independent post-STOP report assembly. Never imports/dispatches GPU code."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'delivery'
OLD=BASE.parent/'QWEN_GDN_FAST_RUNTIME_QUALIFICATION_V1'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x')as f:
        if isinstance(value,(dict,list)):json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
        else:f.write(value)

def main():
    resource=read(BASE/'independent_delivery/resource/RESOURCE_TERMINAL_AUDIT.json')
    partial=read(BASE/'independent_delivery/PARTIAL_NUMERICAL_METADATA_AUDIT.json')
    descriptor=read(BASE/'independent_delivery/source/PARTIAL_INPUT_GDN_DESCRIPTOR_AUDIT.json')
    source_path=next((BASE/'independent_delivery/source').glob('REMOTE_SOURCE_AUDIT_*.json'))
    source=read(source_path)
    archive_path=BASE/'independent_delivery/receipts/FINAL_ARCHIVE_STOP_AUDIT.json'
    archive=read(archive_path)
    assert resource['unique_dispatches']==4 and resource['full_unit_PASS']==0
    assert source['status']=='CURRENT_FROZEN_IDENTITIES_PASS_REAL_STOP_REMAINS'
    assert descriptor['errors']==[] and partial['total_committed_token_receipts']==1014
    assert 'PASS' in archive['status']
    assert resource['release']['current_NVML_compute_apps_empty']
    # Preserve all CPU parent assets verbatim, with the original external seal.
    oldmanifest=read(OLD/'SHA256_MANIFEST.json');oldseal=read(OLD/'FINAL_SEAL_RECEIPT.json')
    assert sha(OLD/'SHA256_MANIFEST.json')==oldseal['manifest_sha256']
    for row in oldmanifest['files']:
        p=OLD/row['path'];assert p.stat().st_size==row['bytes']and sha(p)==row['sha256']
    files={r['path']for r in oldmanifest['files']}|{'SHA256_MANIFEST.json','SHA256SUMS.txt','FINAL_SEAL_RECEIPT.json'}
    for name in sorted(files):
        target=OUT/'parent_cpu'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OLD/name,target)
    # Copy supplemental CPU audit products; never include compressed duplicate,
    # models, raw tensor files, or live/frozen program mutations.
    for path in sorted((BASE/'independent_delivery').rglob('*')):
        if not path.is_file()or path.suffix not in {'.py','.json','.jsonl','.md','.txt','.csv','.log'}:continue
        rel=path.relative_to(BASE/'independent_delivery')
        if rel.name=='make_stop_delivery.py':continue
        dest=OUT/'audits'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    for name in ['ROOT.json','HANDOFF.md','HUMAN_GPU_AUTHORIZATION.json']:
        dest=OUT/'evidence'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/name,dest)
    for path in sorted((BASE/'analysis').glob('*')):
        if path.is_file()and path.suffix in{'.json','.jsonl','.md','.txt'}:
            dest=OUT/'evidence/preparation'/path.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    snapshots=list((BASE/'monitoring').glob('SNAPSHOT_*.json'))
    for p in snapshots:
        dest=OUT/'evidence/monitoring'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    counts=resource['committed_token_counts'];lower=resource['reservation_GPUh_lower'];upper=resource['reservation_GPUh_upper']
    error=resource['first_failure']['error']
    cards=[]
    for x in partial['cards']:
        y={k:v for k,v in x.items()if k not in{'reference_index','env'}}
        y.update(B0_second_fresh='NOT_RUN',B1_fresh='NOT_RUN',performance='NOT_RUN',unit_status='FAIL_CLOSED_STOP'if x['card']=='gpu3'else'ABORTED_SHARED_STOP_NO_RESULT',comparison_status='NOT_RUN_REFERENCE_ONLY')
        cards.append(y)
    put('NUMERICAL_PARITY_RESULTS.json',dict(task_id='QWEN_GDN_FAST_RUNTIME_QUALIFICATION_V1',execution_run_id=BASE.name,status='INCOMPLETE_OR_ENGINEERING_STOP',primary_decision='FAIL_CLOSED_STOP',failure_kind='UNBOUND_ACTUAL_LOADED_LIBRARY_IDENTITY',failure=resource['first_failure'],GPU_dispatches=4,legal_unit_PASS=0,cards=cards,total_committed_reference_tokens=1014,total_reference_raw_bytes=160036398000,reference_only=True,FP_logits_parity='NOT_RUN_NO_FRESH_COMPARISON',FP_state_parity='NOT_RUN_NO_FRESH_COMPARISON',FP_GDN_intermediates_parity='NOT_RUN_NO_FRESH_COMPARISON',FP_cache_next_input_parity='NOT_RUN_NO_FRESH_COMPARISON',FP_fresh_determinism='NOT_RUN',B1_self_determinism='NOT_RUN',H_C128='NOT_QUALIFIED_FOR_THIS_CODEC',H_DUAL_M='NOT_QUALIFIED_FOR_THIS_CODEC',max_abs_or_relative_error='NOT_COMPUTED_NO_COMPARISON_EXISTS',earliest_numerical_divergence='NOT_OBSERVED_IDENTITY_STOP_NOT_A_NUMERICAL_FAIL',metadata_input_descriptor_audit=descriptor['status'],partial_reference_256_index_not_unit_PASS=True))
    put('RESOURCE_LEDGER.json',dict(task_id=resource['task_id'],execution_run_id=BASE.name,status=resource['status'],GPU_dispatches=4,GPU_time_limit=None,EXACT_GPUh='UNRESOLVED',exact_physical_GPUh=resource['exact_physical_GPUh'],observed_worker_device_reservation_GPUh_lower=lower,observed_worker_device_reservation_GPUh_upper=upper,resource_semantics=resource['resource_semantics'],NVML_presence_span_GPUh_descriptive_only=resource['sum_NVML_presence_observation_span_GPUh'],NVML_span_semantics=resource['NVML_span_semantics'],unique_first_exit_observations=4,unique_first_exit_release_observations=4,all_own_resources_released=resource['release'],audit='audits/resource/RESOURCE_TERMINAL_AUDIT.json',old_ledgers_unchanged=True))
    put('FAST_RUNTIME_DECISION.json',dict(task_id='QWEN_GDN_FAST_RUNTIME_QUALIFICATION_V1',execution_run_id=BASE.name,status='FAIL_CLOSED_STOP',scientific_completion='INCOMPLETE_OR_ENGINEERING_STOP',ALLOW_GPU_DIAG='YES',GPU_time_limit=None,maximum_active_GPUs=4,candidate='B1 CUBLAS_WORKSPACE_CONFIG=:16:8',candidate_executed=False,candidate_numeric_equivalence='NOT_RUN',candidate_performance='NOT_RUN',candidate_reproducibility='NOT_RUN',FAST_STRICT_EQUIVALENT_VALIDATED=False,FAST_NEW_PROTOCOL_REPRODUCIBLE=False,approved_Fast_codecs=[],FP='NOT_QUALIFIED_FAST',H_C128='NOT_QUALIFIED_FOR_THIS_CODEC',H_DUAL_M='NOT_QUALIFIED_FOR_THIS_CODEC',fill_optimization='FILL_OPT_SAFETY_UNPROVEN',global_fill_disable_approved=False,fast_formal_launcher_created=False,GPU_dispatches=4,not_a_negative_performance_result=True,not_a_numerical_difference_result=True,no_retry_performed=True,D3='NOT_RUN_FP_QUALIFICATION_FAILED',reason=error))
    put('SOURCE_AND_AUDIT_BINDINGS.json',dict(task_id=source['task_id'],execution_run_id=BASE.name,status=source['status'],CPU_parent_manifest_sha256=oldseal['manifest_sha256'],payload_files_verified=source['payload_files']['count'],control_files_verified=source['control_files']['count'],external_files_verified=source['fresh_external_dependencies']['count'],per_card_execution_files=113,bound_library_paths=28,unbound_library=source['unbound_lib_current_identity'],runtime_loaded_library_complete_identity='UNRESOLVED_AT_FAILURE',current_file_hash_not_retrospective_runtime_PASS=True,source_audit='audits/source/'+source_path.name,archive_audit='audits/receipts/FINAL_ARCHIVE_STOP_AUDIT.json'))
    # This CSV contains actual historical numbers only, never invented new TPS.
    historical=[]
    with(OLD/'PERFORMANCE_COMPARISON.csv').open(newline='')as f:
        for r in csv.DictReader(f):
            if not r['STRICT_value']:continue
            historical.append(dict(metric=r['metric'],codec=r['codec'],value=r['STRICT_value'],unit=r['unit'],status='HISTORICAL_REFERENCE_ONLY_NOT_THIS_RUN',source=r['STRICT_source'],source_sha256=r['STRICT_source_sha256'],scope=r['scope']))
    assert len(historical)==6
    with(OUT/'PERFORMANCE_COMPARISON.csv').open('x',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(historical[0]));w.writeheader();w.writerows(historical)
    put('EXECUTION_PLAN.md',f'''本轮是原CPU封存后的四卡GPU资格执行，最终状态 `FAIL_CLOSED_STOP`。

人类明确授权四卡及无GPU时间限制；累计和单元时间上限已移除，1800秒只作为无真实进度watchdog。冻结名单每卡16单位，先四个256输入FP fresh数值单位，再六个128warmup+128measurement测速，只有原5%+严格不重叠收益门槛通过才进入Profile和H资格。所有B0/B1对照绑定同一卡，不跨卡合并TPS。

实际仅四卡各派发一次 `B0_FP_Q0`，提交 {counts} 个参考token。GPU3在完整参考构建后，动态库身份门禁发现未绑定 `libtorchaudio.so`；全局停止并释放四卡。B0第二fresh、B1、性能、Profile、H及Dual-M均未执行。没有重试、热补丁或增加阶段。详细冻结计划保存在 `remote_evidence/gpuN/qualification/CUBLAS_PREREGISTRATION.json`；原CPU151文件封存完整保存在 `parent_cpu/`。
''')
    put('STRICT_RUNTIME_CONTRACT.md',f'''Strict仍为新版B0 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；B1候选仅`:16:8`。模型Qwen3.5-9B，参数BF16，GDN recurrent carrier FP32；deterministic=True、fill=True、matmul/cudnn TF32=False、Torch/OMP/MKL线程1/1、固定非AIME输入、seed20261015，逻辑设备cuda:0映射到各自冻结物理UUID。

四份完整合同及237外部依赖的当前SHA/bytes复核通过。原输入完整32768 IDs raw/canonical SHA与冻结合同一致，1014条已提交输入/next/cache-cursor及记录Tensor描述/状态SHA谱系核验通过；这不是跨fresh数值比较。

实际执行库身份闭包未通过：`{source['unbound_lib_current_identity']['path']}` 未列入28条prospective library路径或237条外部依赖。退出前完整/proc maps未保留；停止后该文件的当前2393848字节SHA为 `{source['unbound_lib_current_identity']['sha256']}`，不能据此追认运行资格。它是排序遍历中首个被拒绝路径，不能保证只补这一个库就能通过。详见 `SOURCE_AND_AUDIT_BINDINGS.json`。
''')
    put('HISTORICAL_EVIDENCE_REUSE.md','''原根因任务的OLD_FP/NEW_FP中位TPS为13.742036/7.776711；OLD_H+C128/NEW_H+C128为9.064610/6.650598。历史mm CPU self约7.0764/55.8277ms/token，首次launch前约6.8070/55.3913ms，直接GPU GEMM约20.9284/20.9390ms。支持主机launch前开销方向，唯一根因仍UNRESOLVED。

这些数据均为历史参考，不是本轮四卡新测量；H的新旧prefill/cache协议不同，Dual-M的6.801314154 TPS也来自另一冻结任务，不能拼成当前严格同条件Codec比较。`PERFORMANCE_COMPARISON.csv`仅保留六条有来源SHA的历史实测记录。本轮测速0条，不计算提速率，不把保存参考轨迹的耗时当作解码TPS。完整历史审计保留于 `parent_cpu/HISTORICAL_EVIDENCE_REUSE.md`。
''')
    put('CUBLAS_CONFIG_QUALIFICATION.md',f'''本轮GPU已获授权并真实执行，但cuBLAS资格未完成。四卡各仅运行B0首个参考，提交token数252/253/253/256，合计1014。GPU3的 `NUMERICAL_INDEX.json` 的PASS只表示256输入参考构建，`actual_bytes_compared=false`；其worker最终RESULT为FAIL_CLOSED_STOP。另三卡被共享STOP终止，RESULT和完整INDEX缺失仍保留缺失，不补写PASS。

触发点是冻结worker的动态库身份门禁：basename以libtorch/libc10开头会归入torch库，故libtorchaudio.so也必须存在于绑定清单。该实际加载路径不在清单，执行在发布RESULT PASS之前停止。这不是已观察到的B0/B1误差、无收益结论、GPU时限或磁盘满事件。

没有B0第二fresh、B1自重复、B0/B1字节对照，故误差最大值和最早数值分歧均不适用，不能填0。没有任何合法128+128测量，原5%+不重叠门槛未被检验。H/Dual-M仍NOT_QUALIFIED_FOR_THIS_CODEC；条件StageD3因FP资格失败未进入。没有自动修复或重跑。
''')
    put('UNINITIALIZED_MEMORY_SAFETY_AUDIT.md','''本轮未修改或测试fill=False。已保存的ENVIRONMENT记录仍为deterministic_algorithms=True、deterministic_fill=True、TF32=False。

前序CPU审计的13类缓冲、45条源码目录和63项检查原样保留于 `parent_cpu/UNINITIALIZED_MEMORY_SAFETY_AUDIT.md`。历史新增847 kernel/token中724为fill类，但不能把类别计数当作全部由单一fill开关引起的因果证据。generic resize增长、Flash/可选FLA masked tile和跨stream全写后读覆盖仍未完整闭合，结论维持FILL_OPT_SAFETY_UNPROVEN。

本轮未获得或生成独立fill安全资格；没有全局关闭fill/determinism，也没有以输出相同替代未初始化读取安全证明。
''')
    put('OPTIMIZATION_DIFF.md','''原CPU/历史源码、旧ledger及seal均未修改。新执行副本仅增加已授权四卡映射、无GPU时间限额控制、1800秒无进展监测、精确自身进程清理、共享首次失败、同卡对照和跨卡测速互斥、归档流式I/O。

core AST仅允许cuBLAS环境字符串及物理可见设备准入字面值；反向AST必须恢复原文语义。qualify数学/DeepRecorder/replay/Profile控制差分均有CPU断言；GDN、Codec、FP32 state/QDQ/writeback次序保持原样。B1尚未执行，这些工程变化不构成已验证提速。

停止后仅新增独立只读审计和交付程序。未把libtorchaudio补进已冻结白名单、未移除门禁、未热改程序、未重试单位。一次启动前命令模板编译错误和CPU审计工具错误保留在对应history中，均不伪装为GPU试验结果。
''')
    table='\n'.join(f'| GPU{i} | {n}/256 | '+('完整参考INDEX；worker身份FAIL'if i==3 else'共享停止；缺失RESULT保留')+' | 未执行 |'for i,n in enumerate(counts))
    put('FINAL_SCIENTIFIC_REPORT.md',f'''本轮结论是 **FAIL_CLOSED_STOP / INCOMPLETE_OR_ENGINEERING_STOP**。四卡GPU已真实使用，停止原因是执行期加载的libtorchaudio.so未纳入冻结库身份清单；没有完成B0/B1资格或性能比较，因此没有合格Fast Runtime，也没有提速或负性能结论。

| 设备 | 已提交B0参考token | 单位状态 | B1/测速 |
|---|---:|---|---|
{table}

合计1014条参考快照、160,036,398,000字节（约149.05GiB）。它们是首个B0自参考，不是四卡之间或fresh-process间的数值等价证据。GPU3参考INDEX虽写PASS，最终worker RESULT仍FAIL；其他三卡缺失的RESULT未补写。

用户六个主要问题的回答：

1. **新版约7.78 TPS为什么慢？** 历史Profiler支持249次mm普遍存在主机首次launch前开销增大，GPU关联GEMM时间接近。唯一内部根因仍未确定；本轮未增加可合法比较的新吞吐证据。
2. **小cuBLAS Workspace是否提速？** 未知。B1未运行，0条测速，不能计算提速率或认定候选无效。原三重复、5%和不重叠门槛没有放宽。
3. **deterministic fill能否安全优化？** 仍为FILL_OPT_SAFETY_UNPROVEN。CPU覆盖缺口原样保留；本轮fill=True，没有启动第二项GPU假设。
4. **能否证明数学、状态、logits、QDQ和确定性不变而提速？** 尚不能。冻结代码数学保持原样且部分参考已保留，但B0第二fresh、B1重复和跨条件byte比较均未完成；不能把参考自比较的byte_equal=True当成等价PASS。
5. **能否批准可重复的新数值高速协议？** 不能。没有候选重复性或性能证据，也没有已量化的B0/B1差异。未降级阈值将它转成类型B。
6. **是否提供Fast启动脚本？** 没有。`fast_validated_runtime.sh`未生成；Strict入口仅保留冻结CPU身份/环境输出，不能视为本轮GPU数值资格。

执行身份核验：最初小文件下载过滤遗漏的116个C/CUDA源码已从远端独立补齐、逐SHA核验，缺口历史仍保留。485个payload文件、34项共同控制绑定、每卡113个执行源、237个外部依赖（含模型分片）和28个已绑定库路径当前SHA/bytes均匹配。触发的新增libtorchaudio不在该闭包，当前磁盘SHA不能追溯证明退出前加载映射。该门禁在256输入完成后、发布RESULT PASS前执行，因而触发了全局停止。未发现最早数值分歧记录，不计算最大误差；这不等于数值误差为零。

1014条输入/next/cache-cursor、261516次Tensor dtype/shape描述、当前状态与后继初始状态的记录SHA谱系由独立CPU分析核验；这些仅描述已保存B0前缀。已提交原始文件共1014个；另有3个未提交文件360,577,048字节，单独保存且不计入科学结果。远端原始文件的最终当前SHA/bytes和未提交文件分类见 `audits/receipts/FINAL_ARCHIVE_STOP_AUDIT.json`，不以元数据替代实际原始文件核验。所有大型张量留在29永久归档，Mac仅小文件。

资源：仅四次B0首参考派发，四份首次exit/释放观测唯一且完整；独立重算与冻结账本一致。worker/设备预留观测范围 **{lower:.9f}–{upper:.9f} GPUh**，含CUDA前CPU依赖/权重哈希；精确物理GPUh仍UNRESOLVED，NVML首次至末次可见跨度只作描述。四卡任务进程、NVML compute-apps及对应tmux已结束。停止不是GPU时间限额触发；未应用旧任务预算。不同主机wall clock有偏差，计量采用同一55主机monotonic事件，不混用绝对时间求差。

历史参考：OLD_FP/NEW_FP为13.742036/7.776711 TPS，OLD_H/NEW_H为9.064610/6.650598 TPS，另一个Dual-M任务约6.801314154 TPS。均未当作本轮同条件匹配结果；本轮全部当前性能为NOT_RUN，测量显存峰及当前mm/kernel Profile亦未产生。模型加载ENVIRONMENT和NVML观测不是完整测量窗口峰值。

原第十四节七种状态的适用性：

| 状态 | 本轮判定 |
|---|---|
| FAST_STRICT_EQUIVALENT_VALIDATED | 不满足，缺fresh/跨条件/性能资格 |
| FAST_NEW_PROTOCOL_REPRODUCIBLE | 不满足，无B1重复性/收益证据 |
| FAST_CANDIDATE_NO_BENEFIT | 未检验，不能宣称负性能 |
| FAST_CANDIDATE_NUMERICAL_FAIL | 未观察到数值比较失败；实际为身份STOP |
| FAST_CANDIDATE_SAFETY_UNPROVEN | fill独立假设仍安全性未证明；未执行 |
| CPU_ONLY_COMPLETE_GPU_NOT_AUTHORIZED | 仅原CPU封存的历史状态；当前已授权并用GPU |
| FAIL_CLOSED_STOP | **当前主终态**：未绑定实际动态库身份 |

没有修复已冻结清单、删除历史失败、重启已派发单位或增加新配置；H/Dual-M和条件D3因前置资格失败未执行。当前结果只能支持一次真实工程停止交付，不能支持Workspace、Codec科学方向、AIME准确率或长程Future-KL结论。

全部12项要求的交付均在本目录：执行计划、Strict合同、历史复用、cuBLAS资格、fill安全审计、数值JSON、历史性能CSV、Fast决策、差分、资源账本、本报告及SHA清单/封存回执。完整原CPU151文件封存另存parent_cpu，原目录和旧科学资产保持原位。逐件本地/55/29交付核验以 `LOCAL_DELIVERY_VERIFICATION.json` 为外部回执，封存索引排除自身和这些外部锚，避免循环SHA。
''')
    put('runtime_presets/README.md','''没有通过资格的Fast模式，因此不存在fast_validated_runtime.sh。

strict_runtime.sh只转交parent_cpu中逐字节保存的冻结CPU身份/环境入口；支持--check和--print-env，不接受任意GPU进程命令，不自动启动GPU。本轮实际GPU身份门禁失败，不能以此入口声称GPU数值或Codec已资格。
''')
    put('runtime_presets/strict_runtime.sh','''#!/bin/sh
set -eu
exec /bin/sh "$(CDPATH= cd -- "$(dirname -- "$0")/../parent_cpu/runtime_presets" && pwd)/strict_runtime.sh" "$@"
''')
    (OUT/'runtime_presets/strict_runtime.sh').chmod(0o755)
    put('runtime_presets/runtime_identity.json',dict(status='FAIL_CLOSED_STOP_NO_FAST_VALIDATION',strict_cpu_source_manifest_sha256=oldseal['source_manifest_sha256'],strict_original_entry_sha256=sha(OLD/'runtime_presets/strict_runtime.sh'),approved_Fast_codecs=[],runtime_GPU_qualification='INCOMPLETE_ENGINEERING_STOP',entry_scope='CPU_SOURCE_IDENTITY_AND_ENVIRONMENT_ONLY'))
    assert not list(OUT.rglob('fast_validated_runtime.sh'))
    print(json.dumps(dict(status='STOP_REPORTS_ASSEMBLED_PENDING_FINAL_HASH_SEAL',files=len(list(OUT.rglob('*'))),candidate_TPS='NOT_RUN',GPU_dispatches=4)))

if __name__=='__main__':main()
