#!/usr/bin/env python3
"""Assemble truthful CPU-only reports. No synthetic science measurements."""
import csv,datetime,hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(n,j):(R/n).write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n')
source=R/'analysis/source_identity/HISTORICAL_SOURCE_IDENTITY_REPORT.md'
(R/'HISTORICAL_RUNTIME_IDENTITY.md').write_text('当前为HISTORICAL_PORT的CPU准备；下述历史身份不是本轮GPU实测。历史物理分派/二进制精确性及原AIME缺失项保持UNKNOWN。\n\n'+source.read_text())
(R/'OLD_VS_NEW_PROTOCOL_DIFF.md').write_text('OLD与NEW存在真实协议差异，不要求两者byte parity。以下审计中的未来预算建议不构成授权；最终事前合同仅以qualification/GPU_PLAN.json的240秒dispatch/1800秒总上限为准。\n\n'+(R/'analysis/protocol/OLD_NEW_PROTOCOL_AND_INT8_COMPATIBILITY.md').read_text())
put('RESTORATION_MODE.json',{'task_id':'QWEN_GDN_HISTORICAL_RUNTIME_RESTORATION_V1','mode':'HISTORICAL_PORT','mode_status':'CPU_CANDIDATE_PREPARED_NOT_GPU_QUALIFIED','historical_raw_source_copies_match':True,'source_math_changes':0,'port_changes':['m.OLD points at isolated raw-copy runner with same raw SHA','relative aime26_common imported from isolated raw-copy directory','model/framework/native packages reused read-only with current SHA binding','load-time PRE/POST checks and independent outer controller outside performance window'],'historical_binary_EXACT':'UNKNOWN_NOT_CLAIMED','historical_physical_kernel_selection':'NOT_FULLY_RETAINED','historical_full_OLD_token_byte_reference':'NOT_RETAINED','prior_AIME_complete_hardware_and_weight_tokenizer_SHA':'NOT_RETAINED_IN_FULL','restored_GPU_math':'NOT_RUN','restored_fresh_repeatability':'NOT_RUN','restored_TPS':'NOT_RUN','ALLOW_GPU_RESTORE':'NO'})
put('GPU_RESOURCE_LEDGER.json',{'task_id':'QWEN_GDN_HISTORICAL_RUNTIME_RESTORATION_V1','status':'NO_GPU_DISPATCH_CPU_ONLY','ALLOW_GPU_RESTORE':'NO','GPU_queries':0,'GPU_dispatches':0,'actual_GPU_worker_seconds':0,'actual_GPU_hours_for_new_dispatches':0,'first_dispatch_events':[],'first_exit_events':[],'first_release_events':[],'own_resources_release':'NOT_APPLICABLE_NO_DISPATCH','current_GPU_occupancy':'NOT_QUERIED','historical_GPU_ledgers':'READ_ONLY_UNCHANGED_NOT_MERGED','future_budget':{'max_active_GPUs':1,'hard_seconds':1800,'protective_seconds':1650,'cleanup_reserve_seconds':150,'dispatch_max_seconds':240,'cleanup_wait_seconds':120},'no_current_resource_estimate_claimed':True})
put('RESTORED_FP_BENCHMARK_STATUS.json',{'status':'NOT_RUN_GPU_NOT_AUTHORIZED','current_restored_TPS':None,'current_restored_ms_per_token':None,'current_restored_peak_memory':None,'current_repeats':0,'RESTORED_FP_BENCHMARK.csv':'NOT_CREATED_NO_ACTUAL_MEASUREMENTS','historical_reference_file':'HISTORICAL_BENCHMARK_REFERENCE.csv','history_is_not_current_measurement':True})
historical=R/'evidence/reconciliation/MATCHED_RUNTIME_BENCHMARK.csv'
with historical.open(newline='') as f:hist=[x for x in csv.DictReader(f)if x['condition']in ['OLD_FP','NEW_FP']and x['profile']=='False']
with (R/'HISTORICAL_BENCHMARK_REFERENCE.csv').open('w',newline='') as f:
    fields=['evidence_role','original_source_csv_SHA256']+list(hist[0]);w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for row in hist:w.writerow(dict(evidence_role='HISTORICAL_REFERENCE_ONLY_NOT_CURRENT_RESTORATION',original_source_csv_SHA256=sha(historical),**row))
statuses={'HISTORICAL_RUNTIME_RESTORED_AND_QUALIFIED':'NOT_ESTABLISHED_GPU_NOT_RUN','HISTORICAL_RUNTIME_PARTIALLY_RESTORED':'NOT_USED_AS_FINAL_GPU_RESULT_CPU_SOURCE_PREPARATION_ONLY','HISTORICAL_PORT_QUALIFIED':'NOT_ESTABLISHED_GPU_NOT_RUN','FP_ONLY_QUALIFIED_INT8_NOT_RUN':'NOT_ESTABLISHED_FP_GPU_NOT_RUN','RUNTIME_REPRODUCIBILITY_FAIL':'NOT_OBSERVED_NEW_GPU_NOT_RUN','HISTORICAL_ASSETS_INSUFFICIENT':'NOT_APPLIED_TO_SOURCE_PORT; HISTORICAL_BINARY_EXACT_GAPS_RETAINED','CPU_ONLY_COMPLETE_GPU_NOT_AUTHORIZED':'APPLICABLE','FAIL_CLOSED_STOP':'NOT_TRIGGERED_NEW_GPU_NEVER_DISPATCHED'}
put('RUNTIME_RESTORATION_DECISION.json',{'task_id':'QWEN_GDN_HISTORICAL_RUNTIME_RESTORATION_V1','status':'CPU_ONLY_COMPLETE_GPU_NOT_AUTHORIZED','restoration_mode':'HISTORICAL_PORT','historical_source_and_short_window_protocol_recoverable':True,'historical_13_TPS_recovered':'UNKNOWN_NOT_MEASURED_THIS_RUN','new_GPU_math':'NOT_RUN','new_fresh_self_repeat':'NOT_RUN','INT8_GPU_qualification':'NOT_RUN_NOT_QUALIFIED','long_horizon_qualification':'NOT_RUN_NOT_QUALIFIED','new_AIME_or_Future_KL_authorized':False,'official_ready_runtime_launcher_created':False,'candidate_entry':'historical_runtime_candidate.sh --help/--check/--print-env only','decision_states':statuses,'historical_results_must_not_be_mixed_as_matched_new_baselines':True,'GPU_queries':0,'GPU_dispatches':0})
checks=sorted((R/'qualification').glob('CPU_OBSERVER_*.json'));latest=read(checks[-1]);observer=latest['result'];assert observer['status'].startswith('PASS') and latest['returncode']==0
fixture=read(sorted((R/'qualification').glob('CPU_FIXTURE_*.json'))[0])['result'];assert fixture['status'].startswith('PASS')
controller=read(R/'qualification/CONTROLLER_CPU_TEST_RESULTS.json') if (R/'qualification/CONTROLLER_CPU_TEST_RESULTS.json').exists() else None
summary={'status':'PASS_CPU_PREPARATION_NOT_GPU_QUALIFICATION','tiny_real_class_checks':len(fixture['checks']),'tiny_real_cache_observer_checks':len(observer['checks']),'observer_worker_source_sha256':observer['worker_source_sha256'],'worker_current_sha256':sha(R/'worker/restoration_worker.py'),'all_original_tiny_checks_pass':all(x['passed']for x in fixture['checks']),'all_observer_checks_pass':all(x['passed']for x in observer['checks']),'GPU_queries':0,'GPU_dispatches':0,'native_recurrent_FP64_small_fixture':fixture['checks'][0]}
put('qualification/CPU_PREPARATION_SUMMARY.json',summary)
report='''当前状态为 **CPU_ONLY_COMPLETE_GPU_NOT_AUTHORIZED**。已恢复历史OLD源码与短窗口执行协议的隔离候选，模式 **HISTORICAL_PORT**；尚未验证当前9B GPU数学、fresh重复性或13TPS性能恢复。当前ALLOW_GPU_RESTORE=NO，本任务没有GPU查询、初始化或dispatch。

| 对照项 | 本轮OLD恢复候选 | NEW_STRICT参考 |
|---|---|---|
| 当前TPS / ms-token | NOT_RUN | 本轮NOT_RUN |
| 历史短窗口FP中位TPS | 13.742036038730332（历史OLD） | 7.77671073754353（历史NEW） |
| 历史中位ms/token | 72.76942057069391 | 128.58906982000917 |
| 当前9B FP logits/GDN state | NOT_RUN | 本轮NOT_RUN |
| 当前GPU fresh重复性 | NOT_RUN | 本轮NOT_RUN |
| 当前显存峰 | NOT_RUN | 本轮NOT_RUN |
| Runtime身份 | 原源码rawSHA已核验；当前依赖绑定；历史完整二进制UNKNOWN | 已有封存协议只读参考 |

六个问题逐项回答：

1. **历史13.742 TPS的真实OLD是否成功恢复？** 原始执行入口与完整Python数学/Cache路径已经定位并隔离复制；当前性能恢复未验证，不能写成功恢复到13TPS。13.742来自调和任务的三次OLD_FP固定短窗中位，不是原AIME自由生成或INT8吞吐。
2. **缺失什么？** 历史完整driver/cuBLAS/cuDNN实际动态库字节与maps、全部optional/fused物理分派、完整旧AIME权重/Tokenizer原始SHA及逐token数值字节参考未保留；实际OLD目录common的原路径历史pin也有范围缺口，当前SHA与canonical相同。历史调和同卡UUID、输入、32层、24GDN、FP32state、原runner和benchmark源码可恢复。当前四权重分片freshSHA匹配调和共同checkpoint，不能倒补原AIME缺口。
3. **当前服务器同窗实测多少TPS？** 本轮没有执行GPU，答案是NOT_RUN；不拿13.742或7.777填当前实测栏。RESTORED_FP_BENCHMARK.csv没有合法行，故未制造空科学CSV，另有状态JSON和清楚标记的历史参考CSV。
4. **数学与重复性是否可接受？** 真实tiny Qwen类的FP32/BF16 CPU检查通过，包括native GDN、Attention、MLP、Cache、FP64小例、state消费；观察器和控制器CPU回归通过。它们不能替代9B GPU两fresh，也不是32K/81K稳定性证明。OLD保留native chunk cumsum，但装饰器物理分派历史证据不足，不能仅unwrap文件名断言所有实际kernel。当前GPU数学/自身重复性仍NOT_RUN。
5. **能用于后续Hadamard/Group32 Sym/Asym吗？** 有源码和协议上的接入基础；目前仅候选，未批准正式GPU科研。C128的QDQ写回点有现成代码，C32 Sym/Asym的轴/scale/zero-point/时序与数值还需独立冻结及资格；没有继承FP或历史H吞吐。当前连FP_GPU资格也未取得，不能提前用FP_ONLY_QUALIFIED标签。
6. **哪些要重建同协议基线？** 未来FP Native、H+C128 Sym、H+C32 Sym、H+C32 Asym都应在同一恢复协议、模型/输入、旋转、prefill/GDN更新、QDQ注入、Cache生命周期、sampling及Future-KL评分器下重新匹配。历史NEW_STRICT Future-KL、旧AIME与不同OLD预热谱系只作引用，不能代入新匹配比较。

关键恢复事实：历史OLD实际deterministic=False、24/24线程、matmulTF32=False、cuDNN TF32=True/cuDNN deterministic=False；CUBLAS/OMP/MKL/CUDA_DEVICE_ORDER实际unset，来自当次ENV，不是默认值猜测。原runner本身未显式设置这些开关，新控制器仅为新隔离协议显式复现已记录状态，未改NEW。当前CPU导入观察到相同有效值，但尚无本轮GPU环境证明。Torch/CUDA版本相同不能证明历史二进制相同；pip/conda组件并存，未来实际库选择需按前后maps/SHA核验。

隔离副本保留432逻辑copy记录、430物理copy；194个原路径SHA分目录canonical条目独立。Mac的历史路径投影有两组同字节大小写别名，已显式记录，并以canonical副本避免丢源；关键runner/common没有别名。原source 201项及24份复制执行证据已核验，CPU工具错误/审计修正历史保留。没有覆盖历史环境或修改历史结果、源码、seal、账本。

OLD matched H首token不QDQ，共255次；NEW从首tokenQDQ共256次；原AIME又是整段prompt不量化prefill。C128持久cache是FP32 QDQ载体，不是packed INT8。这些差异禁止跨协议混KL。有限性、短prefix重现与代码同AST也不能代替所有未初始化读/长程安全证明；没有关闭fill或删安全检查。

准备好的GPU程序仍被拒绝执行：单卡GPU0原UUID，最大0.5GPUh/1800秒、保护1650、单dispatch240、清理预留150；先两fresh各32token完整数值/cache byte及实际激活驱动FP64递推，再优先三次历史128+128 FP无Profiler测量，最后预算许可才一次NEW短窗。不得改阈值、换卡、重复前缀、自动扩预算或按结果搜索。32prefix、256性能窗与未来32K/81K/INT8分开标记。当前授权JSON与ROOT都为NO，直接CLI YES也会在任何GPU query前拒绝。

仅生成 historical_runtime_candidate.sh 的--help/--check/--print-env CPU入口；没有正式可用Runtime启动器。新的GPU阶段须另有明确人类授权并建立独立执行身份，保留本CPU seal。本任务没有自动启动AIME、Group32、长Future-KL、训练、四卡运行或新的定时跟进。

全部八种状态逐项见RUNTIME_RESTORATION_DECISION.json；本轮只适用CPU_ONLY_COMPLETE_GPU_NOT_AUTHORIZED，不将CPU准备冒称HISTORICAL_PORT_QUALIFIED或恢复速度成功。SOURCE_FREEZE绑定准备程序和事前合同，SHA256_MANIFEST与FINAL_SEAL_RECEIPT绑定本轮交付；历史不变证明范围另列，不伪称检查了整台服务器所有资产。
'''
(R/'FINAL_REPORT.md').write_text(report)
print(json.dumps({'status':'CPU_REPORTS_ASSEMBLED','CPU_summary':summary,'current_TPS':'NOT_RUN'}))
