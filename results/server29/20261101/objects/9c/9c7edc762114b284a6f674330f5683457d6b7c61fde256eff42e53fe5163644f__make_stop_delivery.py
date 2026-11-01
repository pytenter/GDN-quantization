"""Assemble real numerical STOP reports and verified small evidence; no GPU imports."""
import csv,hashlib,json,shutil
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'delivery'
CPU=BASE.parent/'QWEN_GDN_FAST_RUNTIME_QUALIFICATION_V1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(name,value):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x')as f:
  if isinstance(value,(dict,list)):json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
  else:f.write(value)
def copytree(source,target):
 for p in sorted(source.rglob('*')):
  if p.is_file():
   assert not p.is_symlink();dest=target/p.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
def main():
 assert not OUT.exists(),'Existing delivery must be inspected rather than overwritten'
 numeric=read(BASE/'independent_delivery/source/NUMERICAL_FAILURE_CPU_AUDIT.json')
 summary=read(BASE/'independent_delivery/source/STOP_SOURCE_NUMERICAL_AUDIT.json')
 source=read(BASE/'independent_delivery/source/FROZEN_SOURCE_DEPENDENCIES_POST_STOP.json')
 resource=read(BASE/'independent_delivery/resource/RESOURCE_TERMINAL_AUDIT.json')
 archive=read(BASE/'independent_delivery/receipts/FINAL_ARCHIVE_STOP_AUDIT.json')
 assert numeric['checks_passed']==44 and len(numeric['frozen_compare_failure']['differences'])==231
 assert source['status']=='PASS_CURRENT_FROZEN_IDENTITIES_REAL_STOP_UNCHANGED'
 assert resource['actual_dispatches']==12 and resource['completed_B0_unit_PASS']==8 and resource['performance_dispatches']==0
 assert all(resource['all_own_resources_released'][k] is True for k in ['all12worker_and4controller_andwrapper_PIDs_absent','current_NVML_compute_apps_empty','current_all_four_GPU_UUIDs_match','current_task_cmdlines_absent','task_tmux_session_absent'])
 assert len(numeric['pair_finite_and_tensor_SHA_checks'])==468
 assert all(x['nonfinite_elements']==0 and x['tensor_SHA_matches_metadata'] for x in numeric['pair_finite_and_tensor_SHA_checks'])
 diffs={x['name']:x['result']for x in numeric['frozen_compare_failure']['differences']}
 assert diffs['logits']['max_abs_error']==0.53125
 OUT.mkdir()
 # Original CPU seal preserved byte-for-byte; old GPU STOP anchors stay separate.
 manifest=read(CPU/'SHA256_MANIFEST.json');seal=read(CPU/'FINAL_SEAL_RECEIPT.json')
 assert sha(CPU/'SHA256_MANIFEST.json')==seal['manifest_sha256']
 for row in manifest['files']:
  p=CPU/row['path'];assert p.stat().st_size==row['bytes']and sha(p)==row['sha256']
 for name in sorted({x['path']for x in manifest['files']}|{'SHA256_MANIFEST.json','SHA256SUMS.txt','FINAL_SEAL_RECEIPT.json'}):
  dest=OUT/'parent_cpu'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(CPU/name,dest)
 copytree(BASE/'repair_provenance',OUT/'prior_stop_anchors')
 # All downloaded assets are small source/metadata, already per-file verified.
 download=read(BASE/'independent_delivery/SMALL_ASSET_DOWNLOAD_VERIFICATION.json');assert download['status']=='PASS_ALL_DOWNLOADED_SMALL_ASSET_SHA_BYTES'
 copytree(BASE/'remote_evidence',OUT/'remote_evidence')
 for directory in ['analysis','qualification','operations','monitoring']:
  copytree(BASE/directory,OUT/'evidence'/directory)
 for name in ['ROOT.json','HANDOFF.md','USER_REQUEST.txt','HUMAN_GPU_AUTHORIZATION.json','FOLLOWUP_PROMPT.txt']:
  dest=OUT/'evidence'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/name,dest)
 for path in sorted((BASE/'independent_delivery').rglob('*')):
  if not path.is_file()or path.name=='make_stop_delivery.py' or path.suffix.lower()not in{'.py','.txt','.md','.json','.jsonl','.csv','.log'}:continue
  dest=OUT/'audits'/path.relative_to(BASE/'independent_delivery');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
 # Independent assembler itself is delivery control evidence, not a scientific source mutation.
 dest=OUT/'audits/scripts/make_stop_delivery.py';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(__file__),dest)
 rows=[]
 run=BASE/'remote_evidence/run'
 for n in range(4):
  units=[]
  for u in ['B0_FP_Q0','B0_FP_Q1']:
   d=run/('gpu'+str(n))/'execution'/u;r=read(d/'RESULT.json');ix=read(d/'numeric/NUMERICAL_INDEX.json')
   assert r['status']=='PASS'and r['consumed_tokens']==ix['tokens']==256
   assert ix['actual_bytes_compared']==(u=='B0_FP_Q1')
   units.append(dict(unit_id=u,status='PASS',tokens=256,actual_bytes_compared=ix['actual_bytes_compared'],tensor_metadata_digest=ix['all_tensor_digest'],runtime_library_preflight=r['runtime_library_preflight']['status'],runtime_library_postflight=r['runtime_library_postflight']['status']))
  assert units[0]['tensor_metadata_digest']==units[1]['tensor_metadata_digest']
  rows.append(dict(card='gpu'+str(n),B0=units,B0_fresh_determinism='PASS_BYTE_EXACT_SHORT256_TWO_FRESH_SAME_DEVICE',B1_Q0='FAIL_STRICT_BYTE_TOKEN0'if n==3 else'ABORTED_SHARED_STOP_NO_COMMITTED_COMPARISON_NO_RESULT',B1_committed_tokens=1 if n==3 else 0,B1_Q1='NOT_RUN',performance='NOT_RUN_NUMERICAL_PRECONDITION_FAILED',H_C128='NOT_QUALIFIED_FOR_THIS_CODEC',H_DUAL_M='NOT_QUALIFIED_FOR_THIS_CODEC'))
 put('NUMERICAL_PARITY_RESULTS.json',dict(task_id='QWEN_GDN_FAST_RUNTIME_QUALIFICATION_V1',execution_run_id=BASE.name,execution_status='FAIL_CLOSED_STOP',scientific_completion='INCOMPLETE_OR_ENGINEERING_STOP',candidate_decision='FAST_CANDIDATE_NUMERICAL_FAIL',cards=rows,earliest_divergence_token=0,earliest_observed_scalar=numeric['earliest_actual_different_scalar'],actual_different_tensors=231,saved_tensors_per_side=234,both_sides_all_finite=True,input_and_roster_equal=True,metadata_differences_scope='64 downstream cache SHA fields; actual numerical byte differences independently confirmed',FP_logits_parity='FAIL_GPU3_TOKEN0',FP_state_parity='FAIL_GPU3_TOKEN0',FP_GDN_intermediates_parity='FAIL_GPU3_TOKEN0',B1_fresh_determinism='NOT_ESTABLISHED_NO_TWO_COMPLETE_FRESH_RUNS',selected_errors={k:diffs[k]for k in ['GDN/0/q','GDN/0/updated_state','GDN/8/updated_state','logits']},full_per_tensor_error_report='audits/source/NUMERICAL_FAILURE_CPU_AUDIT.json',next_token_B1_consumption='NOT_RUN_STOP_AFTER_FIRST_TOKEN',scope='FP256 qualification terminated at first mismatch; no claim for other cards B1, codecs, AIME or long trajectories'))
 put('RESOURCE_LEDGER.json',resource)
 states=dict(FAST_STRICT_EQUIVALENT_VALIDATED='NOT_SATISFIED_STRICT_NUMERICAL_FAIL_AND_NO_PERFORMANCE',FAST_NEW_PROTOCOL_REPRODUCIBLE='NOT_SATISFIED_NO_B1_COMPLETE_FRESH_REPEATS_OR_SPEED',FAST_CANDIDATE_NO_BENEFIT='NOT_TESTED_NO_PERFORMANCE_RESULT',FAST_CANDIDATE_NUMERICAL_FAIL='APPLIES_GPU3_B1_TOKEN0',FAST_CANDIDATE_SAFETY_UNPROVEN='FILL_OPT_SAFETY_UNPROVEN_FOR_INDEPENDENT_FILL_HYPOTHESIS',CPU_ONLY_COMPLETE_GPU_NOT_AUTHORIZED='HISTORICAL_CPU_PHASE_ONLY_NOT_CURRENT',FAIL_CLOSED_STOP='CURRENT_EXECUTION_TERMINAL_STATUS')
 put('FAST_RUNTIME_DECISION.json',dict(task_id='QWEN_GDN_FAST_RUNTIME_QUALIFICATION_V1',execution_run_id=BASE.name,status='FAST_CANDIDATE_NUMERICAL_FAIL',execution_status='FAIL_CLOSED_STOP',scientific_completion='INCOMPLETE_OR_ENGINEERING_STOP',decision_categories=states,candidate='B1 CUBLAS_WORKSPACE_CONFIG=:16:8',candidate_executed=True,candidate_strict_equivalence='FAIL_OBSERVED_GPU3_FIRST_TOKEN',B1_self_repeatability='NOT_ESTABLISHED',candidate_performance='NOT_RUN',approved_Fast_codecs=[],FAST_STRICT_EQUIVALENT_VALIDATED=False,FAST_NEW_PROTOCOL_REPRODUCIBLE=False,fast_formal_launcher_created=False,fill_optimization='FILL_OPT_SAFETY_UNPROVEN',D3='NOT_RUN_FP_PRECONDITION_FAILED',no_new_GPU_retry_or_threshold_change_after_failure=True,not_a_negative_performance_result=True,internal_kernel_root_cause='UNKNOWN',other_three_cards_B1='NO_COMMITTED_COMPARISON_SHARED_STOP'))
 put('SOURCE_AND_AUDIT_BINDINGS.json',dict(task_id=source['task_id'],execution_run_id=BASE.name,status=source['status'],payload_files=604,common_control_files=77,per_card_execution_files=132,external_dependencies=241,prospective_bound_path_rows=32,prospective_physical_libraries=26,model_weight_shards=4,current_unique_physical_files_hashed=source['unique_physical_files_hashed'],actual_completed_runtime_selected_libraries=14,downloaded_small_files=download['files_verified'],downloaded_small_bytes=download['bytes_verified'],retained_remote_dependencies=download['retained_remote_dependencies'],raw_models_not_downloaded=True,current_disk_hash_not_retrospective_library_proof=True,source_audit='audits/source/FROZEN_SOURCE_DEPENDENCIES_POST_STOP.json',numeric_audit='audits/source/NUMERICAL_FAILURE_CPU_AUDIT.json',resource_audit='audits/resource/RESOURCE_TERMINAL_AUDIT.json',archive_audit='audits/receipts/FINAL_ARCHIVE_STOP_AUDIT.json'))
 historical=[]
 with(CPU/'PERFORMANCE_COMPARISON.csv').open(newline='')as f:
  for x in csv.DictReader(f):
   if x['STRICT_value']:historical.append(dict(metric=x['metric'],codec=x['codec'],value=x['STRICT_value'],unit=x['unit'],status='HISTORICAL_REFERENCE_ONLY_NOT_THIS_RUN',source=x['STRICT_source'],source_sha256=x['STRICT_source_sha256'],scope=x['scope'],current_run_measurement_status='NOT_RUN_NUMERICAL_GATE_FAILED'))
 assert len(historical)==6
 with(OUT/'PERFORMANCE_COMPARISON.csv').open('x',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(historical[0]));w.writeheader();w.writerows(historical)
 put('EXECUTION_PLAN.md','''本次为人类明确授权的独立修复执行，四卡可用、无GPU时间限额；原CPU封存和旧GPU身份STOP永久保留。修复仅补加载库闭包、前置PREFLIGHT及保留POSTFLIGHT，不改数值数学。

冻结每卡16有限单位：B0/B1各两fresh全256数值，全部四卡严格PASS后才进行128warmup+128measurement的三对fresh性能测量，逐卡全局互斥；原收益门槛medianB1>=1.05medianB0且minB1>maxB0，条件性Profile/H与独立D3后置。实际12单位派发：8条B0完整PASS，4条B1首fresh已派发；GPU3 token0不一致触发共享停止，其余三卡尚无已提交B1比较。B1第二fresh、性能、Profile、H、Dual-M均未执行。1800秒仅无真实进度watchdog，未因GPU时限或磁盘停止。没有新重试或放宽阈值。
''')
 put('STRICT_RUNTIME_CONTRACT.md','''参考为NEW_STRICT B0=:4096:8；候选B1=:16:8，均在Torch/CUDA初始化之前通过fresh进程设置。Qwen3.5-9B参数BF16、GDN state FP32、固定非AIME输入prefix256、seed20261015、deterministic/fill=True、matmul/cudnn TF32=False、Torch/OMP/MKL线程1/1。每个对照绑定同一原GPU UUID，逻辑cuda:0；四卡之间不互用字节参考。

四卡B0两fresh全256及全部保存logits/GDN/state/cache均逐字节一致，范围仅此短程FP基线。B1 GPU3 token0存在真实差异；其他三卡被共享停止，不判为数值失败。H和Dual-M未获得当前候选资格。32库路径/26实际库文件及241外部依赖当前SHA匹配，8个完整单位均有真实PREFLIGHT/POSTFLIGHT14库证据；失败B1只有PREFLIGHT，缺失POSTFLIGHT保留缺失，不用当前磁盘SHA追认它。逐份合同保存在remote_evidence/run/gpuN/analysis/STRICT_BASELINE_CONTRACT.json。
''')
 put('HISTORICAL_EVIDENCE_REUSE.md','''历史OLD_FP/NEW_FP中位TPS13.742036/7.776711，OLD_H/NEW_H为9.064610/6.650598；另一个Dual-M任务约6.801314154。旧新H存在prefill/cache协议差异。这些仅历史参考，不能拼成本轮同条件Codec因果对照。

原Profiler每token249个mm，CPU self约7.0764→55.8277ms、首次launch前6.8070→55.3913ms，而直接GPU GEMM约20.9284/20.9390ms；支持主机launch前开销方向，唯一内部根因仍UNKNOWN。本轮在数值门禁停止，新增可合法比较吞吐为0条。PERFORMANCE_COMPARISON.csv仅六条带原来源SHA的历史记录，没有把逐token保存/读回、CPU哈希或整体预留耗时当模型TPS。
''')
 put('CUBLAS_CONFIG_QUALIFICATION.md','''判定：FAST_CANDIDATE_NUMERICAL_FAIL；执行终态FAIL_CLOSED_STOP。B0在四卡各两个fresh完整256均逐字节相同。B1在GPU3首次fresh第0个输入已失败，其他三卡在模型加载或PREFLIGHT阶段共享终止，没有已提交数值比较；不是四卡都观察到B1差异。

双方同token、next-input、cache cursor、codec、GDN层表与234项dtype/shape/stride/offset/bytes roster均相同。CPU从29只读流式重查原始数据：231/234 Tensor实际不同，所有468侧Tensor有限且逐Tensor SHA匹配；64项非Tensor metadata差异为下游cache哈希。首个保存边界GDN/0/q的第0元素由0.043701171875(333d)变为0.044189453125(353d)，不能把该边界当成已定位的内部首个算术kernel。该q最大绝对差0.03125；state最大0.08163261413574219，logits最大0.53125。完整每Tensor maxabs/maxrel在数值审计JSON，relative分母沿用max(abs(reference),1e-300)，近零参考可产生极大相对值，并非nonfinite。

| 当前资格 | B0 | B1 |
|---|---|---|
| 完整FP轨迹 | 每卡2fresh×256 PASS | GPU3 token0失败；另3卡共享中止 |
| fresh确定性 | 同卡逐字节PASS | 未建立 |
| logits/state逐字节等价 | 自基线 | GPU3 FAIL |
| 单卡TPS/峰值/新Profile | NOT_RUN | NOT_RUN |
| H+C128、H+Dual-M | 未测试当前候选适用性 | NOT_QUALIFIED_FOR_THIS_CODEC |

数值优先门禁未放宽，未进入性能、Profiler、H或D3；不存在提速率或负吞吐结论，也没有稳定新数值协议的资格。内部GEMM/库kernel机制未定位。
''')
 put('UNINITIALIZED_MEMORY_SAFETY_AUDIT.md','''沿用原CPU安全审计，完整源码证据见parent_cpu/UNINITIALIZED_MEMORY_SAFETY_AUDIT.md及相关目录。历史新增847 kernel/token中724属fill类，但未证明全部由同一开关引起或可安全移除。generic resize增长、可选FLA/Flash masked tile及跨stream全写后读仍有覆盖缺口，维持FILL_OPT_SAFETY_UNPROVEN。

本轮deterministic=True、fill=True始终保留，未启动fill-off独立GPU假设。一次候选输出相同或不同均不构成未初始化读取安全证明。本轮数值差异不能据此归因为fill或证明关闭fill安全。
''')
 put('OPTIMIZATION_DIFF.md','''修复在新root/source/auth/prereg中进行，旧CPU和旧身份STOP不改。新增实际CPU导入/候选库闭包，补torchaudio与torchvision CUDA runtime等绑定，并在模型加载后任何科学forward之前运行PREFLIGHT，保留POSTFLIGHT和完整初末maps/全部缺失/SHA/bytes/inode证据；reducer核验三份回执。候选库清单不等同实际加载。

新科学路径与旧冻结worker的runtime、qualify、DeepRecorder、transform_core、replay及科学dispatch AST保持一致；数值reducer只增加库证据检查，数值差分/收益数学未改。B0/B1唯一被研究的变量仍cuBLAS工作区。四卡/无时间限额是已授权资源修订，固定名单/安全/阈值不变。此次B1真实数值失败后只进行独立CPU审计/归档/交付，没有GPU重试、更多配置、精度或算法更换。全部CPU工具检查期望更正历史保留。
''')
 lo=resource['current_run_worker_device_reservation_GPUh_lower'];hi=resource['current_run_worker_device_reservation_GPUh_upper'];cl=resource['cumulative_failed_plus_repair_worker_device_reservation_GPUh_lower'];ch=resource['cumulative_failed_plus_repair_worker_device_reservation_GPUh_upper']
 table='\n'.join('| GPU%d | 256 PASS | 256逐字节PASS | %s |'%(n,'token0真实不等，停止'if n==3 else'已派发、0提交、共享停止')for n in range(4))
 put('FINAL_SCIENTIFIC_REPORT.md',f'''本轮候选结论为 **FAST_CANDIDATE_NUMERICAL_FAIL**，执行终态 **FAIL_CLOSED_STOP / INCOMPLETE_OR_ENGINEERING_STOP**。B1小工作区`:16:8`在GPU3首个token与同卡Strict参考不逐字节相等，未获得Fast资格。四卡已释放；吞吐、Profile及Codec适用性后续因数值前置失败未启动。

| 设备 | B0首参考 | B0第二fresh | B1首fresh |
|---|---|---|---|
{table}

实际12次GPU单位派发，8条B0完整256轨迹PASS。每卡B0两次fresh的全部保存Tensor和cache元数据相同。B1仅GPU3提交token0并失败；其他三卡在模型加载或PREFLIGHT过程中共享停止，缺失RESULT及POSTFLIGHT保持缺失，不能判它们为数值错误。B1第二fresh、24个性能单位、8个条件Profile、16个H单位以及条件D3全部NOT_RUN。

首差异的证据：相同input/next-input/cursor及234项Tensor布局，CPU从29流读双方真实raw，231/234 Tensor不等；所有468侧Tensor均finite、逐Tensor SHA/bytes吻合。最早保存边界为GDN/0/q，第0个BF16值0.043701171875→0.044189453125，原始偏移8字节；q最大绝对差0.03125、最大相对差0.8992805755。首层state最大绝对差0.02884435654，全部保存GDN state最大0.08163261413574219（GDN/8/updated_state）；全248320词表logits最大绝对差0.53125、最大相对差4322.5555556。64个metadata差异是下游cache SHA；这不是仅metadata或文件身份问题。相对误差沿用冻结分母max(abs(reference),1e-300)，近零参考会产生极大数值，不代表原始Tensor有NaN/Inf。全部231项差分和精确字节值见audits/source/NUMERICAL_FAILURE_CPU_AUDIT.json。

原六个主要问题逐项回答：

1. **新版约7.78 TPS为何慢？** 历史Profiler仍支持mm主机首次launch前开销增大、直接GPU GEMM时间接近的方向；唯一内部根因UNKNOWN。本轮没有新增合格吞吐测量。
2. **`:16:8`是否提速？** 未知。候选在数值资格阶段失败，性能0条；不能认定更快、更慢或无收益。原三对重复、5%和严格不重叠门槛未被检验。
3. **deterministic fill能否安全优化？** 仍FILL_OPT_SAFETY_UNPROVEN，本轮fill=True，不批准关闭保护，也未启动第二项GPU假设。
4. **能否保持GDN/FP32 state/logits/Codec数学和确定性而提速？** 代码AST/数学保持不变，但该工作区候选在已测试GPU3不满足实际逐字节等价；不批准为Strict等价优化。B0短程fresh确定性通过，不代表32K/81K或所有Codec已验证。
5. **能否作为可重复高速新协议？** 不能。B1只一个token，没有两个完整fresh、自身重复性或合法收益证据；没有降级为FAST_NEW_PROTOCOL_REPRODUCIBLE。未测AIME准确率、长程Future-KL或质量影响。
6. **是否提供Fast启动器？** 没有；fast_validated_runtime.sh未生成。仅保留Strict CPU身份/环境检查入口和明确拒绝的候选元数据。实际GPU资格边界见本轮合同与回执。

库身份修复的范围：原缺库STOP永久保留，新root已补完整预绑定清单并前置检查。8个完整B0单位的PREFLIGHT/POSTFLIGHT均PASS，GPU3 B1的PREFLIGHT也PASS；其POSTFLIGHT因数值失败未执行，不补造。604 payload行、77 common、每卡132执行文件、241外部依赖、32绑定路径、4模型分片当前独立SHA/bytes均匹配。候选库26物理文件和实际每完整进程14选中库范围分开；当前磁盘身份不能替代退出前未保留的映射。最早保存差异位于GDN边界，不足以定位具体内部GEMM/kernel算法。

归档：29永久目录保留1,024参考raw共161,658,445,824 B，加1个失败raw 103,556,616 B，合计1,025文件/161,762,002,440 B，最终逐文件fresh SHA/bytes全PASS。四份ASSET_CATALOG各原子发布一次，4MiB工程探针单列；新目录无孤儿/tmp/未引用raw。实际物理目录1,030文件/161,766,473,722 B包含catalog与probe。所有模型、库二进制及大tensor留远端；下载3,027个小报告/JSON/源码等216,690,723 B，逐SHA/bytes核验，包含.h/.cpp/.cu/.cuh；46项模型/二进制依赖仅远端核验。旧资产及旧失败raw均未删除。

资源：12次派发、首次exit及首次release观测各唯一，冻结事件reducer与四卡终态账本一致。本修复执行worker/设备预留范围{lo:.12f}–{hi:.12f} GPUh，包含CUDA前CPU依赖/模型哈希；旧失败1.922697234406–1.923192716634 GPUh另列，累计{cl:.12f}–{ch:.12f} GPUh。它们是预留观察界，精确物理GPUh仍UNRESOLVED，不能当0或把预留叫计算耗时。所有本任务worker、controller、wrapper、tmux及NVML资源真实退出；停止不是磁盘或时间上限触发。同一55主机monotonic计时，未混用有偏移的主机UTC。B1失败/被杀进程的未保留actual argv明确NOT_RETAINED，保留expected dispatch与STARTED PID/starttick，不虚构历史观测。

历史速度仅作参考：OLD_FP/NEW_FP=13.742036/7.776711 TPS，OLD_H/NEW_H=9.064610/6.650598 TPS，另一Dual-M任务约6.801314154 TPS。PERFORMANCE_COMPARISON.csv仅收录有来源SHA的历史实测。本轮Strict与候选TPS、新Profile的mm CPU时间/首次launch前时间/kernel计数/性能窗口显存峰均NOT_RUN；模型加载NVML显存或约29分钟完整参考构造不是正式解码TPS/窗口峰值。

原第十四节七种状态：

| 状态 | 本轮适用性 |
|---|---|
| FAST_STRICT_EQUIVALENT_VALIDATED | 不满足，已观察严格数值不等且无性能资格 |
| FAST_NEW_PROTOCOL_REPRODUCIBLE | 不满足，B1自身完整重复性/收益未建立 |
| FAST_CANDIDATE_NO_BENEFIT | 未检验，不能宣称无收益 |
| FAST_CANDIDATE_NUMERICAL_FAIL | **适用：GPU3 B1 token0实际数值不等** |
| FAST_CANDIDATE_SAFETY_UNPROVEN | fill独立假设仍未获安全证明，本轮未测试 |
| CPU_ONLY_COMPLETE_GPU_NOT_AUTHORIZED | 仅原CPU阶段历史状态，本轮有明确授权和真实GPU派发 |
| FAIL_CLOSED_STOP | **当前执行终态：首次不等触发全局停止** |

这次给出严格等价资格的负结果，不能推断吞吐、AIME、长程质量或所有卡/Codec的方向。H/Dual-M当前候选均NOT_QUALIFIED_FOR_THIS_CODEC；未训练、未新建研究、未GPU重试、未改阈值或历史seal。原12项文件均已交付；SHA256_MANIFEST/SHA256SUMS和FINAL_SEAL_RECEIPT用于封存，外部LOCAL_DELIVERY_VERIFICATION记录Mac/.55/.29逐件同字节核验，避免循环SHA。父CPU151文件完整另存parent_cpu，旧GPU STOP锚点另存prior_stop_anchors。
''')
 put('runtime_presets/README.md','''没有合格Fast模式，不存在fast_validated_runtime.sh。Strict入口仅转交原CPU封存的身份/环境检查器，支持--check/--print-env，不启动任意GPU工作。本轮新增的B0短256 GPU资格记录独立保存，不能由CPU入口推断长程或Codec已通过。
''')
 put('runtime_presets/strict_runtime.sh','''#!/bin/sh
set -eu
exec /bin/sh "$(CDPATH= cd -- "$(dirname -- "$0")/../parent_cpu/runtime_presets" && pwd)/strict_runtime.sh" "$@"
''');(OUT/'runtime_presets/strict_runtime.sh').chmod(0o755)
 put('runtime_presets/runtime_identity.json',dict(status='FAST_CANDIDATE_NUMERICAL_FAIL',execution_status='FAIL_CLOSED_STOP',strict_environment='B0 :4096:8',B0_scope='TWO_FRESH_FULL256_FP_BYTE_EXACT_EACH_OF_FOUR_FIXED_DEVICES',entry_scope='CPU_SOURCE_AND_ENVIRONMENT_CHECK_ONLY_NO_GPU_DISPATCH',approved_Fast_codecs=[],candidate=':16:8',candidate_approved=False,original_CPU_manifest_sha256=seal['manifest_sha256']))
 assert not list(OUT.rglob('fast_validated_runtime.sh'))
 print(json.dumps(dict(status='REAL_NUMERICAL_STOP_REPORTS_ASSEMBLED_PENDING_REVIEW_AND_SEAL',indexed_candidate='FAST_CANDIDATE_NUMERICAL_FAIL',GPU_dispatches=12,full_B0_PASS=8,new_TPS='NOT_RUN')))
if __name__=='__main__':main()
