"""Read-only scientific audit and additive CPU delivery review after first STOP."""
from pathlib import Path
import hashlib,io,json,os,subprocess,sys,time,tarfile,datetime
D=Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1/execution_05');W=Path('/data/zypan/worktrees/ling-kda-prospective-execution05');E=D.parent/'engineering_repair_05';V=D/'delivery_review'
HEAD='902d7974c89465df882403e96cc8dcc8dc1c0e85'
STOP_SHA='390b6062be8ff2dca0ded48c9e9823477fc980460c0a535eee6d5851e56b588c';SEAL_SHA='214f57ff93ac0208018f23af0c666172f4d625db58521fa87d0b0dd0afc44a8f'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(n,j):
 with (V/n).open('x') as f:json.dump(j,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
assert not V.exists(),'ADDITIVE_REVIEW_ALREADY_EXISTS_NO_OVERWRITE'
assert sha(D/'STOP.json')==STOP_SHA and sha(D/'RESULT_AND_DELIVERY_HASHES.json')==SEAL_SHA
assert not (D/'PROSPECTIVE_COMPLETE.json').exists() and not list((D/'formal').glob('*'))
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()==HEAD
assert subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True)==''
freeze=read(D/'SOURCE_CODE_FREEZE.json')
for n,h in freeze['files'].items():assert sha(W/n)==h,n
seal=read(D/'RESULT_AND_DELIVERY_HASHES.json')
for n,v in seal['files'].items():assert sha(D/n)==v['sha256'] and (D/n).stat().st_size==v['bytes'],n
lines=(D/'SHA256SUMS.txt').read_text().splitlines()
for line in lines:
 h,n=line.split('  ',1);assert sha(D/n)==h,n
os.environ['LING_PROSPECTIVE_RESULTS']=str(D);sys.dont_write_bytecode=True;sys.path.insert(0,str(W))
from gpu_admission import proc_identity,classify,graphics_allowed
from qualified_completion import compare_exact
processes=[]
for pat,field in [('*.LAUNCH.json','worker'),('*.RUNNER.json','identity'),('LANE_LAUNCH_*.json','identity')]:
 for p in sorted(D.rglob(pat)):
  ident=read(p)[field]
  try:live=proc_identity(ident['pid'])==ident
  except (OSError,ValueError):live=False
  assert not live,p
  processes.append(dict(path=str(p),sha256=sha(p),identity=ident,same_identity_live=False))
assert subprocess.run(['tmux','has-session','-t','ling_prospective_execution05'],capture_output=True).returncode!=0
cp=read(D/'CHECKPOINT_MANIFEST.json')
for row in cp['checkpoints']:assert sha(row['path'])==row['sha256'] and Path(row['path']).stat().st_size==row['bytes']
qs=[]
for doc in ['pg19-test-41603','pg19-test-26183']:
 p=D/'runs'/f'{doc}__FP__parent.json';f=D/'runs'/f'{doc}__FP__resumed_parity.json';a=read(p);b=read(f);q=read(D/'qualification_units'/f'{doc}__FP.json')
 assert q['status']=='PASS';compare_exact(a,b,q['qualified_gpu_uuid'])
 profiles=[]
 for stage in ['parent','resumed_parity']:
  profile_path=D/'runs'/f'{doc}__FP__{stage}.RUNTIME_PROFILE.json';nat=read(D/'runs'/f'{doc}__FP__{stage}.NATURAL_EXIT.json');profile=read(profile_path)
  assert nat['runtime_profile_sha256']==sha(profile_path) and profile['status']=='PASS_ACTUAL_FROZEN_RUNTIME_PROFILE' and profile['forward_kernels_executed']
  profiles.append(dict(path=str(profile_path),sha256=sha(profile_path),actual_configs=profile['actually_used_forward_configurations']))
 qs.append(dict(document=doc,condition='FP',qualification='PASS',parent_sha256=sha(p),fresh_sha256=sha(f),complete64_full_V_logit_sha256=a['uninterrupted_suffix_full_logit_sha256'],final_native_cache_sha256=a['final_full_native_cache_sha256'],checkpoint_sha256=a['checkpoint_sha256'],parent_and_fresh_same_target_GPU=q['qualified_gpu_uuid'],observer_ON_OFF_exact=True,FP_audit_state='NOT_APPLICABLE_FP_NONE',INT8_semantics='NOT_APPLICABLE_FP_NO_INT8_STUDENT_QUALIFIED_THIS_EXECUTION',actual_runtime_profiles=profiles))
window=D/'admission/pg19-test-26183__H_to_L6_a50__parent.window.jsonl';rows=[json.loads(x) for x in window.read_text().splitlines()];last=rows[-1]
assert len(rows)==13 and [classify(x) for x in rows]==['AVAILABLE_VERIFIED']*12+['TELEMETRY_INCONSISTENT']
assert last['target'][0]['GPU_util']=='1' and last['direct_NVML']['utilization']['gpu']==0
assert graphics_allowed(last) and not last['direct_NVML_processes']['compute']['processes']
stop=read(D/'STOP.json');status=read(D/'FINAL_MACHINE_STATUS.json');res=read(D/'RESOURCE_REPORT.json');selection=read(D/'FROZEN_4DOC_SELECTION.json')
assert status['ENGINEERING_COMPLETE'] and not status['NUMERICAL_QUALIFICATION_COMPLETE'] and not status['FORMAL_SCIENCE_COMPLETE'] and status['QUALIFIED_CONDITIONS']==2
assert res['actual_GPU_worker_count']==7 and res['natural_exit_worker_count']==6 and res['failed_exit_worker_count']==1
assert sum(x['GPU_hours'] for x in res['stages'].values())==res['GPU_hours_new_execution_05']
V.mkdir()
write('TERMINAL_PROCESS_AUDIT.json',dict(status='ALL_OWN_GPU_WORKERS_RUNNERS_AND_LANES_EXITED',epoch=time.time(),tmux_session_absent=True,identities=processes,no_new_GPU_worker=True))
write('FINAL_EVIDENCE_AUDIT.json',dict(status='PASS_ACTUAL_PARTIAL_ASSETS_NO_FORMAL_SCIENCE',first_STOP_sha256=STOP_SHA,original_final_manifest_sha256=SEAL_SHA,source_commit=HEAD,source_clean=True,source_file_SHA_count=len(freeze['files']),original_small_files_SHA_verified=len(seal['files']),original_SHA256SUMS_records_verified=len(lines),actual_qualifications=qs,actual_checkpoints=cp['checkpoints'],formal_FP=0,formal_students=0,valid_documents=0,actual_primary_positions=0,sum384_verification='NOT_APPLICABLE_NO_STUDENT_TRAJECTORY',primary_KL_and_Delta='NOT_RUN',INT8_qualification_this_execution='NONE_COMPLETED',first_blocker=dict(stage='GPU3_NEW_WORKER_RESOURCE_ADMISSION',window_path=str(window),window_sha256=sha(window),sample_count=13,classifications=[classify(x) for x in rows],last_sample_epoch=last['epoch'],SMI_GPU_util_percent=1,NVML_GPU_util_percent=0,SMI_used_MiB=last['target'][0]['used_MiB'],NVML_used_bytes=last['direct_NVML']['memory_v2']['used'],compute_processes=[],graphics_identity_unchanged=True,underlying_cause='UNKNOWN_NOT_INFERRED_FROM_ONE_NON_ATOMIC_SAMPLE'),secondary_abort='GPU2 pg19-test-41603/H_to_L7_a50 parent exited SAFE_TOKEN_STOP after GPU3 global first STOP; not an independent parity or quantization failure',no_gate_relaxation=True,no_retry=True,scientific_files_changed=[]))
write('DELIVERY_REVIEW_SCOPE.json',dict(status='ADDITIVE_CPU_REVIEW_ONLY_ORIGINAL_SEAL_UNCHANGED',epoch=time.time(),review_source_path=str(Path(__file__)),review_source_sha256=sha(Path(__file__)),reason='Clarify actual two qualifications are FP only; all INT8 and full32K results remain incomplete. Distinguish primary GPU3 admission failure from GPU2 safe abort. Preserve original root report/manifest and first STOP byte-for-byte.',original_manifest_sha256=SEAL_SHA,original_STOP_sha256=STOP_SHA,metadata_only=True,GPU_scientific_units=0))
phase0=read(D/'PHASE0_HISTORICAL_ASSET_AUDIT.json');oldrows='\n'.join(f"|{Path(x['path']).name}|`{x['manifest_sha256']}`|PASS|" for x in phase0['manifests'])
pairs='\n'.join(f"|{x['document']}|{x['P1_predicted_rotation']}|{x['L7_predicted_rotation']}|{x['P1_gap']:.17g}|{x['L7_gap']:.17g}|NOT_RUN|NOT_RUN|NOT_RUN|" for x in selection['documents'])
matrix=read(D/'QUALIFICATION_MATRIX.json')['units'];mrows=[]
for row in matrix:
 d,c=row['document'],row['rotation'];state='PASS' if row['qualification_status']=='PASS' else 'NOT_RUN'
 if d=='pg19-test-41603' and c=='H_to_L7_a50':state='ABORTED_BY_GLOBAL_STOP'
 if d=='pg19-test-26183' and c=='H_to_L6_a50':state='ADMISSION_STOP_WORKER_NOT_LAUNCHED'
 mrows.append(f"|{d}|{c}|{state}|NOT_RUN|")
cptext='\n'.join(f"- `{x['path']}`；{x['bytes']} bytes；SHA `{x['sha256']}`。" for x in cp['checkpoints'])
local='/Users/pytenter/Documents/临时项目1/ling_prospective_extension_4doc_v1/engineering_repair_05/execution_05/delivery/'
stop_time=datetime.datetime.fromtimestamp(stop['epoch'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
report=f'''# Ling/KDA execution_05 最终交付复核（STOP）

科学ID：`LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1`。本报告为终态后的附加CPU复核，原`FINAL_SCIENTIFIC_REPORT.md`、根manifest、源码、首STOP和科学资产均保持原字节。

|状态|实际结果|
|---|---|
|ENGINEERING_COMPLETE|true：CPU启动/配置检查及双GPU canary通过|
|NUMERICAL_QUALIFICATION_COMPLETE|false：2/12，且仅FP条件|
|FORMAL_SCIENCE_COMPLETE|false：FP 0/4，student 0/8，有效文档0/4|
|最终执行状态|FAIL_CLOSED_STOP，未重试，未自动启动后续执行|

## A. 原资产及修复身份

预注册SHA `54d6d24fbc3142a692ef1a6c04a5617cf6b0abe6702a662722f34fc1b6141722`，四篇共同freeze SHA `a5c227d15918a249f8ae2059539c15f94668cb2afddf3aecd8a9f7e13851354b`；FROZEN_SPEC SHA `8ae8927ed37209efccbf19c672b9baeb46c0d7f272601f8f54d1f9609b64f3d1`。原freeze北京时间2026-10-14 10:44:26，早于新长程与任何本轮32K outcome（实际32K未运行）。原100→20历史排除/22不足长度/58资格顺序、实际4篇×16旋转=64评分、固定pair、所有原source/model/input/rotation身份均复用，未择样或改公式。Phase0重新验证历史manifest、逐文件SHA及其SHA清单：

|资产|manifest SHA|核验|
|---|---|---|
{oldrows}

execution01/02/03/04及旧MIXED/posthoc历史只读保留。独立修复commit `{HEAD}`，基于`d3cd88d4643a460e027cae6ba6989628c21f60bf`。源码冻结115文件含6个原FLA JSON和provenance，98个未涉及原Python文件逐字节不变；数学worker、helper、量化器、rotation、KL/reducer及容差未改，资格比较AST与CPU Primary校验原字节保持。变更范围为launcher/env/envelope/退出监督、测试与CPU交付、worker计划计数事件；diff及SHA见`SOURCE_DIFF.patch`、`NUMERICAL_SOURCE_COMPATIBILITY_AUDIT.json`。

根因定位进行了5个预先有界诊断（科学单元0）。同GPU、同进程、同checkpoint/RNG/observer的A→B→A干预证明conv配置BD128/warps4直接产生旧失败fingerprint、原BD8/warps8产生旧parent fingerprint、回A再复现。旧execution04当时实际kernel配置未捕获仍UNKNOWN。修复恢复原冻结6个FLA配置的`FLA_CACHE_MODE=default`及明确配置目录，并保留解释器启动前的`PYTHONHASHSEED=0`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`LING_GATED_RMSNORM_FIXED_CONFIG=BT16_W8`；没有删initializer断言或在worker内补seed。

37项工程检查、10个真实launcher→master/lane→unit_runner→冻结initializer CPU用例（4正6负）、11项profile/envelope检查及4项交付身份检查均PASS，无模型/CUDA。旧CPU fixture最初两个过期元数据/路径错误的FAIL日志保留，修正只涉及测试fixture身份与显式历史fixture路径，不是科学重试。

## B. GPU与首个停止门禁

仅GPU2 `GPU-7d6246ed-c3d1-75bc-c2c7-d02eb2882cca`和GPU3 `GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae`。两卡canary实际环境、UUID、冻结initializer、真实6项FLA loader与正常退出均PASS，canary无模型/token，不算数值资格或32K科学轨迹。所有worker仍单独经过原60秒13点/final3严格0%util/allocated≤64MiB及SMI/NVML/进程归属门禁；Xorg约4MiB的PID4040/UID120/starttime3403显式绑定。

首STOP时间：{stop_time}。GPU3准备启动`pg19-test-26183 × H_to_L6_a50 parent`，13个采样中前12个AVAILABLE_VERIFIED，最后一个SMI利用率**1%**、直接NVML利用率**0%**，原分类器判`TELEMETRY_INCONSISTENT`，也未满足原双零利用率条件。显存SMI15MiB与NVMLv2 15400960 bytes差0.3125MiB，内存算术一致，compute进程列表为空，已知图形身份未变；不能据此解释不一致的底层原因，该原因仍UNKNOWN。没有重复采样来覆盖失败、没有降低阈值或自动重试。原窗口SHA `e6e760e9c98f800323db92dd70c0f9b2e8152932b278a64855887bed8c815d5a`，首STOP SHA `{STOP_SHA}`。

GPU2当时加载中的`41603 × H_to_L7_a50 parent`随后在安全token边界读到全局STOP并退出`SAFE_TOKEN_STOP`；这是首资源失败后的联动中止，不是额外观测到的INT8数值或resume失败。该GPU3条件未实际launch，不算一个GPU worker。所有本轮GPU worker、runner和lane身份已退出，tmux已结束，未杀他人或reset GPU。可读取驱动/host namespace的历史限制仍UNKNOWN，不推断硬件故障。

## C. 资格与checkpoint复用

原10个CP实际SHA保留，但旧完整0→8192 FLA配置缺乏运行记录，因此本轮计划在原12科学key下新建12 parent+12fresh；没有继承旧GPU6/04 PASS。实际只完成**2 parent+2fresh64**，均为FP：41603/GPU2和26183/GPU3。它们通过全部64步完整词表logits、最终native cache、同checkpoint及恢复状态/RNG/observer ON/OFF的原严格合同；FP audit_state=None是N/A，不是INT8 QDQ证明。4个实际model worker均有实际conv BD8/warps8和norm BT16/warps8的post-forward profile、/proc启动环境及NATURAL_EXIT绑定。包括此前26183/FP的失败恢复路径，本轮实际exact PASS；不能把两个FP通过扩展为8个INT8条件通过。

**本轮没有完成任何INT8条件资格**；persistent QDQ写回/下一kernel消费的当前整组资格仍INCOMPLETE，历史结果只保持历史身份。

|文档|条件|实际资格阶段状态|正式32768|
|---|---|---|---|
{chr(10).join(mrows)}

## D/E. 正式轨迹和Primary

4FP/8student均未运行，12条正式轨迹均NOT_RUN，无实际384位置Primary，无sum/384校验对象，不生成空壳科学CSV或把未运行项填0。固定原文档顺序及每篇FP→P1→OriginalL7保持。

|文档|P1旋转|Original L7旋转|P1 midrank gap|L7 midrank gap|K_P1|K_L7|Delta|
|---|---|---|---:|---:|---|---|---|
{pairs}

原评分合同仍为V157184、t=8192+64k (k0..383，末32704)、消费ids[t]后预测ids[t+1]、原FP32 KL(FP||student)、原Python sum/384，Delta=K_P1−K_L7。阈值3.3300064494019496e-6只保留为原经验分辨率，本次没有实际新环境Primary来检验它。P1/L7/tie-unresolved三个已观测计数均0，**缺失4**，不能称四个平局。没有新的独立前瞻科学方向证据，未复现或否定旧MIXED/posthoc候选选择现象。document为独立单位，不并n6，不将384点/层/head伪重复，不做confirmatory p。E_future/Hidden KL/new proxy不计算，未导出的layerwise指标不新造。

## F. 资源、模型与交付

本轮science execution05实际GPU worker **7**：canary2、FP parent2、FP fresh2、全局STOP中止的student parent1；自然退出6、失败退出1。物理admission未launch的条件不计worker。按实际launch-to-exit卡小时求和，含加载/退出，排除CPU/准入/等待，并行卡分别计入。

|stage|实际GPUh|自然退出|失败退出|
|---|---:|---:|---:|
|runtime canary|{res['stages']['runtime_canary']['GPU_hours']:.17g}|2|0|
|parent（含中止）|{res['stages']['parent']['GPU_hours']:.17g}|2|1|
|fresh64|{res['stages']['resumed_parity']['GPU_hours']:.17g}|2|0|
|本轮新增合计|{res['GPU_hours_new_execution_05']:.17g}|6|1|

历史分账：execution01 2.268573776549763、execution02 3.611504961715804、execution03 0.003053035471174452、execution04 0.5042779713869094、5个诊断0.42219087839126584 GPUh；全部尝试合计{res['GPU_hours_all_execution_attempts']:.17g} GPUh，未把历史/诊断算为本轮科学成功。

本轮已测CUDA allocated峰值16486794240 bytes、reserved峰值20128464896 bytes，为原loop-reset scope；中止加载worker没有完整loop峰值记录，不声称覆盖整个生命周期。NVML owned定期样本38个，采样峰值19536MiB，非连续瞬时峰值。5诊断未测CUDA reset峰值为UNKNOWN。未施加人为GPUh或显存百分比硬上限。

两个真实own-history8192 checkpoint（大型文件留服务器，不下载）：

{cptext}

它们的完整实际FPprefix均重新计算，不是旧CP软链接，完整文件SHA恰与历史对应CP一致；这个相等只证明这两个实际已复算FP条件，不外推其他条件。正式FP payload数量0；没有虚构full32K参考。模型权重仍服务器原路径，由`MODEL_WEIGHT_SHA256.txt`/原spec与source资产绑定。

服务器：`{D}`；冻结工作树：`{W}`；本地全部小型封存文件：`{local}`。原根manifest SHA `{SEAL_SHA}`，283个原封存小文件/284条SHA记录（含manifest自身）已复核；本次新增复核报告、终态/证据审计及退出后稳定日志另以`delivery_review/DELIVERY_REVIEW_HASHES.json`和`DELIVERY_REVIEW_SHA256SUMS.txt`封存，原根manifest/SHA/STOP保持不变。下载后按两份清单逐项本地验证，详见本地`LOCAL_FINAL_DELIVERY_VERIFICATION.json`。

## G. 已保留资产与停止边界

首blocker是GPU3资源遥测门禁，非新观察到的resume数值失败或P1/L7胜负。可保留原64评分/四文档freeze、全部历史身份、原数学source、诊断因果证据、修复commit，以及本轮两个具完整启动profile和fresh64 exact证据的FP checkpoint/资格。未完成条件不继承PASS。资源门禁失败不能靠复用这些资产免除；后续若恢复，须独立执行身份、适用的新启动授权和原物理/数值门禁，不能在本STOP目录续跑。本次没有开始下一轮。

## H. 科学建议与15问

当前仅能说已修复的运行配置在两个FP条件上通过真实恢复资格，不能声称persistent INT8/P1优于Original L7、总体优势、可训练性或reasoning/AIME改善。下一步先解决并证实遥测稳定性与可归属，再讨论在同冻结问题下恢复未完成条件；这是建议，未执行新GPU诊断、训练或科学条件。

1. **读了哪些预注册/成功manifest，SHA是否通过？** 原NEXT_PROSPECTIVE_4DOC_PREREGISTRATION.md及正式协议freeze、原成功MIXED与execution01/02/03/04封存均读取并SHA PASS，详见A与Phase0。预注册与四文档完整SHA如A。
2. **100→58输入资格完全复现？** 历史真实retokenization的100/20/22/58、raw/token SHA与顺序保留并复核；本轮未再编码或择样。不能把历史复现冒称本轮重新tokenization。
3. **实际短程筛多少篇，是否≤40？** 原实际4篇，每篇16旋转，共64分数，在最多40篇内选齐四篇；本轮新增短程scoring0，合法复用64原分数，未重筛。
4. **最终四篇？** 41603、2544、7987、26183。
5. **P1/L7旋转与midrank gap？** 逐篇完整映射见D/E表（6/4、7/6、6/4、5/6），全部分数/ranks原样封存。
6. **freeze早于32K outcome？** 是，原freeze早于任何新长程；本轮正式32K没有开始，原receipt字节未改。
7. **所有门禁通过？** 否。启动工程/双canary通过；数值/observer/fresh资格只有两个FP PASS，INT8资格未完成；GPU3后续资源门禁FAIL。不能称12/12或persistent INT8整组合格。
8. **4FP+8student全部完成？** 否，正式FP0/4、student0/8、valid0/4；两个8256资格parent不计完整32K。
9. **四篇完整精度K_P1/K_L7/Delta？** 全部NOT_RUN，无合法Primary值，不填0。
10. **P1/L7/tie-unresolved各多少？** 已观测胜负计数0/0/0，缺失4；不是四个平局。
11. **有方向性独立前瞻证据？** 没有；新完整文档科学n=0，不能宣称P1增量预测价值或稳定优势。
12. **复制旧MIXED/Posthoc的候选选择现象？** 未检验，当前无新32K outcome，既不支持也不否定；旧MIXED/posthoc永久保留、不并n6。
13. **真实worker/GPUh/峰值？** 7 worker、{res['GPU_hours_new_execution_05']:.17g}新增GPUh；历史分账、CUDA16486794240/20128464896 bytes与38样本NVML19536MiB及其scope见F。无退出时段UNKNOWN未伪装0，本轮所有launch均已有退出receipt。
14. **最终commit/SHA/manifest/绝对目录？** commit `{HEAD}`；首STOP `{STOP_SHA}`；原root manifest `{SEAL_SHA}`；本地绝对目录 `{local}`；新增review清单单独封存且SHA包含其manifest自身。
15. **严格STOP，无训练/L8/L9/AIME？** 是，全部自有worker/runner/lane退出，tmux结束；无梯度、新Loss/L8/L9/STE/Cayley/optimizer/AIME/free generation/Qwen或下一执行。交付后暂停本跟进。

```json
{json.dumps(status,ensure_ascii=False,indent=2)}
```
'''
with (V/'FINAL_DELIVERY_REVIEW.md').open('x') as f:f.write(report)
# Preserve original seal. Supplemental manifest binds added review and terminal logs.
extra=['RESULT_AND_DELIVERY_HASHES.json','SHA256SUMS.txt','controller.log']+[str(p.relative_to(D)) for p in D.glob('lane_GPU-*.log')]+[str(p.relative_to(D)) for p in V.iterdir() if p.is_file()]
extra=sorted(set(extra));files={n:dict(bytes=(D/n).stat().st_size,sha256=sha(D/n)) for n in extra}
write('DELIVERY_REVIEW_HASHES.json',dict(status='SEALED_ADDITIVE_CPU_REVIEW_FIRST_STOP_PRESERVED',original_manifest_path=str(D/'RESULT_AND_DELIVERY_HASHES.json'),original_manifest_sha256=SEAL_SHA,original_files_verified=283,first_STOP_sha256=STOP_SHA,source_commit=HEAD,files=files,self_exclusion='supplement excludes itself and supplemental SHA sums; supplemental SHA sums includes its own manifest'))
supp=V/'DELIVERY_REVIEW_HASHES.json'
with (V/'DELIVERY_REVIEW_SHA256SUMS.txt').open('x') as f:f.write(''.join(v['sha256']+'  '+n+'\n' for n,v in files.items())+sha(supp)+'  delivery_review/DELIVERY_REVIEW_HASHES.json\n')
assert sha(D/'STOP.json')==STOP_SHA and sha(D/'RESULT_AND_DELIVERY_HASHES.json')==SEAL_SHA
all_files=set(seal['files'])|set(files)|{'delivery_review/DELIVERY_REVIEW_HASHES.json','delivery_review/DELIVERY_REVIEW_SHA256SUMS.txt'}
archive=E/'EXECUTION05_FINAL_DELIVERY.tar.gz'
with tarfile.open(archive,'x:gz') as t:
 for n in sorted(all_files):
  info=tarfile.TarInfo(n);info.size=(D/n).stat().st_size;info.mode=0o644
  with (D/n).open('rb') as raw:t.addfile(info,raw)
print(json.dumps(dict(status='FIRST_STOP_AND_ORIGINAL_SEAL_PRESERVED_CPU_REVIEW_PACKAGED',original_files=283,original_SHA_records=284,supplement_manifest_sha256=sha(supp),supplement_files=len(files),total_delivery_files=len(all_files),archive_path=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=sha(archive))))
