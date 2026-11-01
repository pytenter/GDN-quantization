# Ling/KDA Phase B STOP 只读取证审计

最终判定：`ELIGIBLE_FOR_ISOLATED_REPAIR`。这是隔离工程修复候选，**不是新GPU实验许可，也不是长期科研确认**。本轮先制定审计计划，随后读取真实服务器资产、静态源码与git记录，最后在全新目录交付；未执行修复。

## 1. 冻结科学选择恢复与资格

| 本轮新文档 | 固定设备 | frozen pair | P1 predicted winner | L7_TOTAL predicted winner | 双方midrank gap |
| --- | --- | --- | --- | --- | --- |
| pg19-test-860 | GPU6 | H_to_L6_a50 vs H_to_L7_a25 | H_to_L6_a50 | H_to_L7_a25 | 8 / 5 |
| pg19-test-5956 | GPU3 | H_to_L6_a50 vs H_to_L7_a50 | H_to_L6_a50 | H_to_L7_a50 | 6 / 5 |

A历史Phase A=37702/5396；B既有P1-vs-B0=48693/51410；C本次真正冻结Phase B=860/5956，互不混用。冻结输入与prior path registry明确排除A/B及其他旧文档。文档内容raw/token SHA均重验MATCH，合法token数各32769。数据身份为本地`phase3_mixed_hardware_v1_20261003` PG19 test快照路径+内容SHA；未在freeze发现上游dataset revision，不以目录日期虚构Hub版本。旧exact-tokenizer-reencoding证据已重哈希；本轮未重tokenize。

Route A保持P1 vs原L7_TOTAL。窗口起点256/1024、各H128，对应[256,384)、[1024,1152)，全18KDA/all16heads；两窗等权平均、lower safer。完整16 roster未改，每个rotation身份见selection audit。原L7本就H128 recurrent-aware；原helper定义保留，matched deployment tensor应用并非历史512-token训练采样完整重演。P1采用已冻结query-readout energy定义，本轮没有重新发明指标。

独立CPU检查：672个原score cell逐一与既有两篇canonical匹配；672个midrank cell独立重算MATCH；两文档全部eligible pair列表及固定min-gap/lexicographic优先顺序完全匹配原freeze，未重新执行selector或发布新selection。原始raw分数、每窗和汇总完整排序、FP tensor身份、roster/source身份保留在`PHASEB_FROZEN_SELECTION_AUDIT.json`。

### pg19-test-860：两窗平均分与完整16条件midrank

| rotation | P1 | P1 rank | L7_TOTAL_ORIGINAL | L7 rank |
| --- | ---: | ---: | ---: | ---: |
| DA_fromH | 5.7061138591940619e-09 | 2 | 0.0084295608003230882 | 2 |
| DA_fromL6 | 5.1541323431436581e-09 | 1 | 0.0078832727886037901 | 1 |
| H | 1.4696237116140765e-08 | 13 | 0.013484003622579621 | 12 |
| H_to_L6_a25 | 1.4769278688930854e-08 | 14 | 0.014350378851304413 | 15 |
| H_to_L6_a50 | 1.2768224921799742e-08 | 7 | 0.013725893895752961 | 14 |
| H_to_L7R_a25 | 1.4191393171714139e-08 | 11 | 0.013619372697576182 | 13 |
| H_to_L7R_a50 | 1.3492236883294595e-08 | 8 | 0.012819274301364203 | 7 |
| H_to_L7_a25 | 1.5341870218676945e-08 | 15 | 0.013326732407222153 | 9 |
| H_to_L7_a50 | 1.4114519109043044e-08 | 10 | 0.013434682045954105 | 11 |
| H_to_L8_a25 | 1.4606443610176711e-08 | 12 | 0.013389291243584012 | 10 |
| H_to_L8_a50 | 1.3726484171527886e-08 | 9 | 0.012956366109392548 | 8 |
| IDENTITY | 3.7680650599725141e-08 | 16 | 0.055053291813237593 | 16 |
| L6 | 6.2558942470758439e-09 | 3 | 0.0090383795104571618 | 3 |
| L7 | 1.0157878316263691e-08 | 6 | 0.011202977927496249 | 5 |
| L7R | 9.221593044728138e-09 | 4 | 0.01098628251838818 | 4 |
| L8 | 1.0025544172620471e-08 | 5 | 0.011349933613928442 | 6 |

### pg19-test-5956：两窗平均分与完整16条件midrank

| rotation | P1 | P1 rank | L7_TOTAL_ORIGINAL | L7 rank |
| --- | ---: | ---: | ---: | ---: |
| DA_fromH | 5.0551423047551225e-09 | 3 | 0.0095641291445645038 | 2 |
| DA_fromL6 | 4.2002950562114449e-09 | 1 | 0.0093024887573847082 | 1 |
| H | 9.8782513280326611e-09 | 15 | 0.014447442254095222 | 13 |
| H_to_L6_a25 | 9.1244294342374133e-09 | 10 | 0.014245397471313481 | 12 |
| H_to_L6_a50 | 8.6329392523509796e-09 | 8 | 0.015220823661366012 | 15 |
| H_to_L7R_a25 | 9.5276018186041256e-09 | 13 | 0.014145786801236682 | 11 |
| H_to_L7R_a50 | 8.4185727278196509e-09 | 7 | 0.013821893392560014 | 9 |
| H_to_L7_a25 | 9.2142482532864278e-09 | 11 | 0.013764691439064336 | 8 |
| H_to_L7_a50 | 9.7765411322114915e-09 | 14 | 0.014077574472139531 | 10 |
| H_to_L8_a25 | 9.4766521296918427e-09 | 12 | 0.014462023445958039 | 14 |
| H_to_L8_a50 | 8.6909812679891729e-09 | 9 | 0.013565035564170103 | 7 |
| IDENTITY | 2.6019603804883218e-08 | 16 | 0.046418354482739232 | 16 |
| L6 | 4.8909623018289494e-09 | 2 | 0.010087628483233857 | 3 |
| L7 | 7.2776946691277544e-09 | 6 | 0.011952991736507101 | 5 |
| L7R | 6.502894112081492e-09 | 4 | 0.011384183869267872 | 4 |
| L8 | 6.7738512576909216e-09 | 5 | 0.012228089063683001 | 6 |

## 2. Outcome leakage审计

结论：`NO_OUTCOME_LEAKAGE_FOUND_IN_AUDITED_ASSETS`。pair methods冻结于2026-10-11T14:05:19.555756+08:00，早于正式calibration；selection于2026-10-11T14:41:21.189710+08:00，晚于两篇完整calibration且早于persistent launch。selector静态数据入口只有短calibration、rules/route、input/source身份及账本，不读新长程标签；文档hash order/seed0及prior exclusions一致。

两qualification在stream前失败，`runs_b_persistent`无canonical/micro/formal结果，完整cost gate不存在，新增32K FP/labels/science units均0。这与freeze声明new_long_outcomes_generated_or_read=false一致。该结论限于审计资产，不能声称证明所有未记录的人为访问历史；不使用既有B组长程KL来选择C组pair。

## 3. 首个实际qualification failure，不按终态猜测

首失败：`GLOBAL_TORCH_DETERMINISTIC_ALGORITHMS_ENABLED_ASSERTION`，GPU6/860，worker第63行：

```python
assert torch.are_deterministic_algorithms_enabled()
```

预期True；实际False由原失败的boolean assert与installed getter源码推定，原日志未独立打印flag值。STARTED=2026-10-11T15:08:30.057384+08:00，STOP=2026-10-11T15:08:46.404388+08:00。对应原STOP：`/data/zypan/experiment_runs/LING_KDA_L7_OBJECTIVE_REDUNDANCY_AND_PERSISTENT_VALIDATION_V1/runs_b_persistent/qualification_pg19-test-860_gpu6.STOP.json`；SHA=`b9f31ce9ac844fc069f1222106cdb03d335effac338b054d0c7c0cc948585809`。

```text
Traceback (most recent call last):
  File "/data/zypan/experiment_runs/LING_KDA_L7_OBJECTIVE_REDUNDANCY_AND_PERSISTENT_VALIDATION_V1/run_persistent_stage_v1.py", line 63, in main
    assert torch.are_deterministic_algorithms_enabled()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
```

GPU3/5956随后于2026-10-11T15:08:46.958776+08:00同一行失败。两log都显示32/32模型shard完成加载，随后同一traceback；model-type/deprecation/generation-flags警告不是首个FAIL。实际加载完成不等于执行了模型forward。

顺序：CPU freeze/input/device/900s完整资格roster reservation→LAUNCH_AUTHORIZATION→worker verify/receipt/deadline→STARTED→GPU runtime/UUID独占assert→peer/deadline guard→load assets/inputs→Capture模型加载→globalflag FAIL→exclusive STOP→共享latch。各中间门禁只可由原控制流证明已返回，缺少独立精确timestamp；本报告不伪造逐gate时间或把它们写成完整GPU资格PASS。原记录的细节、两卡事件顺序、source lines、原traceback见`FIRST_FAILURE_GATE_REPORT.json`。

TF32/CUBLAS后续assert及第一token尚未到达；真实512-token replay/full-logit parity、persistent QDQ/writeback消费、hybrid cache数值、finite/repeat、8448micro/32K都未执行。此前Phase A/B短程GPU PASS仍是原阶段证据，不替代新persistent资格。晚于首STOP的guard exception是已存在latch所致，不是首因；两.partial latch保留；预GPU CPU构建错误也单独保留，不混为这次首个qualification门禁。

分类：工程runtime配置/guard配套前置条件不匹配。没有证据把它归为测得的数值不确定性、QDQ科学失败、OOM、硬件异常或预算预测失败。worker无显式enable后又要求global flag为True，固定FLA selector/seed/CUBLAS配置并不等价于这个flag。可提出小范围显式runtime初始化与telemetry修复，但本轮不改源码、不移除assert、不清STOP。

## 4. 当前资源、已发生开销及预算边界

只读快照时间=2026-10-11T22:58:06.433168+08:00。GPU3=`GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae`；GPU6=`GPU-de53f243-b271-521c-d875-d96611ed8574`。两张RTX3090各24576MiB，当前15MiB/0%util，nvidia compute-apps为空且fuser无holder。旧PID341582/341594及对应tmux已退出。空闲是时间点事实，不是独占预约。

当前kernel=`7.0.0-38-generic`；driver=`580.178.04`；Torch=`2.7.1+cu128`；CUDA runtime distribution=`12.8.57`；fla-core=`0.5.2`；Transformers=`4.57.6`；Triton=`3.3.1`。版本来自CPU metadata/文件及nvidia，不importGPU库；当前Torch compile/runtime通过+cu128/metadata间接核对，未做forward资格。所有冻结依赖SHA匹配不等于未绑定runtime文件全体已资格。reboot-required/shutdown-schedule均无标记，仅为快照，不保证未来无维护。

结果分区free=1138266955776 bytes（1060.094GiB）；/tmp=775.284GiB；Triton/Torch缓存同结果分区；/dev/shm=125.759GiB。`/scratch`和`/data/zypan/scratch`不存在，不把它们当已预留scratch。原2GiB结果空间门槛现在满足，未来峰值temp/cache/storage/连续性仍需测量和预检；没有清理缓存。

10条历史device job（A四条、B smoke+calibration四条、persistent qualification两条）由各receipt到canonical完成或STOP时间独立重算，总device-time=2.089558591710197GPUh，process-release保守界限=2.104673161506653GPUh。剩余分别13.910441408289802/13.895326838493347GPUh；不是预算耗尽。初始0账本仅保留，不能替代总A+B+persistent账本；等待、其他实验、CPU不计本轮GPU开销。

两qualification各900s cap总0.5GPUh只是限额，不是保证成功或未来成本。真实8448micro=0，因此完整四student32K+两必要同GPU FP保守cost/VRAM还不能验证预估。原冻结规则需两卡micro实际最慢post2048block速率×32768/8448增长包络、1.5margin+30s、加载/FP/两student/所有历史资格计入；native-cache extrapolated peak<=85%显存，完整总<=16。不可用其他runner历史速度或partial sample替代。详见`RESOURCE_AND_BUDGET_AUDIT.json`。

## 5. 可修复性与后续资格

结论`ELIGIBLE_FOR_ISOLATED_REPAIR`仅表示具备明确原始首因、选择有效且身份未变的隔离工程候选。显式enable可能切换algorithm/触发unsupported-op RuntimeError并改变性能；不能宣称修复一定成功，也不能把本CPU审计冒充GPU PASS。修复预计只涉及runtime setup/observability与新code/freeze，dataset/roster/pairs/winners/P1/L7/QDQ/FP同GPU配对原则/metric/endpoint均不允许变。

下一步条件（本轮全部未执行）：

1. 另行获得工程修复与资格执行授权；新隔离目录/新提交/新 freeze，不删除或清除旧 STOP/latch，不复用旧启动标签。
2. 只改 runtime 初始化与可观察性：显式设置并记录确定性配置，保留原 assert，不使用 warn_only 绕过，不更换 Torch/CUDA/FLA 或固定 kernel selector。
3. 验证 frozen document/token/rotation/model/selection SHA、全部配置及同一固定 GPU UUID；空闲与独占要在新启动前再次检查。
4. 两卡分别完整重做新的512-token×2 GPU qualification：同 GPU 新 FP reference/replay与 full-logit parity、真实INT8 codes/inverse/native-coordinate writeback、next-kernel consumption、storage separation、全18KDA/6MLA/conv缓存、finite/repeat。CPU PASS 不替代这些门禁。
5. 两篇都通过后才允许同引擎8448 micro；必须跨2048/8192边界继续每token QDQ，量测loading/JIT/FP+两student总耗时、峰值VRAM、native cache growth、storage与连续性。
6. 累计全部历史+新资格+micro开销，用原1.5安全系数与增长包络计算完整四student32K+两同GPU FP成本；总保守值<=16、VRAM<=85%及存储/连续性都PASS才可申请/执行正式长程。不得用旧速度或部分roster替代。
7. 继续使用已冻结RouteA/pairs/winners，primary KL(FP||student)[8192,32768)，secondary[2048,32768)同轨迹；anchor后持续QDQ，不要求原协议没有的跨GPU/driver FP bitwise identity。

## 6. Provenance及字节保护

真实主worktree：`/data/zypan/experiment_runs/LING_KDA_L7_OBJECTIVE_REDUNDANCY_AND_PERSISTENT_VALIDATION_V1`。当前HEAD=`a7921dd07ece62f28bfd4cfcef0eef5c928955a2`，不是测量commit；历史PhaseA/CPUanalyze/calibration/pairaudit/persistent五提交均实际存在。persistent14测量文件逐个git-show内容SHA与runfile相等，persistent commit=`f1fa49883d5f1f65ed47d8ddc0b37a62cbee49d1`。当前tracked staged/unstaged diff为空，但有132条untracked evidence，所以不声称整个worktree无文件变化。两上游source仓库HEAD均与SOURCE_IDENTITIES相同，bridge仓库另有既有dirty/untracked项目，完整status见provenance；绑定科学源码SHA全部匹配。

服务器431项manifest/sidecar/source/asset检查全部MATCH；32权重分片完整SHA重算且metadata与fresh audit相符；旧80canonical+80sidecar及两永久Attempt STOP重哈希MATCH。没有独立128panel覆盖证据，不虚报128项PASS。原实验335个非.git/非__pycache__文件审计前后及补充核验字节相同；本轮仅新审计目录产生文件，不修改模型/rotations/旧reports/hashmanifest。

| 原始相对路径（STOP/log为绝对路径） | 实际SHA256 | 审计 |
| --- | --- | --- |
| `PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json` | `cd7314e812bf206da8c8a08d7df934b738f52cc3be25b8c4b7ddc45f603492b7` | MATCH |
| `PROSPECTIVE_DOCUMENT_SELECTION.json` | `aeb19c2602f58f2e1f35b5142e20ac56c4c1e2f247404d5c0be96adf7c6069a0` | MATCH |
| `PHASE_B_PAIR_METHODS_FREEZE.json` | `a387df6426ba260a42f84012e318bc61a8a10e662711edeb9359a39d471a021c` | MATCH |
| `PHASE_B_CALIBRATION_MATCHED_TABLES.json` | `7c454614eea1a3afae24c2cb39a2759ed0b2b39d36b7554e1e21c004eb6d3555` | MATCH |
| `SOURCE_IDENTITIES.json` | `c41badce52059908fa5e9507f57102e172323531f35de66146163bfdcb0284d4` | MATCH |
| `PHASE_B_CALIBRATION_CODE_FREEZE.json` | `4841b38e4a151cd0598d557d42d37daa88a38e9c76c51e531206db35fba247dd` | MATCH |
| `PHASE_B_CALIBRATION_CONFIG.json` | `ce3268d5bcb993f3a1ec3eb1f2a4b1fd542aad21286964612bc95bba07f8d86c` | MATCH |
| `PERSISTENT_CODE_AND_ASSET_FREEZE_V1.json` | `5cb4cac2995b55ff15ab98c77ac633f275242fc23459c6075fab4f02f9516edb` | MATCH |
| `run_persistent_stage_v1.py` | `b019284b22c4a1431bca3add75783345a42641447b2cfb5a7b12a6c70e2f192d` | MATCH |
| `persistent_engine_v1.py` | `d448121735fd5ed7eff3a62689d359f17eca4b5467972ff69d299073a2b865c8` | MATCH |
| `persistent_stage_common_v1.py` | `6ddccc5690226e6bb1106fffc115efe3313709a4de046f2b831cc4cd982af3f9` | MATCH |
| `final_stop_20261011/DELIVERABLE_MANIFEST.json` | `6e294b72faa93c0ae1ae554850778cbb9022c0f1540905b70e81186ed6610a21` | MATCH |
| `final_stop_20261011/RESULT_AND_DELIVERY_HASHES.json` | `bc80ed6b9699e2d84213fc422820887442494f0408282da1c68aedd658d3f5a5` | MATCH |
| `final_stop_20261011/SHA256SUMS.txt` | `36f51b6d4c22e5c0ea07229a4bed4f46f3f8ef3baa98691dbc70f6c2d03ce19d` | MATCH |
| `final_stop_20261011/GPU_COST_LEDGER.json` | `734dfd3d2911e696b47ed29cdd59b996aa52df927132a7a71bf2fdfff2ded7c3` | MATCH |
| `GPU_COST_LEDGER.json` | `1ec4b572331d4989770eabc7b7a0a003068619b2ebcad45fce75a436c8c8ed68` | MATCH |
| `/data/zypan/experiment_runs/LING_KDA_L7_OBJECTIVE_REDUNDANCY_AND_PERSISTENT_VALIDATION_V1/runs_b_persistent/qualification_pg19-test-860_gpu6.STOP.json` | `b9f31ce9ac844fc069f1222106cdb03d335effac338b054d0c7c0cc948585809` | VERIFIED |
| `/data/zypan/experiment_runs/LING_KDA_L7_OBJECTIVE_REDUNDANCY_AND_PERSISTENT_VALIDATION_V1/runs_b_persistent/qualification_pg19-test-860_gpu6.log` | `419273f0a765eb91f280a164ed915a41bf386d887bc8eb5504c277e64f27edaa` | VERIFIED |
| `/data/zypan/experiment_runs/LING_KDA_L7_OBJECTIVE_REDUNDANCY_AND_PERSISTENT_VALIDATION_V1/runs_b_persistent/qualification_pg19-test-5956_gpu3.STOP.json` | `1b5aa8d31d9de8faacb159a3ef7ea77fee21cfa64a89c9baeedf8872d35ac020` | VERIFIED |
| `/data/zypan/experiment_runs/LING_KDA_L7_OBJECTIVE_REDUNDANCY_AND_PERSISTENT_VALIDATION_V1/runs_b_persistent/qualification_pg19-test-5956_gpu3.log` | `f4142ab62a4842ee615a9550685a4c3b0b615168740fe7cda5f7d9612c438f68` | VERIFIED |

完整原证据/command stdout/原git status/所有file identity/hash-check结果在`READONLY_SERVER_EVIDENCE.json`，SHA=`24e5af27dd1ec2b482779e53bcb0f91862b132d3bd293f71844ecff88f1105d2`；补充静态依赖/原receipts/进程观察在`READONLY_SERVER_SUPPLEMENT.json`，SHA=`ac28344edc9ff49f905a7e85488c648eb9a83ec202e395fe1e42fd69d60297c2`。没有使用用户交接文本模拟远端资产。三个JSON各附完整provenance；新deliverable自身hash见`SHA256SUMS.txt`（manifest正常不hash自身）。

## 7. 精确交付路径和本轮停止点

本地：`C:\Users\admin\Desktop\操作系统实验\lab2\ling_l7_phaseb_stop_forensic_audit_v1`

远端：`/data/zypan/experiment_runs/LING_KDA_P1_L7_PHASEB_STOP_FORENSIC_AUDIT_V1`

六项指定输出为PHASEB_FROZEN_SELECTION_AUDIT.json、FIRST_FAILURE_GATE_REPORT.json、RESOURCE_AND_BUDGET_AUDIT.json、REPAIR_FEASIBILITY_ASSESSMENT.md、FINAL_READONLY_AUDIT_REPORT.md、SHA256SUMS.txt；另保留stdlib-only collector/builder和真实只读证据用于复核。不对旧实验追加PASS/新STOP或更改monitor。本轮结果交付即停止，不开展提案中的修复或后续资格。

NO_NEW_GPU_SCIENTIFIC_RUN = TRUE
NO_NEW_32K_LABELS = TRUE
NO_TRAINING = TRUE
