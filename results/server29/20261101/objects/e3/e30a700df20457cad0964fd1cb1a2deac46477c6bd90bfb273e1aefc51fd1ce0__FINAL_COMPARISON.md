# dP32 吞吐专项：独立最终审核与完整比较

**最终状态：COMPLETE_NO_CLEAR_SPEEDUP。** 全部18组真实测量完成。仅非AIME固定输入；未运行dP32准确率评估。
本补充只读取已有工件；主SOURCE、主MANIFEST、数学和旧任务均未修改。单位TPS=实际decode forward/s，非生产serving吞吐。

## 同卡完整比较

短测每轮32 warmup+128计时forward；长测实际到16384，前32个作warmup，计时16352个。三轮顺序8→16→32、16→32→8、32→8→16。

|指标|dP8|dP16|dP32|
|---|---:|---:|---:|
|短测固定工作量TPS|4.645963|4.789177|5.097195|
|相对本轮dP8短测%|0.000000|3.082537|9.712340|
|相对本轮dP16短测%|-2.990359|0.000000|6.431547|
|长测32–16384 TPS|5.350327|5.527066|5.742168|
|短测平均TPOT中位ms|215.240619|208.804153|196.186336|
|长测平均TPOT中位ms|186.904463|180.927813|174.150241|
|边界前最后一个非边界读出fused GPU图计时µs|24.156160|30.289919|45.066240|
|单层边界整体同步host中位µs|14839.597046|25664.187968|36947.563291|
|短测128步全部24层写回次数|384|192|96|
|长测计时16352步全部24层写回次数|49056|24528|12264|
|全部24层buffer容量bytes|3397632|6795264|13590528|
|实测瞬时buffer payload峰值bytes|3397632|6795264|13590528|
|全部24层packed base bytes|8896512|8896512|8896512|
|末尾optional exponent bytes|0|0|0|
|短测CUDA allocated peak bytes|17950618112|17954518528|17962319360|
|长测CUDA allocated peak bytes|18501581824|18505219072|18512493568|
|长测CUDA reserved peak bytes|22332571648|22340960256|22347251712|

RUNTIME_VERIFIED：实际measurement和quantizer_audit；写回计时段由实测计数与已对齐的warmup边界相减。长测计数总值49152/24576/12288含warmup，计时段为49056/24528/12264。单层诊断不等于全模型排他耗时。
短测dP32相对16 +6.43%、相对8 +9.71%；长测相对16 +3.89%、相对8 +7.32%。长测16的5.47–6.16与32的5.62–5.83重叠；短测8的4.40–5.29与32的4.97–5.19也重叠。每配置只有3轮，不能称统计显著或普遍稳定加速。

## 全部原始测量及同上下文区间

|模式|轮次|配置|TPS|Decode wall s|平均TPOT ms|Host P50 ms|Host P95 ms|
|---|---:|---:|---:|---:|---:|---:|---:|
|long|0|8|5.353264841|3054.584535912|186.801892|170.02343386411667|390.19761979579926|
|long|0|16|5.527066205|2958.531596020|180.927813|165.55450856685638|398.245632648468|
|long|0|32|5.828077739|2805.727845788|171.583161|139.69390094280243|221.8622863292694|
|long|1|16|5.471762963|2988.433546945|182.756455|166.02518409490585|398.29032123088837|
|long|1|32|5.618083656|2910.601016477|177.996638|142.00760424137115|227.49584913253784|
|long|1|8|5.350327038|3056.261773393|186.904463|170.23667693138123|390.86833596229553|
|long|2|32|5.742168325|2847.704747602|174.150241|144.31428909301758|214.93607759475708|
|long|2|8|5.044629064|3241.467269644|198.230630|170.85473239421844|388.9215588569641|
|long|2|16|6.164391573|2652.654330462|162.222011|136.8596851825714|400.30956268310547|
|short|0|8|5.286620213|24.212066472|189.156769|UNMEASURED|UNMEASURED|
|short|0|16|4.816452473|26.575576261|207.621690|UNMEASURED|UNMEASURED|
|short|0|32|5.097194939|25.111851037|196.186336|UNMEASURED|UNMEASURED|
|short|1|16|4.712459619|27.162036464|212.203410|UNMEASURED|UNMEASURED|
|short|1|32|4.971885721|25.744759068|201.130930|UNMEASURED|UNMEASURED|
|short|1|8|4.404782354|29.059324548|227.025973|UNMEASURED|UNMEASURED|
|short|2|32|5.193499199|24.646196157|192.548407|UNMEASURED|UNMEASURED|
|short|2|8|4.645963215|27.550799280|215.240619|UNMEASURED|UNMEASURED|
|short|2|16|4.789176771|26.726931602|208.804153|UNMEASURED|UNMEASURED|

Host P50/P95为实际长测host迭代延迟，包含原sampler、Python、原检查以及原检查自身可能的同步；没有新增逐token CUDA同步，不是CUDA kernel P50/P95。短测原fixed_run没有逐token采样，保留UNMEASURED。

|区间|轮次|配置|相同token TPS|Wall s|
|---|---:|---:|---:|---:|
|2048–8192|0|8|5.334311819|1151.788685963|
|8192–16384|0|8|5.536367820|1479.670474648|
|2048–8192|0|16|5.604692929|1096.224195957|
|8192–16384|0|16|5.497425939|1490.151953056|
|2048–8192|0|32|5.806120817|1058.193619028|
|8192–16384|0|32|5.835363318|1403.854319498|
|2048–8192|1|16|5.643872830|1088.614181295|
|8192–16384|1|16|5.202893998|1574.508341566|
|2048–8192|1|32|5.563969486|1104.247608691|
|8192–16384|1|32|5.811649802|1409.582524687|
|2048–8192|1|8|5.276593386|1164.387617201|
|8192–16384|1|8|5.518679424|1484.413094372|
|2048–8192|2|32|5.819811582|1055.704280779|
|8192–16384|2|32|5.526829469|1482.224129796|
|2048–8192|2|8|5.045384017|1217.746752113|
|8192–16384|2|8|5.042359803|1624.636146620|
|2048–8192|2|16|6.096049924|1007.865761772|
|8192–16384|2|16|6.532069139|1254.120222256|

## 单层边界与fused诊断

|配置|pending|边界token?|两fused GPU图中位µs|普通host提交中位µs|
|---|---:|---|---:|---:|
|8|0|False|24.240320|194.542110|
|8|4|False|32.048640|348.933041|
|8|6|False|24.156160|195.346773|
|8|7|True|24.461759|201.515853|
|16|0|False|20.387840|225.245953|
|16|8|False|40.210239|315.040350|
|16|14|False|30.289919|240.273774|
|16|15|True|28.240960|224.806368|
|32|0|False|21.411840|226.438046|
|32|16|False|30.586879|240.668654|
|32|30|False|45.066240|223.539770|
|32|31|True|47.162561|410.765409|

每次Direct Read明确发射2个mixed-head fused kernels；完整K4 decode/norm和其他PyTorch操作的总launch数NOT_PROFILED。访问pending为0..P−1条，读出后增加本步u/log-decay；P−1 pending是边界token的读出阶段，P−2是其前一个非边界token，不能将边界前读出当完整提交。

|配置|边界整体host µs|重建event µs|encode含packing event µs|packing event µs|encode减packing µs|边界/P µs|
|---|---:|---:|---:|---:|---:|---:|
|8|14839.597046|7194.111824|3208.047986|455.632001|2758.543983|1854.949631|
|16|25664.187968|16330.160141|3898.096085|538.032010|3351.184085|1604.011748|
|32|36947.563291|27994.031906|3416.703939|472.576007|2946.719937|1154.611353|

event为同一stream的elapsed，可能包含host提交空隙；不是SM active时间。encode减packing是INFERRED residual，含scale/zp/exponent/RNE/layout/checks/dispatch，不是独立编码kernel测量。边界/P是单层诊断摊销估计，不是全模型实测节省；各列各自取中位，不能把中位数当严格可相加分解。
dP32写回次数对16减半、对8减75%，但单次重建随pending增多变慢；codec每次encode并没有按周期等比例减少。不能把次数下降说成总写回耗时已实测减半。dP32满窗口读出47.16µs高于16的28.24µs；长缓存确有访问/寄存器成本，收益与成本相抵。

## State、buffer与scratch内存

RUNTIME_VERIFIED：单层单token的packed K4 codes [1024] uint8=1024；FP16 scales [1,16,4]×2=128；packed zero [32] uint8=32；u [1,32,128] FP32=16384；log-decay [1,32] FP32=128，总计17696bytes。每层容量P×17696，24层容量24×P×17696。Python _slots里的None不分配tensor payload，已提交的记录清为None。
最大跨调用pending=P−1；本步append后提交前瞬时pending=P，独立boundary tensor inventory确实逐shape×element_size验证。全模型报告的peak是各层tensor payload峰值之和，不声称各层峰值在同一时刻同时驻留；容量不是始终live。末尾全部周期对齐pending=0，optional exponent末尾0；不能从末尾0推断整个长轨迹从未分配指数，轨迹指数峰值UNMEASURED。
packed base每层370688bytes，24层8896512bytes；codes-only为5bit/value，含scale/zp的base为5.65625bit/value（当前e0）。runtime head-map每层128bytes另记。完整State形状[1,32,128,128] FP32=2097152bytes，只在边界重建时临时出现，没有跨decode持久dense State。边界重建是原始u序列计算，不缓存完整高精度State。
storage_report的vector_scratch_peak_payload_bytes是每层向量scratch payload计数峰值再求和，dP8/16/32为6316032/12632064/25264128bytes；源码逐次累加(keys, updates, alphas, key, query)的numel×element_size，FP32 alphas、batched K、FP32 u stack等临时分配确实存在，但此计数不等于全部同时live的真实峰值，更不等于allocator reserved。重建FP32 State和量化临时张量另计，完整生命周期峰值未被逐分配跟踪。
dP32对16多6795264bytes buffer容量；本轮长测allocated峰值只多7274496bytes，整模型约18.5GB allocated、22.3GB reserved。没有OOM或nonfinite记录。不能由这些数据推出SM occupancy不受影响；occupancy/SM-active没有测量。

## 编译器实际元数据与测量限度

|可出现pending范围|观测specialization数|registers/thread范围|loader spills|
|---|---:|---|---|
|0..7|16|56..80|[0]|
|0..15|32|56..96|[0]|
|0..31|64|56..167|[0]|

实际编译API共64次，总wall 1.837254s，包含可能的磁盘cache加载，不能称纯编译。loader提供的spills全0，寄存器上限随pending到167；这是资源需求证据，不是实测occupancy。shared/warps原flags不变，详见COMPILER_TELEMETRY.json。加载/first-touch/诊断与主TPS分别计时。

## 身份、数值、失败及交付范围

CODE_VERIFIED/RUNTIME_VERIFIED：SOURCE_IDENTITY的41份原dP16/dP8科学文件逐SHA相同；LQUnrollBuffer仅允许周期8/16/32的构造器准入文本变更。正式快Triton kernel、原fixed_run、原codec/BF16norm批量K、quantizer/mask/sampler/输入保持原字节。无低效replay fallback、无额外数学/period专属优化。
两fresh真实CUDA资格各24cases均exit0/release；V2按30份相同科学文件和原真实STARTED/EXIT证据复用资格，没有重跑GPU门禁或复用任何benchmark状态/轨迹。原0.5%跨轨迹、0.1%同actual-state阈值不变；最坏0.3187185%、0.0474805%。31/32/33、63/64/65与preserve、prompt边界均直接检查计数；所有legacy UNEXPLAINED case都有same-base反事实，局部未解释0。并非bitwise等价证明。
59文件e604快照在GPU前因lineage审查扩展被替换，保留未启动身份；V1 SOURCE0237的120文件、0有效benchmark、真实benchmark/report/parent/Bash失败保留。漏原vendor资产使模型前校验失败；报告glob把.compile.json当数值结果导致CPU报告失败。224文件V1独立CPU recovery逐55/MacSHA复核。V2补原资产、CPU报告glob/回归，原真实CPU exit0/7/143和STOP/COMPLETE/OVERLAP分支不当GPU结果。
V2最终GPU测速及CPU报告/Bash都真实0；原终态helper的remote manifest换行SyntaxError发生于远程程序编译阶段，旧claim/FIRST_FAILURE保留；另发现文本allowlist漏.log。新的独立CPU recovery完成197主/恢复文本SHA，原helper和恢复claim均未重试。本额外审核再次核全部固定输入、主seal、模型17及实际regular mapped并另NOREPLACE sibling封存；不修改历史SOURCE、主报告或主seal。相关wrapper负记录随evidence保留。

## 为什么低于旧6.65/6.79，以及后续判断

确实使用原快速配置。旧快基准在GPU1，当前同型号GPU3；不是同一时刻的隔离因果对照，不能将绝对差异归因于换卡/并发。旧全模型短测TPOT约150.30/147.18ms，本轮约215.24/208.80ms，增约65/62ms；独立fused core只从19.10→24.46µs、27.95→28.24µs（旧kernel观测在GPU2），量级解释不了全模型差额。未改算法的prefill也由约2.97s升到约3–4.8s，因此INFERRED指向更广的运行/host/同步/环境开销，但根因仍UNRESOLVED。时钟1695MHz、无launch blocking、cgroup无throttle只是观察，不能排除未采到的因素。
dP32显示小幅潜在吞吐收益，同时buffer容量翻倍且pending读出/寄存器成本增加。仅从吞吐，没有充分证据优先投入一次完整准确率评估；若未来独立研究关注降低写回频率且接受额外缓存，可再单独授权评估。当前不自动运行dP32 AIME或新增任何实验。
未测项明确保留：实际occupancy、SM active、融合核三个内部排他时间、全模型排他writeback占比、全临时张量同时峰值/指数轨迹峰值。

源码证据（冻结副本相对路径）：scripts/throughput_dp32.py:20 micro和actual_precommit库存，:103 long_run和:115逐token循环、:126区间；src/lq_unroll_buffer.py:43批量原codec K、:66临时State重建、:118 K编码/:120 pending读出、:134 scratch计数、:149追加u记录；原始计时/容量/退出以相邻JSON和runtime收据为准。
