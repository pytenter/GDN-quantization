# QWEN_GDN_HADAMARD_DUAL_AXIS_INT8_FEASIBILITY_V1

PHASE=CPU_ONLY。最终分类 CPU_CODEC_QUALIFIED_ONLY；完整封存状态见 GATE_STATUS_FINAL.json 和 hashes/DELIVERY_VERIFICATION.json。只证明 CPU 实现与工程约束，不构成 full-model 结果。GPU_HOURS_USED=0；GPU pilot、C8 均未授权。

## 实测结果

A=C128_CANONICAL，B=C128_FP16_SCALE，C=C128_LS_FP16，D=DUAL_M_INT8。固定 seed20261015，9个案例，B=1/H=32/K=V=128，batch_two为B=2。生成参数和逐输入 SHA 在每个 attempt 的 preregistration.json，先保存，再执行任何 numerical gate；无样本筛选。输入都视作 Hadamard Key 坐标；只有 hadamard_synthetic 从 native 合成数据施加一次 H.T。

| 固定合成案例 | A Canonical | B FP16 scale | C 单轴 LS | D Dual-M |
|---|---:|---:|---:|---:|
| normal | 4.176641737e-05 | 4.176753349e-05 | 4.126982938e-05 | 4.082832921e-05 |
| hadamard_synthetic | 4.169223232e-05 | 4.169933328e-05 | 4.12009102e-05 | 4.087809459e-05 |
| row_imbalance | 0.0001121952485 | 0.0001122147079 | 0.0001108969038 | 2.219221379e-05 |
| column_imbalance | 4.094881995e-05 | 4.094722101e-05 | 4.048472409e-05 | 3.675513054e-05 |
| sparse | 4.62828821e-06 | 4.660824119e-06 | 4.033139983e-06 | 2.842065175e-06 |
| zeros | 0 | 0 | 0 | 0 |
| single_outlier | 1.360395751e-06 | 1.37592918e-06 | 1.373404847e-06 | 1.46389863e-06 |
| signed_ties | 3.111387679e-05 | 3.111387679e-05 | 2.707390586e-05 | 2.294823319e-05 |
| batch_two | 4.179393083e-05 | 4.179310648e-05 | 4.128951647e-05 | 4.088671998e-05 |


| codec | mean relative MSE | mean absolute MSE | mean max abs | mean p99 abs | mean encode / decode ms |
|---|---:|---:|---:|---:|---:|
| A | 3.505546774e-05 | 0.7052036849 | 3.8763825 | 1.5586197 | 9.3301 / 0.7700 |
| B | 3.50636147e-05 | 0.7052841251 | 3.8660159 | 1.5595263 | 7.6639 / 0.4949 |
| C | 3.41802594e-05 | 0.6946976534 | 3.8598543 | 1.5493809 | 10.9456 / 0.6384 |
| D | 2.319940946e-05 | 0.2778552461 | 3.5006153 | 1.1988358 | 13.3047 / 0.9012 |


上述“mean”是九种预设分布的等权描述性平均，max/p99 列也是各案例指标的平均，**不是 pooled max/p99**。没有统计总体含义、doc 独立性或显著性检验；绝对误差被不同幅度分布影响。完整36行指标、每行 quantiles 和全部7次 timing 样本见 `analysis/attempt_002/codec_comparison.json`，平铺数表见对应 CSV。CPU单线程，2次warmup，7次计时，中位数；包括当前验证代码开销，未做性能优化，不可推断 GPU 性能。

| codec | mean zero code | mean |code|=127 | mean |unrounded level|>127 |
|---|---:|---:|---:|
| A | 0.329779625 | 0.0261761347 | 0.000805113051 |
| B | 0.329779095 | 0.0261770884 | 0.00290075938 |
| C | 0.329778035 | 0.0261767705 | 0.00306267209 |
| D | 0.287537045 | 0.0189039442 | 0.0083878835 |


overflow/clipping proxy 明确定义为 |最终 scale 下未 round 的 level|>127；这不是硬件 overflow。实际 round 后越过范围而被 clamp 的比例另存 actual_rounded_clamp_fraction，scale FP16 分布逐案例保存；无 row scale 的条件为 N/A。饱和比例包含正常 absmax 映射到127的情况，不等价于信息丢失比例。

D 的等权 relative-MSE 平均比 A 低 33.8209%。普通高斯约降低2.25%，H合成约1.95%，row imbalance约80.22%；single_outlier 则**退化7.6083%**。所有四组比较（A→B、B→C、C→D、A→D）的逐案例 raw delta 与 relative gain 保存在 paired_comparisons.json；zero-reference 的 relative MSE 定义为0，relative gain N/A。局部退化保留，不要求 D 对所有输入占优，也不把单一有利分布当总体证据。

## 极端 scale 行为

这些是预先列入 G3 的 stress inputs，另行描述且不混入九案例均值。输入1e-30时四者均量化为0，relative MSE=1；这是 scale floor 下的精度丢失。输入±1e10时 A relative MSE=0，B/C为0.9983368904589415，D为3.04633348096e-9。B/C最大可重构幅度8319008，所以在超出FP16单轴scale范围时会出现严重 clipping；有限输出不等于数值质量合格。FP32_MAX时 A 按冻结正式函数 fail-closed，B/C/D relative MSE=1，D最大重构幅度544928301056。D 在有限 FP16 双轴范围之外同样会发生灾难性失真，不能据此声称任意输入稳定。

详见 analysis/scale_stress_description/results.json；D 在三个 stress case 中也与官方 fit_state codes/scales/reconstruction exact。一旦真实 state 落入明显 scale clipping 区域，未来 pilot 需按事前范围门禁停止；本轮不偷偷更改 scale 格式或算法解决它。

## 物理表示

以单层[1,32,128,128]为例，实测 tensor storage bytes 和实际序列化长度：

| codec | int8 codes | row scale | column scale | header | total bytes | bits/value |
|---|---:|---:|---:|---:|---:|---:|
| A | 524288 | 0 | 16384 | 32 | 540704 | 8.25048828125 |
| B | 524288 | 0 | 8192 | 32 | 532512 | 8.12548828125 |
| C | 524288 | 0 | 8192 | 32 | 532512 | 8.12548828125 |
| D | 524288 | 8192 | 8192 | 32 | 540704 | 8.25048828125 |


每个 encoded 对象仅拥有 int8 codes、允许的 scale、shape/codec metadata；无完整浮点矩阵、pivots、compensation。元数据用32字节版本化小端 wire header，payload为实际 tensor storage；Python对象/allocator开销不包含在 wire payload 内，已明确区分。D 与本轮物理 A 的 tensor bits/value 都为8.25；D的两份FP16 scale恰好等于A的一份FP32 scale。B/C为8.125，D较C多8192bytes。不能把此物理 A 表示与旧正式 runtime 混淆：旧正式 recurrent cache实际是FP32 QDQ数值，单层state本体2097152bytes，尚无新的 full-model resident INT8集成资格。

## 门禁、失败和重现

最终 attempt_002：G0–G5全部PASS；G6为独立封存和两端下载核验，见最终回执。G1覆盖9案例×4输入dtype（FP32/FP16/BF16/FP64）的共享参考 exact codes/scales/dequant，以及9案例的原正式CPU fake-cache24层执行。G2官方 all8/ones/pivots0 的9案例四项全部 torch.equal，无容差。G3共49项检查，另有2个 fresh-process code digest 完全一致；G5四codec连续128步，每步decode→FP32更新→encode→丢弃瞬时浮点state，在1/8/32/128记录，并与另一个独立进程逐hash一致。没有跨step未QDQ float carry。CPU recurrence是合成更新，不是Qwen delta rule/model forward。

**首个数值失败**：attempt_001 G3，FP32_MAX时新A adapter没有继承原正式writeback的finite guard，decode返回Inf。最小反例[1,1,1,1]已保存，并独立执行原正式函数确认其抛FloatingPointError。该失败后G4/G5均NOT_RUN。只对新adapter补同一fail-closed检查；测试要求此极端值明确拒绝，未放宽任何 parity、改变EPS/round/clamp/fitting/面板。旧 attempt原代码、traceback、反例和修复说明完整保留。源码采集前两次旧Git参数不兼容也保留为 preflight_001/002，未造成历史文件写入。当前 attempt 无 unresolved failure。

CPU环境Python3.10.18、Torch2.5.1+cu121（CUDA编译版本不代表本轮使用GPU），CUDA_VISIBLE_DEVICES空串，CPU线程1，未初始化CUDA、未使用MPS、未加载9B、未forward或训练。执行入口：

```sh
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ydai/miniconda3/envs/bitdecode/bin/python run_cpu.py --attempt NEW_UNIQUE_ATTEMPT
```

需在本实验根目录运行；使用已有CPU可用Torch，无模型下载；同名attempt存在时直接拒绝覆盖。新的attempt不会自动封存或触发GPU。analysis/read_only_scale_stress.py仅补充预先指定边界的描述性统计。

## 真实资产及科学范围

REAL_STATE_DATA_AVAILABLE=NO（本轮有资格且可核验的已有Hadamard FP快照）。已发现其他native FP资格快照，但其轨迹及坐标不符合本次面板纳入条件；未加载或强行冒充Hadamard轨迹。GDN历史G指标、标量MSE/KL不能还原完整state。没有扫描/复制84GB旧8-doc tensor；C6R state不使用。8192后连续q/k/g/beta、omega、w与paired errors=NOT_AVAILABLE。

用户冻结的AIME FP52/60、H+C12841/60、C6R40/60、C7 33/60、Native18/60仅为背景，本轮没有重评分、改判或新结果。旧简单scale调参负结果和functional增量未确认是独立历史事实，不可替代新双轴验证；NoInflation8-doc的oracle干预结果也不证明本codec。

## 下一阶段判断与仍待测量项

建议**有条件考虑**下一次独立授权的小GPU资格pilot：支持证据是 exact reference、物理表示和recurrent接口资格，以及多数预设合成分布有局部MSE下降；反对/限制证据是outlier退化、FP16范围灾难、额外fit开销及缺真实Hadamard FPstate质量证据。当前只起草 NEXT_GPU_PILOT_PREREG_DRAFT.md，未执行。

下一阶段必须实测模型加载/峰值VRAM、单token fit/decode+writeback开销、prefill与恢复数值/时间、吞吐、FP全词表logits与CPU审计成本、实际cache storage和瞬时FP缓冲、checkpoint/磁盘/归档带宽、8/32/128 canary及2K资源预算。CPU本轮timing不能转成可靠GPU小时预测。正式full-model QDQ timing、cache integration、FP equivalence、Future-KL、AIME均NOT_RUN。尚不能判断长程误差传播、功能收益、与rotation互补性，或部署可靠性。

Phase0封存后STOP，等待新授权；GPU_PILOT_AUTHORIZED=NO，C8_AUTHORIZED=NO，无自动下一阶段。
