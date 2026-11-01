# 性能差异证据状态（执行后）

冻结前假设逐字节保留于pre_gpu_reports/PERFORMANCE_CAUSE_HYPOTHESES.md。本文件是独立CPU解释，GPU执行源/数学没有更改。

| 假设/事实 | 状态 | 证据和限制 |
|---|---|---|
| 历史13.38/9.12有真实原记录支撑 | OBSERVED | 8原始SHA匹配分片，各60条；ΣN/ΣT=13.384239/9.115225。 |
| 40.7%表面下降仅由TPS聚合造成 | RULED_OUT | 历史均值/中位仍接近13.4；同条件旧13.742、新7.777。 |
| 当前速度测于32K cache位置 | RULED_OUT | 原测量1tokenprefill/128warmup/128measure，当前逐tokenRoPE和8KV实测到256。 |
| 新旧runtime在同条件下有稳定差距 | OBSERVED | 四组各3次普通测量；FP范围不重叠，同GPU/input/cache/timing。 |
| 新FP主要慢在可观察的GPU矩阵计算本身 | UNVERIFIED | Profiler aten::mm关联GPU时间20.928/20.939ms/token近似，hostself7.076/55.828显著不同；完整普通critical-path未独占拆分。 |
| aten::mm主机侧调用为主要应定位范围 | INFERRED | 两个Profiler最大CPU self增量48.751ms/token在mm，次数249/token相同；与普通wall差55.820方向/量级相符，但profiler扰动明显，hostself可含库内部工作/等待，不是已证明独占CPU算力瓶颈。 |
| 额外fill与有限性检查确实执行 | OBSERVED | 真实CUDAKernel2710→3557/token，差847；isfinite1→25、all1→26；fill_24→823。不能把次数直接换算正常耗时份额。 |
| 新路径额外24次scalar全局同步导致主要降速 | RULED_OUT | 实际item与cudaStreamSynchronize均1/token，没有新24次；不排除其它未展开等待。 |
| 单一deterministic开关解释主要差距 | RULED_OUT | 保留其它新设置，仅关开关3次中位7.996658，未恢复13.742；不等于所有确定性相关后端配置成本为零。 |
| det开关有小的性能影响 | INFERRED | 本轮观测+2.828%，短窗N3有次序/运行变动；非数值等价或独占贡献率。 |
| intraop1线程单独造成主要差距 | RULED_OUT | 仅改24线程3次中位7.655933，interop仍1、OMP/MKL仍1；没有声称复现全部历史线程状态。 |
| 每token反复compensated chunk造成稳定Decode降速 | RULED_OUT | 每trial chunk24次均在计时前；计时recurrent3072次、源码无该cumsum。初始化/状态的间接影响未独立拆分。 |
| CUBLAS_WORKSPACE_CONFIG或其它后端设置为主因 | UNVERIFIED | 确认设置不同，但本轮没有独立CUBLAS/interop/cudnn交叉消融，不凭环境变量断言因果。 |
| 完全不同GDN数学或替代fusedcore造成差距 | UNVERIFIED | Pythonfallback90d一致，本轮旧57Kernel名称全在新64中，recurrent计数相同；名称不能证明参数/数值相同，历史optional依赖不全。 |
| ColumnLS/Dual-M导致FP或H+C128差距 | RULED_OUT | 本轮四主条件未调用Dual-M/ColumnLS。 |
| 今天组间换卡/不同SMclock解释差距 | RULED_OUT | 固定UUID；普通18窗SM中位均1695MHz，独立cache/输入绑定；遥测2秒采样不是连续时钟证明。 |
| 原历史GPUclock/权重/optional包与今天完全相同 | UNVERIFIED | 历史UUID/clock/全权重和tokenizer逐文件SHA缺失；今天sharedcheckpoint4分片CPUrawSHA只证明本轮。 |
| CUDA Graph已经是被证明的解决方案 | UNVERIFIED | 未开发/未测；不自动开展优化，未证明其能保留科学数值门禁。 |

终态RUNTIME_DIFFERENCE_CONFIRMED_AND_STOP；底层主因与独占成本拆分仍UNRESOLVED，不用“OBSERVED热点”冒充已完成因果消融。
