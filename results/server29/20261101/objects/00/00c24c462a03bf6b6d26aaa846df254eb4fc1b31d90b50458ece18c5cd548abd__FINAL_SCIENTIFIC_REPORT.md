本轮状态：**HISTORICAL_PORT_FP_QUALIFIED_SPEED_RECOVERED**。恢复协议为 HISTORICAL_PORT，历史二进制 EXACT 身份仍 UNKNOWN。

十个问题逐项回答：

1. **是否使用最新有效源码？** 本轮源清单407项逐SHA/bytes核验，其中403项继承、控制器/观察worker/计划/新请求4项明示修订；父有效清单SHA为 64ca5853e4ed65ab3e2c59d9deb85b4aa026f0f712abd5496495b7b912fcb457。
2. **真实Driver/NVML身份通过了吗？** 每个实际派发单位的Driver与PRE/POST own-PID/库身份逐项保留；未完成或失败单位保持未通过。全部自有进程退出与所有UUID释放由外部root证明准入，本工具不查询GPU。
3. **是否执行真实9B FP？** 至少有实际完成且身份完整的FP单位。
4. **两次fresh是否逐token逐Tensor一致？** PASS_FULL_32TOKEN_RETAINED_CPU_PAIR_COMPARISON；torch.equal、byte、maxabs/maxrel、首token/layer/head来自CPU对实际保留raw的比较，signed-zero字节差与数值相等分开。
5. **真实激活FP64递推是否合格？** 完整32-token两fresh与保留raw复核通过。每unit需32×24的9类真实中间态与独立FP64从token0 carry重放；不是全模型FP64或32K/81K误差证明。
6. **当前128+128实际TPS？** 三次合法OLD中位 13.774973185423399 TPS，逐次数据见下表。
7. **达到约13TPS了吗？** 达到预设工程目标。
8. **若未达到，原因是什么？** 首次失败、未完整单位及reducer错误见结构化回执；身份/数值/资源/观测不足分开，不能将未测或失败解释为速度确实慢。首失败：没有FIRST_FAILURE文件；不虚构失败原因。
9. **能否用于后续科研？** 仅本次已测FP短前缀与短缓存窗口具有对应证据，仍需后续独立授权和科学合同。
10. **Group32还缺什么？** H+C128/C32 Sym/Asym分组、scale/zero-point、prefill/QDQ/writeback/next-consumption及persistent shadow须独立冻结、同恢复协议FP基线匹配并实际GPU资格；本次INT8/AIME/长Future-KL/训练均未运行。

实际普通短窗观察：

| 单位 | 合法身份完整 | TPS | ms/token | 128-token wall秒 |
|---|---|---:|---:|---:|
| OLD_FP_R0 | True | 13.382607053095176 | 74.72385582514107 | 9.564653545618057 |
| OLD_FP_R1 | True | 13.951699074157712 | 71.67585787829012 | 9.174509808421135 |
| OLD_FP_R2 | True | 13.774973185423399 | 72.5954226218164 | 9.292214095592499 |
| NEW_FP_R0 | True | 7.741047785182266 | 129.18147875461727 | 16.53522928059101 |

资源采用首次dispatch、首次退出、首次所有UUID释放事件；进程预留GPUh区间为 0.4047023999608225 至 0.40566131705211267。精确物理GPUh为 **UNRESOLVED**；NVML owned-PID峰仅为采样下界，不能称连续精确峰。当前无GPU时间硬上限，仍执行1800秒真实无进展watchdog及8GiB raw/10GiB磁盘保留。

MODEL_LOAD_TIMING记录原loader/reference调用wall，初始化/reset及同步边界分别列出；WORKER_CUDA_ALLOCATOR_OBSERVATION汇总显式reset前epochs及结束计数，原RESULT的warmup/decode peak保持原值。它覆盖进程内观察阶段，不冒充进程退出之后的遥测。

成功中间态包括 query/key/value/gate/beta/core_actual/state_actual/native_norm_query/native_norm_key，原cache/logits raw另保留；捕获完成不是数学PASS，必须结合独立重放。

历史13.742036038730332和7.77671073754353仅是历史同窗背景，不是本轮数据，也不证明INT8速度。原CPU530包与控制修订51包保持不变。本报告不封存、不删raw、不重试GPU或CPU reducer，不创建空科学CSV。
