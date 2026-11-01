# Qwen Group32实际GPU资格报告

整体：**INCOMPLETE_WITHIN_AUTHORIZED_GPU_BUDGET**。控制器终态：`BUDGET_INSUFFICIENT_FOR_NEXT_LEGAL_UNIT`。预算0.5 GPUh已固定，未更改科学程序、未重启单位、未补跑FP，也没有启动32K/AIME/训练。

| 条件 | GPU Codec | 最大完整fresh长度 | Persistent资格 | INT8 TPS | 32K准入 |
|---|---|---:|---|---|---|
| B0 | GPU_CODEC_QUALIFIED | 32 | PASS_32_TOKEN_PREFIX_ONLY | NOT_RUN | 尚未具备完整前置资格 |
| B1 | GPU_CODEC_QUALIFIED | 32 | PASS_32_TOKEN_PREFIX_ONLY | NOT_RUN | 尚未具备完整前置资格 |
| B2 | GPU_CODEC_QUALIFIED | 32 | PASS_32_TOKEN_PREFIX_ONLY | NOT_RUN | 尚未具备完整前置资格 |

1. **B0复现canonical？** CPU在24层×32heads、12,582,912元素上值/字节均完全一致。GPU微型B0实际37fixture×2次通过codes/scales/负零/重建严格参考；保留CUDA scalar reciprocal语义，不将朴素CPU除法当作CUDA bit参考。canonical7项AST及FP数学不变。协议是本次token0 QDQ，不能冒充历史OLD首token漏QDQ协议。
2. **B1 Group32 Sym GPU资格？** GPU Codec为GPU_CODEC_QUALIFIED；真实9B完成与fresh长度见表。未完成的128/512或测速不自动升级PASS。
3. **B2 Asym资格？** GPU Codec为GPU_CODEC_QUALIFIED。37组冻结边界/真实state fixture逐字段核验FP32 scale、UINT8 zero-point/codes、int16相减及decode；同进程同卡重复2次。真实模型完成范围见表，没有扩大scale容差或修改舍入规则。
4. **统一Runtime和时序？** 全部实际canary使用相同HISTORICAL_PORT、H128、输入、BF16模型、FP32 carrier、每token逐层QDQ；只变codec。没有Dual-M/permutation/rotation/新FP优化。所有实际数值单位同一GPU UUID，不跨卡引用fresh byte参考。
5. **首token persistent？** 所有PASS canary均从token0写回，完整运行共4608个层QDQ更新。stepwise query长度必须为1，未用chunk末尾单次QDQ代替逐token。未完成单位不声称完整资格。
6. **真实next-consumption？** 已完成单位共4464次实际native helper initial_state检查：实际值和完整字节对照上一token整数编码的临时CPU解码。不是指针/hash代替。未量化state不参与下一次递归；native FP32 QDQ carrier仍48MiB，不是物理packed INT8。
7. **fresh-process？** 表中最大完整配对长度以内，两个独立模型进程在同UUID、相同种子/输入下实际完整raw逐字节比较通过。CPU两fresh证据保留。短32/128不能称512、32K或81K稳定性；未完成R1不形成配对资格。
8. **真实TPS？** 仅实际完成128warmup+128measurement且同步、无Profiler/逐token观察的测量才进入CSV。目前测量条数0；其余为NOT_RUN。canary包含CPU完整字节观察、哈希与归档，其wall不能当吞吐，也不以历史9.064610或FP13.774973填当前值。
9. **风险与显存？** 完成范围内执行state/scale/level/decode/logits finite检查。实际峰值逐单位记录于RESULT；不把编码理论字节当实测显存节省。B0 scales384KiB，B1 scales1.5MiB，B2另加zp384KiB；codes各12MiB，Sym负零bool flags临时12MiB，native持久carrier仍48MiB。完整canary的round前clipping计数未单独留存，不能把边界codes计数冒称overflow。37fixture严格通过不构成长程误差上界。H GPU正交阈值检查通过，但没有另称CPU/GPU H变换输出bit等价证明。
10. **32K Future-KL候选？** 仅表中具备完整短程数值与fresh前置资格者可作为下一任务候选，仍必须新预算/文档/用户授权。缺少计划的512验证不能直接进入；测速NOT_RUN表示缺少性能证据，不解释为数值否决。后续匹配FP_NATIVE/H+C128/H+C32 Sym/H+C32 Asym、32768输入、384位置、全248320词表FP32 KL与文档聚合方案保留；没有用历史NEW_STRICT KL替代。

资源：进程预留GPUh观察区间[0.38975954150574077, 0.39091705589244763]；**EXACT_GPUh=UNRESOLVED**，不能把轮询时间当精确物理计算时间。append-only事件保留dispatch、首次exit、首次release及上下界。独立进程/NVML检查确认原worker全部释放；预算未被重置。CPU准入/依赖hash与GPU预留时间分开。

冻结控制器的cleanup_usage_incomplete=True字段有记录逻辑问题：退出循环时name已指向未派发的B0_128_R0，而child仍是最后完成的B2_32_R1，造成该布尔值误报。独立复核7个真实dispatch均有且仅有一次exit/release，全部7段也都在usage中，资源区间没有缺项。原字段/源码保持不变，在RESOURCE_LEDGER中单独记录这一解释，不将其当作0GPUh或精确物理时间。

归档：96个当前大raw对象、16734191616字节，.29永久存储已fresh流式SHA/bytes核验；临时孤儿数0。大tensor未下载Mac，历史raw未删除。本次科学源与依赖在结束后复核。失败回执数0；所有原始回执/日志保留，预算NOT_RUN不解释为科学负结果。

前序CPU seal/542文件和旧FP最终seal保持不变。前序finisher当时未seal与随后实际最终seal是两个时间点，完整provenance均保留。
