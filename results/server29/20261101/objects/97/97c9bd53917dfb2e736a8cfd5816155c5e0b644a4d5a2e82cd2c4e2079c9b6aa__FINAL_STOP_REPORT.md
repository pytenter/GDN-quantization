# Ling/KDA P1-vs-L7 V3 中文汇报

最终状态：{"ENGINEERING_STATUS": "STOP", "NUMERICAL_QUALIFICATION_STATUS": "NOT_RUN", "MICROBENCHMARK_STATUS": "NOT_RUN", "FORMAL_32K_STATUS": "NOT_RUN", "P1_VS_L7_SCIENTIFIC_VERDICT": "NOT_EVALUATED_NO_QUALIFIED_32K_RESULT", "completed_student_32k": 0, "completed_same_GPU_FP": 0, "error": "AssertionError('FRESH_PROCESS_EXACT_REPEAT_FAIL')", "code_commit": "31646483521ce6732be80b7cd64675b3a7b0663a", "v3_gpu_hours": 0.26360433081785833, "epoch": 1791774011.728206}

1. V2 guard具体问题：把目标compute与fuser所有holder合并，仅允许自己PID；失败瞬间hw未保存。历史首因未知，不把当前空闲当作历史原因证明。
2. V3逐次核验物理UUID、NVML compute PID、worker /proc starttime/boot_id/UID，失败前保存raw NVML/fuser和每PID分类。
3. 修复仅涉及guard、namespace、launcher和必要telemetry；V2 worker/runtime及原engine字节不变。CPU结果见CPU_REGRESSION_RESULTS.json；实际CUDA guard probe不是512数值资格。
4. 资格始终GPU6→GPU3串行；后续是否并行只由一次轻量dual guard probe决定，详见GPU_GUARD_SCHEDULING.json（若已执行）。
5. 两卡512×2：NOT_RUN。仅完整canonical、fresh-process exact fingerprint及全部机械门禁通过才标PASS。
6. 8448 micro：NOT_RUN。真实速度/allocated/reserved/cache/block记录在MICRO_8448_RESULTS.json；未运行则无数值。
7. 真实32K物理资源：NOT_RUN。无16GPUh/85%门槛，依据实际native/workspace growth和容量。
8. 正式student：0/4；没有拼接或更换样本。
9. same-GPU完整FP：0/2。原engine每文档一条新FP流在线配对两student，full-logit字节hash识别，未声称FP tensor文件已归档。
10. 文档860 P1/L7 KL：见下方实际文档表；没有完成则未测得。
11. 文档5956 P1/L7 KL：见下方实际文档表；没有完成则未测得。
12. 各文档支持谁：[]。
13. P1独立预测增量：NOT_EVALUATED_NO_QUALIFIED_32K_RESULT。仅两独立文档；不能宣称普遍排名优势、统计充分确认或梯度等价。
14. 本轮成本：0.263604330818 GPUh，含probe/loading/JIT/FP和student，按两设备求和。显存实际peak见各canonical/micro；CPU、等待、其他任务不计。
15. 是否做梯度资格：本轮未执行任何训练。只有方向支持才值得单独提出P1_GRADIENT_TRAINABILITY_QUALIFICATION；任何结果均不能直接证明STE/Cayley可优化或改善AIME。

经验数值uncertainty只基于资格期同logit输入的CPU64/GPU32 arithmetic及reducer验证，正式前冻结，非普适误差证明或统计显著性。微小Delta不强判胜负。V1/V2 STOP保留；本轮无自动patch/retry。

远端结果：/data/zypan/experiment_runs/LING_KDA_P1_L7_FAST_RECOVERY_AND_FORMAL_32K_V3

代码worktree：/data/zypan/worktrees/ling-kda-p1-l7-fast-recovery-and-formal-32k-v3，commit 31646483521ce6732be80b7cd64675b3a7b0663a；engine d448121735fd5ed7eff3a62689d359f17eca4b5467972ff69d299073a2b865c8。
