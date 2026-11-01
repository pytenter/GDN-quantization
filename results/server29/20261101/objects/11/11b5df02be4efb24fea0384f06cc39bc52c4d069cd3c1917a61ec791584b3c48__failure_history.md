全部历史保留：

1. 原 CPU V1 stress：±1e10 的 B/C 严重失真；FP32_MAX 的 B/C/D 极端风险；旧 native CUDA cumsum bitwise FAIL 及 fixed-route 历史失败不追溯改写。证据仍在父目录 inherited_cpu_v1 和 runtime_sources 的封存输入。
2. 原 CPU-only task 因 GPU_NO 形成 PARTIAL_STOP，G0/G1/G2 PASS。未覆盖。
3. execution_01：漏注入 tsha，全局缺失导致 CPU NameError；启动拦截不足，误 dispatch 后人工核对 cmdline 并终止 worker。耗时35.3840524852秒。无合格科学回执；部分 forward 是否开始未知。其全部24文件已独立 seal。
4. execution_02 pre-GPU CPU setup：缺 analysis 目录，补齐后重新完成必要 codec CPU 检查；CPU_SETUP_FAILURE_001.json 留存。
5. execution_02 pre-GPU exact CPU：codes/scales 一致，dequant hash 因负零不同；修复 canonical 写回 I/O 符号位后，24层原函数逐字节 PASS。旧预修复文件与 CPU_ENGINEERING_FAILURE_002.json 留存。
6. execution_02 GPU A0：Doc1 token1，BF16 full-vocab logits relative_L2=0.006718461729707917 > 0.002。FIRST_FAILURE.json、保存的两份 logits 与 CPU 独立重算完全一致。后续 A0 fresh-process、A1/A2/B0/B1/B2 均 NOT_RUN。

本 attempt 结束后没有热修、没有 GPU 重跑、没有改容差或另选文档/设备。当前内部最早差异位置未定位，不能声称数值问题已修复。
