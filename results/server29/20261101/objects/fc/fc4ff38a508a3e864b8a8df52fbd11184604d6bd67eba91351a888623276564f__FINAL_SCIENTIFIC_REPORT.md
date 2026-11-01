本轮结论是 **FAIL_CLOSED_STOP / INCOMPLETE_OR_ENGINEERING_STOP**。四卡GPU已真实使用，停止原因是执行期加载的libtorchaudio.so未纳入冻结库身份清单；没有完成B0/B1资格或性能比较，因此没有合格Fast Runtime，也没有提速或负性能结论。

| 设备 | 已提交B0参考token | 单位状态 | B1/测速 |
|---|---:|---|---|
| GPU0 | 252/256 | 共享停止；缺失RESULT保留 | 未执行 |
| GPU1 | 253/256 | 共享停止；缺失RESULT保留 | 未执行 |
| GPU2 | 253/256 | 共享停止；缺失RESULT保留 | 未执行 |
| GPU3 | 256/256 | 完整参考INDEX；worker身份FAIL | 未执行 |

合计1014条参考快照、160,036,398,000字节（约149.05GiB）。它们是首个B0自参考，不是四卡之间或fresh-process间的数值等价证据。GPU3参考INDEX虽写PASS，最终worker RESULT仍FAIL；其他三卡缺失的RESULT未补写。

用户六个主要问题的回答：

1. **新版约7.78 TPS为什么慢？** 历史Profiler支持249次mm普遍存在主机首次launch前开销增大，GPU关联GEMM时间接近。唯一内部根因仍未确定；本轮未增加可合法比较的新吞吐证据。
2. **小cuBLAS Workspace是否提速？** 未知。B1未运行，0条测速，不能计算提速率或认定候选无效。原三重复、5%和不重叠门槛没有放宽。
3. **deterministic fill能否安全优化？** 仍为FILL_OPT_SAFETY_UNPROVEN。CPU覆盖缺口原样保留；本轮fill=True，没有启动第二项GPU假设。
4. **能否证明数学、状态、logits、QDQ和确定性不变而提速？** 尚不能。冻结代码数学保持原样且部分参考已保留，但B0第二fresh、B1重复和跨条件byte比较均未完成；不能把参考自比较的byte_equal=True当成等价PASS。
5. **能否批准可重复的新数值高速协议？** 不能。没有候选重复性或性能证据，也没有已量化的B0/B1差异。未降级阈值将它转成类型B。
6. **是否提供Fast启动脚本？** 没有。`fast_validated_runtime.sh`未生成；Strict入口仅保留冻结CPU身份/环境输出，不能视为本轮GPU数值资格。

执行身份核验：最初小文件下载过滤遗漏的116个C/CUDA源码已从远端独立补齐、逐SHA核验，缺口历史仍保留。485个payload文件、34项共同控制绑定、每卡113个执行源、237个外部依赖（含模型分片）和28个已绑定库路径当前SHA/bytes均匹配。触发的新增libtorchaudio不在该闭包，当前磁盘SHA不能追溯证明退出前加载映射。该门禁在256输入完成后、发布RESULT PASS前执行，因而触发了全局停止。未发现最早数值分歧记录，不计算最大误差；这不等于数值误差为零。

1014条输入/next/cache-cursor、261516次Tensor dtype/shape描述、当前状态与后继初始状态的记录SHA谱系由独立CPU分析核验；这些仅描述已保存B0前缀。已提交原始文件共1014个；另有3个未提交文件360,577,048字节，单独保存且不计入科学结果。远端原始文件的最终当前SHA/bytes和未提交文件分类见 `audits/receipts/FINAL_ARCHIVE_STOP_AUDIT.json`，不以元数据替代实际原始文件核验。所有大型张量留在29永久归档，Mac仅小文件。

资源：仅四次B0首参考派发，四份首次exit/释放观测唯一且完整；独立重算与冻结账本一致。worker/设备预留观测范围 **1.922697234–1.923192717 GPUh**，含CUDA前CPU依赖/权重哈希；精确物理GPUh仍UNRESOLVED，NVML首次至末次可见跨度只作描述。四卡任务进程、NVML compute-apps及对应tmux已结束。停止不是GPU时间限额触发；未应用旧任务预算。不同主机wall clock有偏差，计量采用同一55主机monotonic事件，不混用绝对时间求差。

历史参考：OLD_FP/NEW_FP为13.742036/7.776711 TPS，OLD_H/NEW_H为9.064610/6.650598 TPS，另一个Dual-M任务约6.801314154 TPS。均未当作本轮同条件匹配结果；本轮全部当前性能为NOT_RUN，测量显存峰及当前mm/kernel Profile亦未产生。模型加载ENVIRONMENT和NVML观测不是完整测量窗口峰值。

原第十四节七种状态的适用性：

| 状态 | 本轮判定 |
|---|---|
| FAST_STRICT_EQUIVALENT_VALIDATED | 不满足，缺fresh/跨条件/性能资格 |
| FAST_NEW_PROTOCOL_REPRODUCIBLE | 不满足，无B1重复性/收益证据 |
| FAST_CANDIDATE_NO_BENEFIT | 未检验，不能宣称负性能 |
| FAST_CANDIDATE_NUMERICAL_FAIL | 未观察到数值比较失败；实际为身份STOP |
| FAST_CANDIDATE_SAFETY_UNPROVEN | fill独立假设仍安全性未证明；未执行 |
| CPU_ONLY_COMPLETE_GPU_NOT_AUTHORIZED | 仅原CPU封存的历史状态；当前已授权并用GPU |
| FAIL_CLOSED_STOP | **当前主终态**：未绑定实际动态库身份 |

没有修复已冻结清单、删除历史失败、重启已派发单位或增加新配置；H/Dual-M和条件D3因前置资格失败未执行。当前结果只能支持一次真实工程停止交付，不能支持Workspace、Codec科学方向、AIME准确率或长程Future-KL结论。

全部12项要求的交付均在本目录：执行计划、Strict合同、历史复用、cuBLAS资格、fill安全审计、数值JSON、历史性能CSV、Fast决策、差分、资源账本、本报告及SHA清单/封存回执。完整原CPU151文件封存另存parent_cpu，原目录和旧科学资产保持原位。逐件本地/55/29交付核验以 `LOCAL_DELIVERY_VERIFICATION.json` 为外部回执，封存索引排除自身和这些外部锚，避免循环SHA。
