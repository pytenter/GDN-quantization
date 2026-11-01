# QWEN_GDN_FAST_RUNTIME_QUALIFICATION_V1 执行前计划

当前授权：ALLOW_GPU_DIAG=NO。本轮完成所有允许的CPU分析、可执行脚本及CPU资格，然后封存停止；不查询GPU/NVML、不导入torch运行时、不加载模型、不做forward、不操作GPU进程。历史源码、结果、seal、ledger不改。

## 优先级与工作包

1. 核验前序root-cause任务完整manifest及cuBLAS预登记的原始SHA/内容；用冻结执行源与ENV恢复NEW_STRICT合同，OLD仅历史参考；未知信息明确UNKNOWN。
2. 复用前序已对账847kernel与249mm时间线，不重复大trace解析；仅补齐fill安全所需ATen/模型/自定义buffer源证据。
3. 分配类别逐一审查完整写入先于读取、masked/分支/异步风险。未覆盖任何关键实现则FILL_OPT_SAFETY_UNPROVEN，不批准全局关闭fill。
4. 生成仅针对cuBLAS :4096:8/:16:8的独立fresh子进程资格/性能入口。env必须在torch/CUDA导入前生效；保留globaldet、fill、安全检查、数学、dtype、CPU线程及QDQ时机。
5. CPU测试覆盖控制门禁、源身份/AST最小差异、byte-diff定位、fresh进程调度与不可覆盖资源事件。CPU PASS不能标GPU数值/资源PASS。
6. 当前无GPU授权：所有FP/State/Codec数值与性能候选NOT_RUN；新GPUh=0；交付Strict封装和不可投入使用的候选配置，不生成fast_validated_runtime.sh。
7. 汇总来源明确的性能表、数值状态、Fast决策、diff及科学边界；全部新资产SHA/bytes核验与final seal。

## 前序合同与本请求的优先级

原cuBLAS预登记只读保留，绝不修改。当前请求要求NEW内部B0/B1且数值资格先行；前序原合同曾包含OLD/NEW先测速、仅两条byte资格。若准备后续执行，需在本root用明确addendum列出最新请求对顺序/重复数的覆盖；它不构成GPU授权。总单卡1800秒/0.5GPUh、1650保护+150清理不扩大，不自动执行第二个fill假设。

后续cuBLAS矩阵首先4条fresh数值（B0/B1各两次，全部256输入逐token），再至多3对裸性能；仅可信收益后按剩余预算安排H+C128适用性资格，Dual-M未测则拒绝Fast模式。任何身份/数值/资源失败立即fail-closed，不追加配置/seed/线程/窗口或重试。具体上限、预算与阶段门禁在资格文件冻结，CPU阶段不执行。

## 通过标准

Strict byte级等价、fresh确定性和内存/资源安全必须与性能改善分开判定。建议有效改善>=5%且超过重复波动，事前登记；原记录若已有不同准则则显式保留，不事后选择。FP通过不外推Codec/32K/81K。可重复但不等价的候选只能保留为待独立新协议研究，未有实际GPU证据不授予类型B。

最终本轮预期合法状态CPU_ONLY_COMPLETE_GPU_NOT_AUTHORIZED；若身份失败，则FAIL_CLOSED_STOP。禁止空壳数值、伪造CSV测量、未验证Fast入口与无限试错。
