Hadamard + Dual-M INT8
Qwen3.5-9B / AIME26

Primary:
Correct / 60 = 39/60 (65.0000%)

Seed 1:
Correct / 30 = 19/30

Seed 2:
Correct / 30 = 20/30

Total GPUh = 110.873351250
Total wall time = 27.5981小时（正式GPU阶段；含加载至释放）
Median tokens/s per GPU = {"0": 6.47542795393898, "1": 6.700943032897408, "2": 6.426027154817302, "3": 6.849544115791693}

终态：FORMAL_COMPLETE_60_60_AND_STOP。全部60个problem×seed唯一，原Canonical30与seeds1/2，原采样warper顺序及Top-K tie/EOS规则，原FrozenV4统一评分，没有中途按正确率改计划。EOS 55，length 5，runtime/nonfinite 0/0。中位输出30491.0 tokens，总输出2324521 tokens。四卡合计阶段吞吐23.396521 token/s，不能当单卡吞吐。详细峰值/分GPU成本见THROUGHPUT_AND_GPU_COST。

数值路径为原32K Dual-M Reference、H128合格Q/K位置、all8 signedINT8/ones/noFP16pivot/FP16row+column/一次columnLS/最终reencode，保留floor/ceiling/guard/舍入与signed-zero审计。GDN/H/QDQ deterministicTrue、原采样函数执行期间临时native许可并在finally恢复True、TF32False、原compensated chunk，BF16权重。只对完整prefill末尾做一次QDQ，再每次decode后QDQ；不是prefill内部每prompt token都QDQ，且不同于旧AIME首次prefill未QDQ。Attention KV未量化。FP32 QDQ carrier不是物理packedINT8常驻部署；瞬时编码tensor不等于常驻压缩。

每卡64非AIME固定输入（16prefill+48cachedforward）两freshprocess同GPU全trace精确；官方all8/ones仅作原codec算子参照，非另一个正式实验臂。未声称跨GPUbitwise或81920全程数值等价已获短canary证明。正式每token保留数值/scale风险、24层QDQ与next-consumption审计。已知official signed-zero byte差异在canary原始stat保留，未放宽门禁。

本轮只有一个正式臂。历史H+C128的41/60或C6R40/60只作背景，因prefillQDQ/数值runtime等合同不同，不能宣称严格同协议提升或下降。60次生成只有30个不同问题，两个seed并非60个独立问题；不将该样本外推为普适算法结论。

原始tokens/text保存在raw_results，评分后完整单条在results，完整60条FORMAL_RESULTS_60.jsonl。原始未完整attempt永久保留且不评分为数学错误。模型权重/大tensor不下载，本轮未训练、未跑FP/H_C128/STEPQuant/C6R或新32K，未优化或gitpush。全部own进程与NVML已释放。运行费用含资格/加载/正式至真实释放，不套用性能诊断预算。
