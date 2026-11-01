# 论文机制与本实现的边界

2026-10-27 已直接读取两篇指定 v1 HTML，包括相关推导段落；以下为简要机制归纳，不是官方源码核验。

[LeapQuant §3.1/3.2、Appendix A/B.2](https://arxiv.org/html/2609.38166v1)：窗口起点量化状态固定，缓存更新向量，读出展开为 base 的矩阵向量乘与缓存向量贡献；仅边界形成完整状态并重新量化。GDN 每 head 标量衰减；步长可以等价地放进写 key 或校正向量。论文另有 compensator 与 residual smoothing。本任务按授权关闭这两项；未访问或声称复现官方 CUDA kernel。

[PACE §2.1/2.3、5.1](https://arxiv.org/html/2610.29545v1)：周期控制状态重新舍入频率，buffer 计入驻留内存；mixed 配置使用 K4 与 BF16 values。论文的 kernel 与本项目冻结 Python replay 路径不可仅按名称等同。

CODE_VERIFIED（本目录）：`LQUnrollBuffer.step` 的 u 已包含 beta，实际缓存 K4/uFP32/loggFP32；`packed_readout` 用原 PACE G32 mixed4/8 base 与原 scale/exponent 规则，两个 head 分组 kernel，不形成逐 token dense State。`_delta_state` 仅 decode 边界使用实际 u 重建，调用原 encoder。数值路径为新实现；与论文共享展开机制，不共享未核验的官方实现身份或性能结论。

论文推导在固定层输入系数下分析递推；不能据此推断全模型在不同 BF16/量化轨迹下 logits 或生成 token 相同。本轮 full-model teacher forcing 已发现可测下游差异，须与同输入/同 base 算子误差分别报告。
