# Dual-M backward policy

本政策用于资格准备。CPU 局部资格已通过；没有运行 GPU、A800、正式训练或 32K 评价。

## 固定数值与梯度合同

Arm A 直接调用权威 C6R `qwen_c128_ste`，保留原 `value + (exact_dequant - value).detach()` 前向表达式、codes、scale 和梯度。未为“统一实现”改变旧 C128。

Arm D 直接调用正式 32K 已使用的冻结 `codec.encode(value, 'DUAL_M_INT8')` 和原 `decode()`，保持 signed INT8、零 pivots、importance=1、FP16 row/column scales、原 column-LS、final re-encode、clamp、torch.round ties-to-even 与重构乘法次序。原 codec 的 source.detach 保留。

仅在 Arm D 的硬重构输出之外加入一个 Python `torch.autograd.Function`：forward 直接返回 detached hard reconstruction；backward 把上游 gradient 原样传给 FP32 state 输入，hard 参数无梯度。没有自定义 CUDA backward，没有新增 recurrence 或 Loss。

整个 Dual-M codec 使用 identity STE。row/column estimation、LS、FP16 conversion、round、clamp 和最终 codes/scales 全部 detach；没有未记录的 scale-gradient 路径。饱和区、ties 和 scale floor/ceiling 也使用同一 identity surrogate，不改成零梯度 mask。这是事前固定的 surrogate，不是 hard quantizer 的真导数，也不等于对 hard rounding 做有限差分。

两臂的整个 QDQ 对 state 的 backward 语义均为 identity。原 Cayley solve、H 后 Delta 两次原矩阵乘、原 recurrent、norm-before-gate、out_proj hook、State/Functional 定义及训练 detach/horizon 合同不因本政策改变。返回的 hard QDQ state 是下一 token 的真实数值输入；不存在跨 token 未量化 FP shadow state。

## CPU 实际证据与限制

冻结 CPU source manifest SHA：`fb4d2e6e7b436a21eab922127a285392965d29deae6c201cc6175278976e3c46`。一次运行通过 206 项检查，`CUDA_initialized=false`，没有 CPU 首失败。

- 正式冻结 Dual-M 与训练包装在真实 `[1,32,128,128]` state、zero、constant、tiny、random、non-contiguous、sparse outlier 上的 codes、row/column scales、reconstruction 原始字节相同。
- frozen levels 的正负 ties-to-even/clamp 与 frozen safe_half 的 FP16 floor/ceiling/midpoint 已单元验证；NaN/Inf 输入按冻结接口拒绝。这不宣称穷举了所有可能的 final-LS ties 或全部极端输入。
- Arm A 包装器相对原 C128 legacy STE 的前向 bytes、metadata 与 input gradients 保持；Dual-M 对所有已测输入的 cotangent 恒等。旧 C128 legacy STE 与 hard C128 虽未出现数值不等，真实 state 中却有 **551 个零符号位差异**：hard 的负零被 legacy 消减相加变为正零。两者不能声称逐字节相同，详见 `C128_HARD_VS_LEGACY_DISCLOSURE.json`；此披露不改变 Arm A 对历史 legacy 的一致性合同，也不改变 Dual-M 的真实 hard-forward 合同。
- 复用原 norm class、recurrent、`_out_hook`、`captured_losses` 和 state recovery 的 AST body，验证 24 层 theta 的 finite/nonzero gradients；原 `Functional + 0.1 State` 合成定义与 graph 已核验。
- 代表性单层/单 head 的原 H64 局部微测，两臂各三次 Adam(lr=.001, weight_decay=0)、clip=1 更新完成；最大正交误差 `3.814697265625e-06`，低于原 `1e-4`。
- 在合法 optimizer/cache-discard 边界保存 rotation、Adam 与 RNG，第三步恢复重放的 loss、gradient norm、参数和 optimizer hash、诊断输入与 teacher-target hash 完全相同。micro checkpoint 明确不能导出为正式科研旋转。

真实 fixture 来自已观察 Doc5 的 **C6R+Dual-M 存储坐标**，不是 clean native FP teacher state。q/k/v/gate 及 CPU norm/projection weights 是固定 seed 的局部诊断输入。它验证原 helper/局部计算图，不是完整 9B teacher/student 训练轨迹，也不是完整 1024-token/8-capture 文档更新。原 Loss 的完整模型训练资格仍为 NOT_TESTED；局部 loss 变化不构成科学收益证据。

`DUAL_M_FORWARD_PARITY.json` 和 `GRADIENT_QUALIFICATION_CPU.json` 保存实际结果。数据隔离、GPU admission 和任何付费执行的门禁必须独立满足，不能由 CPU 局部 PASS 替代。
