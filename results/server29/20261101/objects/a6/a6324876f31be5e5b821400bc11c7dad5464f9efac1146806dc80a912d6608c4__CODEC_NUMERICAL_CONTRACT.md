# Qwen Key Group32 数值合同 V1

冻结范围：B0=Key-H128+C128 Sym；B1=相同 H128+C32 Sym；B2=相同 H128+C32 Asym。三者输入均为 finite FP32 `[B,H,Key128,Value128]`，分组只沿 Key，参数不跨 batch/head/Value/layer。元数据尺度为 FP32。非连续输入按逻辑索引处理。

H128、q/k normalization 和 FP helper 完整继承 `src/canonical_h128.py` 中相同 AST。H 坐标的 state 是核心输出的直接持久 state，不在每次 QDQ 时重复旋转。历史 native FP state 小样本只为 CPU codec fixture 离线用 H.T @ S 转换一次。

## Sym B0/B1

每组 `safe=max(max(abs(x)), FP32(1e-12))`，`s=safe/127.0`，`r=round(x/s)`，`q=clip(r,-127,127)`，`decode=q*s`。eager 分步，RNE，FP32；B0 与 canonical 相同操作图，B1 仅组长不同。编码为真实 INT8。

原 canonical 使用浮点 codes，round 可保留 -0。转 INT8 会丢失符号，因此另存布尔 negative_zero_flags（仅代码为0且原舍入符号为负），decode 时恢复 -0。这是无损编码元数据，两种 Sym 一致使用；不是浮点 shadow，也不改变任何量化级。实际 flags 以 bool 每元素1字节保存，不声称只有1 bit。实际持久模型 carrier 仍是原 FP32 QDQ cache。

PyTorch 2.5.1 CPU 标量 /127 是 FP32 真除法；CUDA 的 CPU 标量除数优化使用 FP32 reciprocal 后乘法。CPU_CANONICAL 直接复现旧 CPU 数学；另有 CUDA_LEGACY_EMULATED 参考：RN32(safe * RN32(1/127))。该模拟来自源码，尚不是 GPU 证明，必须由 GPU codec-only 全字段 bit 比较确认。不能以朴素 CPU 除法和 CUDA 的 1 ULP 差异事后更改 B0。

## Asym B2

逐操作 RN32：`a=min(0,min(x)); b=max(0,max(x)); span=b-a; safe=max(span,FP32(1e-12)); s=safe / tensor_FP32(255)`。a/b 的零规范为 +0。除数是同设备、同形状 FP32 Tensor，以避开标量 reciprocal 优化。`z=clip(RNE(RN32(-a/s)),0,255)`，UINT8；`r=RNE(RN32(x/s))`；`q=clip(RN32(r+float32(z)),0,255)`，UINT8；先 int16 做 q-z，再转换 FP32 乘 s。重建的零为 +0。不使用 FP16、FMA、torch.compile 或融合修改。

零组 z=0、codes=0、decode=+0，scale=RN32(FP32(1e-12)/FP32(255))。正/负常数组包含零端点，按同一规则处理。跨度溢出、scale 非正/非 finite、level/decode 非 finite 均 FAIL，不降精度、不换规则。

## 严格比较与失败证据

编码字段 codes/scales/zp/negative-zero flags 和 decode，比较实际 dtype/shape/逐元素值及字节。GPU B0/B1 对照 CUDA semantic CPU 参考；B2 对照独立 NumPy逐组索引参考。合成边界另用精确 Fraction 舍入证据。Hadamard跨设备矩阵乘的误差与 codec bit 资格分开；GPU完整模型fresh同卡仍要求字节一致。

首次数值失败保存原输入、min/max/peak/span/scale/zero_ratio/levels/rounded/codes/zp/decoded 的完整 dtype/shape/raw bits。相关分支停止，不改阈值重试。共享身份/资源失败全停。此合同在 CPU 资格运行前落盘；GPU 始终默认不授权。

依据：[PyTorch CUDA division](https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu)、[CPU division](https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp)。来源推断仅用于预登记 GPU 预期，不替代真实 GPU gate。
